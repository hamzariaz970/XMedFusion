"""
HIL Fine-tuning Script
======================
Called by the /api/hil/finetune endpoint.
Downloads approved HIL reports + scans, generates disease labels from
doctor reports using KEYWORD_MAP, then fine-tunes the BioMedCLIP
classifier head.

Changes from v1:
  - Minimum sample guard (MIN_HIL_SAMPLES = 5)
  - Timestamped weight backups (not just one overwritten backup)
  - Pre-downloads images to local cache before training (no HTTP per epoch)
  - Auto-cleanup of caches older than 7 days
"""

import os
import json
import sys
import hashlib
import datetime
import shutil
import requests
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from io import BytesIO
import open_clip
from pathlib import Path

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
MIN_HIL_SAMPLES = 5
IMAGE_CACHE_DIR = os.path.join("data", "cache", "hil_images")
CACHE_MAX_AGE_DAYS = 7

# ---------------------------------------------------------
# DISEASES & KEYWORD MAPS (same as train_biomed_encoder_kg_agent.py)
# ---------------------------------------------------------
DISEASES = [
    "Cardiomegaly", "Pleural Effusion", "Edema", "Pneumothorax",
    "Infiltrate", "Consolidation", "Lung Opacity", "Nodule",
    "Atelectasis", "Fracture"
]

KEYWORD_MAP = {
    "Cardiomegaly": [
        "cardiomegaly", "cardiac enlargement", "heart is enlarged",
        "heart size is enlarged", "enlarged heart", "heart is large",
        "cardiac silhouette is enlarged",
    ],
    "Pleural Effusion": [
        "pleural effusion", "effusions", "costophrenic blunting",
    ],
    "Edema": [
        "edema", "pulmonary edema", "vascular congestion",
        "pulmonary congestion",
    ],
    "Pneumothorax": ["pneumothorax", "pneumothoraces"],
    "Infiltrate": [
        "infiltrate", "infiltrates", "airspace disease",
        "airspace opacity",
    ],
    "Consolidation": [
        "consolidation", "consolidations", "focal consolidation",
    ],
    "Lung Opacity": [
        "opacity", "opacities", "opacification", "haziness",
    ],
    "Nodule": [
        "nodule", "nodules", "mass", "granuloma", "pulmonary nodule",
    ],
    "Atelectasis": ["atelectasis", "atelectatic", "volume loss"],
    "Fracture": ["fracture", "fractures", "rib fracture"],
}

NEGATION_PHRASES = [
    "no ", "no evidence", "without ", "negative for", "free of",
    "clear of", "absent", "not ", "denies ", "ruled out",
    "resolution of", "resolved", "no definite", "no acute",
    "no focal", "no obvious", "no significant",
]


def extract_labels(report_text: str) -> np.ndarray:
    """Extract disease labels from doctor report using keyword matching."""
    report = report_text.lower()
    labels = np.zeros(len(DISEASES), dtype=np.float32)
    
    for idx, disease in enumerate(DISEASES):
        keywords = KEYWORD_MAP[disease]
        for k in keywords:
            if k in report:
                pos = report.find(k)
                context_start = max(0, pos - 120)
                context = report[context_start:pos]
                last_period = max(context.rfind('.'), context.rfind('!'), context.rfind('?'))
                if last_period >= 0:
                    context = context[last_period + 1:]
                negated = any(neg in context for neg in NEGATION_PHRASES)
                if not negated:
                    labels[idx] = 1.0
                    break
    return labels


# ---------------------------------------------------------
# Image Cache Utilities
# ---------------------------------------------------------
def download_hil_images(samples: list, cache_dir: str = IMAGE_CACHE_DIR) -> list:
    """Download all HIL images once to a local cache before training."""
    os.makedirs(cache_dir, exist_ok=True)
    local_samples = []
    downloaded = 0
    cached = 0
    
    for url, labels in samples:
        filename = hashlib.md5(url.encode()).hexdigest() + ".png"
        local_path = os.path.join(cache_dir, filename)
        
        if os.path.exists(local_path):
            cached += 1
        else:
            try:
                resp = requests.get(url, timeout=15)
                resp.raise_for_status()
                with open(local_path, "wb") as f:
                    f.write(resp.content)
                downloaded += 1
            except Exception as e:
                print(f"⚠️ Failed to download {url}: {e}")
                continue
        
        local_samples.append((local_path, labels))
    
    print(f"📥 Image cache: {downloaded} downloaded, {cached} cached, {len(local_samples)} total")
    return local_samples


def cleanup_old_caches(cache_dir: str = IMAGE_CACHE_DIR, max_age_days: int = CACHE_MAX_AGE_DAYS):
    """Remove cached images older than max_age_days."""
    if not os.path.exists(cache_dir):
        return
    cutoff = datetime.datetime.now().timestamp() - (max_age_days * 86400)
    removed = 0
    for f in os.listdir(cache_dir):
        fp = os.path.join(cache_dir, f)
        if os.path.isfile(fp) and os.path.getmtime(fp) < cutoff:
            os.remove(fp)
            removed += 1
    if removed:
        print(f"🧹 Cleaned up {removed} cached images older than {max_age_days} days")


# ---------------------------------------------------------
# Dataset & Model
# ---------------------------------------------------------
class HILDataset(Dataset):
    """Dataset for HIL fine-tuning from LOCAL cached files + labels."""
    
    def __init__(self, samples, preprocess):
        self.samples = samples  # list of (local_path, labels_array)
        self.preprocess = preprocess
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, labels = self.samples[idx]
        try:
            image = Image.open(path).convert("RGB")
            image = self.preprocess(image)
            return image, torch.tensor(labels, dtype=torch.float32)
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            return torch.zeros((3, 224, 224)), torch.tensor(labels, dtype=torch.float32)


class BioMedCLIPClassifierHead(nn.Module):
    """Must match the architecture in train_biomed_encoder_kg_agent.py"""
    def __init__(self, num_classes, input_dim=512):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        self.residual = nn.Linear(input_dim, num_classes)
    
    def forward(self, features):
        return self.head(features) + self.residual(features)


def run_hil_finetune(approved_scans: list, num_epochs: int = 10) -> dict:
    """
    Fine-tune the BioMedCLIP classifier head with approved HIL data.
    
    Args:
        approved_scans: list of dicts with 'image_url', 'findings', 'impression'
        num_epochs: number of training epochs
    
    Returns:
        dict with training results
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Prepare samples
    samples = []
    for scan in approved_scans:
        report_text = f"{scan.get('findings', '')} {scan.get('impression', '')}"
        labels = extract_labels(report_text)
        samples.append((scan['image_url'], labels))
    
    if len(samples) < MIN_HIL_SAMPLES:
        return {
            "error": f"Need at least {MIN_HIL_SAMPLES} approved reports to fine-tune. Got {len(samples)}.",
            "num_samples": len(samples),
        }
    
    print(f"📊 HIL Fine-tuning: {len(samples)} samples")
    
    # 1b. Pre-download all images to local cache
    cleanup_old_caches()
    local_samples = download_hil_images(samples)
    
    if len(local_samples) < MIN_HIL_SAMPLES:
        return {
            "error": f"Only {len(local_samples)} images downloaded successfully. Need at least {MIN_HIL_SAMPLES}.",
            "num_samples": len(local_samples),
        }
    
    # 2. Load model
    model_name = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    backbone, _, preprocess = open_clip.create_model_and_transforms(model_name)
    
    # Freeze backbone
    for param in backbone.parameters():
        param.requires_grad = False
    
    classifier = BioMedCLIPClassifierHead(len(DISEASES)).to(device)
    backbone = backbone.to(device)
    
    # Load existing weights if available
    weights_dir = "model_weights/KG_Agent/biomed_clip"
    head_path = os.path.join(weights_dir, "biomed_clip_head_best.pth")
    if os.path.exists(head_path):
        classifier.load_state_dict(torch.load(head_path, map_location=device))
        print(f"✅ Loaded existing classifier weights from {head_path}")
    
    # 3. Create dataset & loader (from local cache, not URLs)
    dataset = HILDataset(local_samples, preprocess)
    loader = DataLoader(dataset, batch_size=min(8, len(local_samples)), shuffle=True)
    
    # 4. Train (head only - fast)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(classifier.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    backbone.eval()
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        classifier.train()
        total_loss = 0
        
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            
            with torch.no_grad():
                features = backbone.encode_image(imgs)
                features = F.normalize(features, dim=-1)
            
            logits = classifier(features)
            loss = criterion(logits, lbls)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(loader)
        scheduler.step()
        
        print(f"  Epoch {epoch+1}/{num_epochs}: Loss={avg_loss:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            # Timestamped backup (preserves history)
            os.makedirs(weights_dir, exist_ok=True)
            if os.path.exists(head_path):
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = os.path.join(weights_dir, f"biomed_clip_head_backup_{timestamp}.pth")
                shutil.copy2(head_path, backup_path)
                print(f"  📦 Backed up previous weights → {backup_path}")
            
            torch.save(classifier.state_dict(), head_path)
    
    print(f"✅ HIL Fine-tuning complete! Best loss: {best_loss:.4f}")
    
    return {
        "status": "complete",
        "num_samples": len(local_samples),
        "epochs": num_epochs,
        "best_loss": round(best_loss, 4),
    }
