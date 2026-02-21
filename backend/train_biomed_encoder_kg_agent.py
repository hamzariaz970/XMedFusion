"""
KG Agent BioMedCLIP Classifier Training Script
==============================================
Two-phase training:
  Phase 1: Linear probing (frozen backbone, trains head only)
  Phase 2: Fine-tuning (unfreeze last N ViT blocks)

Architecture must match kg_agent.py and vision.py BioMedCLIPClassifierHead.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
from tqdm import tqdm
import open_clip
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, classification_report
from torchvision import transforms

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
CONFIG = {
    "data_dir": "data/iu_xray/images",
    "annotation_file": "data/iu_xray/annotation.json",
    "save_dir": "model_weights/KG_Agent/biomed_clip",
    "model_name": "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
    
    "batch_size": 32,
    "head_lr": 1e-3,
    "backbone_lr": 1e-5,
    "weight_decay": 1e-4,
    "val_split": 0.15,
    "seed": 42,
    "num_workers": 0,
    "unfreeze_blocks": 2,
    
    # Focal loss (Phase 2 only)
    "focal_alpha": 0.75,
    "focal_gamma": 2.0,
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------
# DISEASES & KEYWORD MAPS
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
        "heart is moderately enlarged", "heart is mildly enlarged",
        "heart is severely enlarged", "cardiac silhouette is enlarged",
        "cardiomediastinal silhouette is enlarged",
    ],
    "Pleural Effusion": [
        "pleural effusion", "effusions", "costophrenic blunting",
        "blunting of the costophrenic",
    ],
    "Edema": [
        "edema", "pulmonary edema", "vascular congestion",
        "pulmonary congestion", "fluid overload",
    ],
    "Pneumothorax": [
        "pneumothorax", "pneumothoraces",
    ],
    "Infiltrate": [
        "infiltrate", "infiltrates", "airspace disease",
        "airspace opacity", "air space opacity", "air space opacities",
        "airspace consolidation",
    ],
    "Consolidation": [
        "consolidation", "consolidations", "focal consolidation",
        "lobar consolidation",
    ],
    "Lung Opacity": [
        "opacity", "opacities", "opacification", "haziness",
        "hazy opacity", "airspace opacity",
    ],
    "Nodule": [
        "nodule", "nodules", "mass", "granuloma", "calcified nodule",
        "pulmonary nodule", "lung nodule", "calcified granuloma",
    ],
    "Atelectasis": [
        "atelectasis", "atelectatic", "volume loss",
    ],
    "Fracture": [
        "fracture", "fractures", "rib fracture",
    ],
}

NEGATION_WINDOW = 120  # Must cover long comma-separated lists in radiology reports
NEGATION_PHRASES = [
    "no ", "no evidence", "without ", "negative for", "free of",
    "clear of", "absent", "not ", "denies ", "ruled out",
    "resolution of", "resolved", "removed", "no definite",
    "no definitive", "no visible", "no acute", "no focal",
    "no large", "no obvious", "no significant", "no suspicious",
    "no displaced", "nondisplaced",
]


# ---------------------------------------------------------
# DATASET
# ---------------------------------------------------------
class CXRDataset(Dataset):
    def __init__(self, annotation_path, img_dir, preprocess, augment=False):
        self.img_dir = img_dir
        self.preprocess = preprocess
        self.augment = augment
        self.samples = []
        
        # Simple augmentations (no RandomErasing - it corrupts normalized tensors)
        self.aug_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
        ])
        
        with open(annotation_path, "r") as f:
            raw_data = json.load(f)
            data = raw_data.get("train", []) + raw_data.get("val", []) if isinstance(raw_data, dict) else raw_data
            
        print(f"Parsing {len(data)} reports to generate Ground Truth labels...")
        
        for item in tqdm(data):
            img_path = item.get("image_path", [])
            if isinstance(img_path, list):
                if not img_path: continue
                rel_path = img_path[0]
            else:
                rel_path = img_path
                
            full_path = os.path.join(self.img_dir, rel_path)
            if not os.path.exists(full_path):
                continue
                
            report = item.get("report", "").lower()
            labels = np.zeros(len(DISEASES), dtype=np.float32)
            
            for idx, disease in enumerate(DISEASES):
                keywords = KEYWORD_MAP[disease]
                is_present = False
                
                for k in keywords:
                    if k in report:
                        start = 0
                        while True:
                            pos = report.find(k, start)
                            if pos == -1:
                                break
                            
                            context_start = max(0, pos - NEGATION_WINDOW)
                            context = report[context_start:pos]
                            
                            # Sentence boundary
                            last_period = max(context.rfind('.'), context.rfind('!'), context.rfind('?'))
                            if last_period >= 0:
                                context = context[last_period + 1:]
                            
                            negated = any(neg in context for neg in NEGATION_PHRASES)
                            
                            if not negated:
                                is_present = True
                                break
                            
                            start = pos + len(k)
                        
                        if is_present:
                            break
                
                if is_present:
                    labels[idx] = 1.0
            
            self.samples.append((full_path, labels))
            
        print(f"✅ Loaded {len(self.samples)} valid image-report pairs.")
        
        all_labels = np.array([s[1] for s in self.samples])
        print("📊 Label distribution:")
        for idx, disease in enumerate(DISEASES):
            count = int(all_labels[:, idx].sum())
            pct = 100 * count / len(self.samples)
            print(f"   {disease:>20s}: {count:4d} ({pct:5.1f}%)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            image = Image.open(path).convert("RGB")
            
            # Apply augmentation BEFORE preprocess (on PIL image)
            if self.augment:
                image = self.aug_transform(image)
            
            image = self.preprocess(image)
            return image, torch.tensor(label, dtype=torch.float32)
        except Exception:
            return torch.zeros((3, 224, 224)), torch.tensor(label, dtype=torch.float32)


# ---------------------------------------------------------
# MODEL
# ---------------------------------------------------------
class BioMedCLIPClassifierHead(nn.Module):
    """Deeper classifier head with residual connection. Must match kg_agent.py and vision.py."""
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
        
        # Zero-init residual so it starts as a no-op
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    def forward(self, features):
        return self.head(features) + self.residual(features)


class BioMedCLIPClassifier(nn.Module):
    """Full model: BioMedCLIP backbone + classifier head."""
    
    def __init__(self, num_classes, model_name, unfreeze_blocks=0):
        super().__init__()
        print(f"Loading BioMedCLIP: {model_name}")
        self.backbone, _, self.preprocess = open_clip.create_model_and_transforms(model_name)
        
        # Freeze everything first
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Unfreeze last N ViT blocks
        if unfreeze_blocks > 0:
            self._unfreeze_blocks(unfreeze_blocks)
        
        # Classifier head
        self.classifier = BioMedCLIPClassifierHead(num_classes)
    
    def _unfreeze_blocks(self, n):
        visual = self.backbone.visual
        blocks = None
        if hasattr(visual, 'trunk') and hasattr(visual.trunk, 'blocks'):
            blocks = visual.trunk.blocks
        elif hasattr(visual, 'transformer') and hasattr(visual.transformer, 'resblocks'):
            blocks = visual.transformer.resblocks
        elif hasattr(visual, 'blocks'):
            blocks = visual.blocks
        
        if blocks is not None:
            total = len(blocks)
            start = max(0, total - n)
            for i in range(start, total):
                for param in blocks[i].parameters():
                    param.requires_grad = True
            print(f"🔓 Unfroze ViT blocks [{start}:{total}] ({n} blocks)")
            
            if hasattr(visual, 'trunk') and hasattr(visual.trunk, 'norm'):
                for param in visual.trunk.norm.parameters():
                    param.requires_grad = True
    
    def get_preprocess(self):
        return self.preprocess
    
    def get_param_groups(self, head_lr, backbone_lr):
        head_params = list(self.classifier.parameters())
        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        groups = [{"params": head_params, "lr": head_lr}]
        if backbone_params:
            groups.append({"params": backbone_params, "lr": backbone_lr})
        return groups

    def forward(self, x):
        with torch.set_grad_enabled(any(p.requires_grad for p in self.backbone.parameters())):
            features = self.backbone.encode_image(x)
            features = F.normalize(features, dim=-1)
        return self.classifier(features)


# ---------------------------------------------------------
# FOCAL LOSS (Phase 2 only)
# ---------------------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()


# ---------------------------------------------------------
# EVALUATION (consistent loss calculation)
# ---------------------------------------------------------
def evaluate(model, loader, device, criterion=None):
    """Evaluate using the SAME criterion as training for consistent val loss."""
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0
    
    if criterion is None:
        criterion = nn.BCEWithLogitsLoss()
    
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            logits = model(imgs)
            loss = criterion(logits, lbls)
            total_loss += loss.item()
            
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.append(probs)
            all_labels.append(lbls.cpu().numpy())
            
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    try:
        auc = roc_auc_score(all_labels, all_preds, average="macro")
    except:
        auc = 0.5
    
    # Use per-disease optimal thresholding for F1
    best_f1s = []
    for i in range(all_labels.shape[1]):
        best_f1 = 0
        for t in [0.3, 0.4, 0.5]:
            f1 = f1_score(all_labels[:, i], (all_preds[:, i] > t).astype(int), zero_division=0)
            best_f1 = max(best_f1, f1)
        best_f1s.append(best_f1)
    
    macro_f1 = np.mean(best_f1s)
    
    return total_loss / len(loader), auc, macro_f1, all_preds, all_labels


def optimize_thresholds(all_preds, all_labels, diseases):
    """Per-disease threshold optimization."""
    thresholds = {}
    print("\n🎯 Optimizing per-disease thresholds on validation set...")
    
    for idx, disease in enumerate(diseases):
        best_thresh = 0.5
        best_f1 = 0.0
        
        for t in np.arange(0.10, 0.91, 0.05):
            preds = (all_preds[:, idx] > t).astype(int)
            f1 = f1_score(all_labels[:, idx], preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = float(round(t, 2))
        
        thresholds[disease] = best_thresh
        print(f"   {disease:>20s}: threshold={best_thresh:.2f}  (F1={best_f1:.3f})")
    
    return thresholds


# ---------------------------------------------------------
# TRAINING HELPERS
# ---------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device, dataset=None):
    model.train()
    if dataset:
        dataset.augment = True
    total_loss = 0
    for imgs, lbls in loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, lbls)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def save_checkpoint(model, save_dir):
    head_path = os.path.join(save_dir, "biomed_clip_head_best.pth")
    torch.save(model.classifier.state_dict(), head_path)
    
    backbone_path = os.path.join(save_dir, "biomed_clip_backbone_finetuned.pth")
    unfrozen = {}
    for name, param in model.backbone.named_parameters():
        if param.requires_grad:
            unfrozen[name] = param.data.clone()
    if unfrozen:
        torch.save(unfrozen, backbone_path)


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
if __name__ == "__main__":
    os.makedirs(CONFIG["save_dir"], exist_ok=True)
    
    # A. Model (start with frozen backbone)
    model = BioMedCLIPClassifier(
        len(DISEASES), CONFIG["model_name"], unfreeze_blocks=0
    ).to(DEVICE)
    preprocess = model.get_preprocess()
    
    # B. Dataset
    full_dataset = CXRDataset(
        CONFIG["annotation_file"], CONFIG["data_dir"], preprocess, augment=False
    )
    
    indices = list(range(len(full_dataset)))
    train_idx, val_idx = train_test_split(
        indices, test_size=CONFIG["val_split"], random_state=CONFIG["seed"]
    )
    train_ds = Subset(full_dataset, train_idx)
    val_ds = Subset(full_dataset, val_idx)
    
    train_loader = DataLoader(
        train_ds, batch_size=CONFIG["batch_size"],
        shuffle=True, num_workers=CONFIG["num_workers"], pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=CONFIG["batch_size"],
        shuffle=False, num_workers=CONFIG["num_workers"], pin_memory=True
    )
    
    # =========================================================
    # PHASE 1: Linear Probing — NO pos_weight, plain BCE
    # =========================================================
    print(f"\n{'='*60}")
    print(f"📌 PHASE 1: Linear Probing (Frozen Backbone)")
    print(f"   Epochs: 20 | LR: {CONFIG['head_lr']} | Loss: Plain BCE")
    print(f"   Train: {len(train_ds)} | Val: {len(val_ds)}")
    print(f"{'='*60}")
    
    criterion_p1 = nn.BCEWithLogitsLoss()  # No pos_weight — keeps loss stable
    optimizer_p1 = optim.AdamW(
        model.classifier.parameters(), lr=CONFIG["head_lr"], weight_decay=CONFIG["weight_decay"]
    )
    scheduler_p1 = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_p1, mode='max', factor=0.5, patience=4, verbose=True
    )
    
    best_f1 = 0.0
    patience = 0
    
    for epoch in range(20):
        train_loss = train_one_epoch(
            model, train_loader, criterion_p1, optimizer_p1, DEVICE, full_dataset
        )
        
        full_dataset.augment = False
        val_loss, val_auc, val_f1, _, _ = evaluate(model, val_loader, DEVICE, criterion_p1)
        scheduler_p1.step(val_f1)
        
        lr = optimizer_p1.param_groups[0]['lr']
        print(f"  P1 Epoch {epoch+1:2d}: Train={train_loss:.4f} | Val={val_loss:.4f} | AUC={val_auc:.4f} | F1={val_f1:.4f} | LR={lr:.2e}")
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience = 0
            save_checkpoint(model, CONFIG["save_dir"])
            print(f"  ⭐ New Best! (F1: {val_f1:.4f})")
        else:
            patience += 1
            if patience >= 6:
                print(f"  ⏹️ Phase 1 early stop at epoch {epoch+1}")
                break
    
    print(f"\n✅ Phase 1 complete. Best F1: {best_f1:.4f}")
    
    # Reload best P1 weights
    model.classifier.load_state_dict(
        torch.load(os.path.join(CONFIG["save_dir"], "biomed_clip_head_best.pth"), map_location=DEVICE)
    )
    
    # =========================================================
    # PHASE 2: Fine-tune backbone + head with Focal Loss
    # =========================================================
    print(f"\n{'='*60}")
    print(f"🔓 PHASE 2: Fine-tuning (Last {CONFIG['unfreeze_blocks']} ViT Blocks)")
    print(f"   Head LR: {CONFIG['backbone_lr']} | Backbone LR: {CONFIG['backbone_lr']/5}")
    print(f"   Loss: Focal (γ={CONFIG['focal_gamma']}, α={CONFIG['focal_alpha']})")
    print(f"{'='*60}")
    
    # Unfreeze backbone
    model._unfreeze_blocks(CONFIG["unfreeze_blocks"])
    
    criterion_p2 = FocalLoss(alpha=CONFIG["focal_alpha"], gamma=CONFIG["focal_gamma"])
    
    param_groups = model.get_param_groups(
        head_lr=CONFIG["backbone_lr"],       # 1e-5 for head
        backbone_lr=CONFIG["backbone_lr"]/5   # 2e-6 for backbone
    )
    optimizer_p2 = optim.AdamW(param_groups, weight_decay=CONFIG["weight_decay"])
    scheduler_p2 = optim.lr_scheduler.CosineAnnealingLR(optimizer_p2, T_max=15, eta_min=1e-7)
    
    patience = 0
    
    for epoch in range(15):
        train_loss = train_one_epoch(
            model, train_loader, criterion_p2, optimizer_p2, DEVICE, full_dataset
        )
        
        full_dataset.augment = False
        val_loss, val_auc, val_f1, _, _ = evaluate(model, val_loader, DEVICE, criterion_p2)
        scheduler_p2.step()
        
        lrs = [f"{g['lr']:.2e}" for g in optimizer_p2.param_groups]
        print(f"  P2 Epoch {epoch+1:2d}: Train={train_loss:.4f} | Val={val_loss:.4f} | AUC={val_auc:.4f} | F1={val_f1:.4f} | LRs={lrs}")
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience = 0
            save_checkpoint(model, CONFIG["save_dir"])
            print(f"  ⭐ New Best! (F1: {val_f1:.4f})")
        else:
            patience += 1
            if patience >= 6:
                print(f"  ⏹️ Phase 2 early stop at epoch {epoch+1}")
                break
    
    print(f"\n✅ Phase 2 complete. Best F1: {best_f1:.4f}")
    
    # =========================================================
    # FINAL EVALUATION & THRESHOLDS
    # =========================================================
    print("\n📊 --- FINAL EVALUATION ---")
    best_path = os.path.join(CONFIG["save_dir"], "biomed_clip_head_best.pth")
    model.classifier.load_state_dict(torch.load(best_path, map_location=DEVICE))
    
    backbone_path = os.path.join(CONFIG["save_dir"], "biomed_clip_backbone_finetuned.pth")
    if os.path.exists(backbone_path):
        unfrozen = torch.load(backbone_path, map_location=DEVICE)
        for name, param in model.backbone.named_parameters():
            if name in unfrozen:
                param.data = unfrozen[name]
    
    full_dataset.augment = False
    _, _, _, all_preds, all_labels = evaluate(model, val_loader, DEVICE)
    
    bin_preds = (all_preds > 0.5).astype(int)
    print(classification_report(all_labels, bin_preds, target_names=DISEASES, zero_division=0))
    
    thresholds = optimize_thresholds(all_preds, all_labels, DISEASES)
    
    threshold_path = os.path.join(CONFIG["save_dir"], "thresholds.json")
    with open(threshold_path, "w") as f:
        json.dump(thresholds, f, indent=2)
    print(f"\n✅ Thresholds saved to {threshold_path}")
    
    print(f"\n{'='*50}")
    print(f"Training Complete!")
    print(f"   Best Val F1: {best_f1:.4f}")
    print(f"   Model: {CONFIG['save_dir']}/biomed_clip_head_best.pth")
    print(f"   Backbone: {CONFIG['save_dir']}/biomed_clip_backbone_finetuned.pth")
    print(f"   Thresholds: {threshold_path}")
    print(f"{'='*50}")
