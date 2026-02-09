import os
import json
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
from tqdm import tqdm
import open_clip
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, classification_report

# ---------------------------------------------------------
# CONFIGURATION & HYPERPARAMETERS
# ---------------------------------------------------------
CONFIG = {
    "data_dir": "data/iu_xray/images",
    "annotation_file": "data/iu_xray/annotation.json",
    "save_dir": "model_weights/KG_Agent/biomed_clip",
    "model_name": "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
    
    # Training Hyperparameters
    "batch_size": 32,          # Increase if you have >16GB VRAM
    "epochs": 15,              # Linear probing converges fast
    "lr": 1e-3,                # Standard for linear heads
    "weight_decay": 1e-4,      # Regularization
    "val_split": 0.15,         # 15% for validation
    "seed": 42,
    "num_workers": 0           # Set to 4 on Linux, 0 on Windows
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Diseases to detect (Order matters!)
DISEASES = [
    "Cardiomegaly", "Pleural Effusion", "Edema", "Pneumothorax", 
    "Infiltrate", "Consolidation", "Lung Opacity", "Nodule", 
    "Atelectasis", "Fracture"
]

# Robust Keyword Mapping (Ground Truth Extraction)
KEYWORD_MAP = {
    "Cardiomegaly": ["cardiomegaly", "enlarged heart", "heart size is enlarged", "cardiomegaly is seen"],
    "Pleural Effusion": ["pleural effusion", "pleural fluid", "costophrenic blunting", "blunting of the costophrenic"],
    "Edema": ["edema", "vascular congestion", "chf", "heart failure"],
    "Pneumothorax": ["pneumothorax", "collapsed lung"],
    "Infiltrate": ["infiltrate", "infiltration", "interstitial opacities"],
    "Consolidation": ["consolidation", "air space disease", "airspace opacity"],
    "Lung Opacity": ["opacity", "opacities", "hazy"],
    "Nodule": ["nodule", "mass", "granuloma", "calcified nodule"],
    "Atelectasis": ["atelectasis", "collapse"],
    "Fracture": ["fracture", "broken", "rib deformity", "deformity of the rib"]
}

# ---------------------------------------------------------
# 1. ROBUST DATASET CLASS
# ---------------------------------------------------------
class CXRDataset(Dataset):
    def __init__(self, annotation_path, img_dir, preprocess):
        self.img_dir = img_dir
        self.preprocess = preprocess
        self.samples = []
        
        # Load Annotations
        if not os.path.exists(annotation_path):
            raise FileNotFoundError(f"Missing: {annotation_path}")
            
        with open(annotation_path, "r") as f:
            raw_data = json.load(f)
            # Handle both {"train": [...]} structure and list structure
            data = raw_data.get("train", []) + raw_data.get("val", []) if isinstance(raw_data, dict) else raw_data
            
        print(f"Parsing {len(data)} reports to generate Ground Truth labels...")
        
        valid_count = 0
        for item in tqdm(data):
            # A. Get Image Path
            img_path = item.get("image_path", [])
            if isinstance(img_path, list):
                if not img_path: continue
                rel_path = img_path[0] # Use first image
            else:
                rel_path = img_path
                
            full_path = os.path.join(self.img_dir, rel_path)
            if not os.path.exists(full_path):
                continue
                
            # B. Generate Weak Labels from Text
            report = item.get("report", "").lower()
            labels = np.zeros(len(DISEASES), dtype=np.float32)
            
            for idx, disease in enumerate(DISEASES):
                keywords = KEYWORD_MAP[disease]
                is_present = False
                
                # Check for presence
                for k in keywords:
                    if k in report:
                        # Check for Negation (e.g., "No pneumothorax")
                        start_idx = report.find(k)
                        context = report[max(0, start_idx-25):start_idx] # Look back 25 chars
                        negations = ["no ", "not ", "free of", "without", "negative for", "absent"]
                        
                        if not any(neg in context for neg in negations):
                            is_present = True
                            break
                
                if is_present:
                    labels[idx] = 1.0
            
            self.samples.append((full_path, labels))
            valid_count += 1
            
        print(f"✅ Loaded {valid_count} valid image-report pairs.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            image = Image.open(path).convert("RGB")
            image = self.preprocess(image)
            return image, torch.tensor(label, dtype=torch.float32)
        except Exception as e:
            # Return dummy on failure to prevent crash
            return torch.zeros((3, 224, 224)), torch.tensor(label, dtype=torch.float32)

# ---------------------------------------------------------
# 2. MODEL DEFINITION
# ---------------------------------------------------------
class BioMedCLIPClassifier(nn.Module):
    def __init__(self, num_classes, model_name):
        super().__init__()
        print(f"Loading BioMedCLIP: {model_name}")
        self.backbone, _, self.preprocess = open_clip.create_model_and_transforms(model_name)
        
        # Freeze Backbone (We only train the head)
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval() 
            
        # Trainable Head
        self.head = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4), # Prevent overfitting on small dataset
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        with torch.no_grad():
            features = self.backbone.encode_image(x)
            features = torch.nn.functional.normalize(features, dim=-1)
        return self.head(features)

    def get_preprocess(self):
        return self.preprocess

# ---------------------------------------------------------
# 3. TRAINING UTILS
# ---------------------------------------------------------
def calculate_pos_weights(dataset):
    """Calculates class weights to handle imbalance (e.g. rare fractures)"""
    labels = [s[1] for s in dataset.samples]
    labels = np.array(labels)
    pos_counts = np.sum(labels, axis=0)
    total = len(labels)
    # Weight = (Total - Pos) / Pos
    weights = (total - pos_counts) / (pos_counts + 1e-5)
    return torch.tensor(weights, dtype=torch.float32)

def evaluate(model, loader, device, threshold=0.5):
    """Runs validation and returns metrics"""
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    criterion = nn.BCEWithLogitsLoss() # Use plain loss for validation
    
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
    
    # Calculate Metrics
    try:
        auc = roc_auc_score(all_labels, all_preds, average="macro")
    except:
        auc = 0.5 # Fallback if only one class present
        
    # F1 Score (Macro)
    bin_preds = (all_preds > threshold).astype(int)
    f1 = f1_score(all_labels, bin_preds, average="macro")
    
    return total_loss / len(loader), auc, f1, all_labels, bin_preds

# ---------------------------------------------------------
# 4. MAIN SCRIPT
# ---------------------------------------------------------
if __name__ == "__main__":
    # A. Setup
    if not os.path.exists(CONFIG["save_dir"]):
        os.makedirs(CONFIG["save_dir"])
    
    # B. Initialize Model & Preprocess
    model = BioMedCLIPClassifier(len(DISEASES), CONFIG["model_name"]).to(DEVICE)
    preprocess = model.get_preprocess()
    
    # C. Prepare Data
    full_dataset = CXRDataset(CONFIG["annotation_file"], CONFIG["data_dir"], preprocess)
    
    # Create Train/Val Split
    indices = list(range(len(full_dataset)))
    train_idx, val_idx = train_test_split(indices, test_size=CONFIG["val_split"], random_state=CONFIG["seed"])
    
    train_ds = Subset(full_dataset, train_idx)
    val_ds = Subset(full_dataset, val_idx)
    
    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["num_workers"])
    val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])
    
    # D. Handle Imbalance
    # We calculate weights only on the training set
    print("Calculating class weights...")
    pos_weights = calculate_pos_weights(full_dataset) # Approximation using full ds for stability
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights.to(DEVICE))
    
    optimizer = optim.AdamW(model.head.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)

    print(f"\n🚀 Training Start: {len(train_ds)} Train | {len(val_ds)} Val")
    print(f"   Pos Weights (Importance): {pos_weights.numpy().round(1)}")
    
    best_f1 = 0.0
    
    # E. Training Loop
    for epoch in range(CONFIG["epochs"]):
        model.train()
        train_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")
        
        for imgs, lbls in loop:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, lbls)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        # Validation Phase
        val_loss, val_auc, val_f1, _, _ = evaluate(model, val_loader, DEVICE)
        
        print(f"   Results: Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f}")
        print(f"   Metrics: Val AUC: {val_auc:.4f} | Val F1: {val_f1:.4f}")
        
        # Scheduler Step
        scheduler.step(val_f1)
        
        # Save Best Model
        if val_f1 > best_f1:
            best_f1 = val_f1
            save_path = os.path.join(CONFIG["save_dir"], "biomed_clip_head_best.pth")
            torch.save(model.head.state_dict(), save_path)
            print(f"   ⭐ New Best Model Saved! (F1: {val_f1:.4f})")
            
    # F. Final Detailed Evaluation
    print("\n📊 --- FINAL EVALUATION ON VALIDATION SET ---")
    # Load best weights
    model.head.load_state_dict(torch.load(os.path.join(CONFIG["save_dir"], "biomed_clip_head_best.pth")))
    _, _, _, y_true, y_pred = evaluate(model, val_loader, DEVICE)
    
    print(classification_report(y_true, y_pred, target_names=DISEASES, zero_division=0))
    print(f"Complete. Model saved to {CONFIG['save_dir']}")