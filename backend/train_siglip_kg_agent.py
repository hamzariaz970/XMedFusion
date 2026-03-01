"""
train_siglip_kg_agent.py
--------------------------------------------------
Trains a multi-label disease classifier using MedGemma's built-in
SigLIP vision encoder as the frozen (then fine-tuned) backbone.

Two-phase training (same as train_biomed_encoder_kg_agent.py):
  Phase 1: Linear probe  — only the classification head is trained
  Phase 2: Fine-tuning   — last N transformer blocks + head are unfrozen

Multi-view support: each study may have 2 X-ray views (frontal + lateral).
Each view is encoded independently; valid-view mean pooling is applied before
the classification head.

Labels: 14-class CheXpert schema from data/iu_xray/llm_annotations.json
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, classification_report
from torchvision import transforms
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

from config import HF_TOKEN
os.environ["HF_TOKEN"] = HF_TOKEN
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
CONFIG = {
    "medgemma_id":    "google/medgemma-4b-it",
    "annotation_file":"data/iu_xray/llm_annotations.json",
    "img_dir":        "data/iu_xray/images",
    "save_dir":       "model_weights/KG_Agent/siglip_classifier",

    "batch_size":     16,   # 16 studies/batch — SigLIP encoder only (no LLM), very light
    "head_lr":        1e-3,
    "backbone_lr":    1e-5,
    "weight_decay":   1e-4,
    "num_workers":    0,
    "max_views":      2,    # frontal + lateral
    "unfreeze_blocks":4,    # last N ViT blocks unfrozen in phase 2
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DISEASES = [
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
    "Edema", "Consolidation", "Pneumonia", "Atelectasis",
    "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture",
    "Support Devices", "No Finding",
]
NUM_CLASSES = len(DISEASES)


# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────────
class CXRDataset(Dataset):
    def __init__(self, data, img_dir, processor, augment=False):
        """
        data      : filtered list of dicts from llm_annotations.json
        img_dir   : base image directory
        processor : MedGemma AutoProcessor (handles SigLIP pixel normalisation)
        augment   : random augmentation during training
        """
        self.img_dir   = img_dir
        self.processor = processor
        self.augment   = augment
        self.max_views = CONFIG["max_views"]
        self.samples   = []   # list of (list[str paths], np.array labels)

        self.aug_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
        ])

        for item in data:
            img_paths = item.get("image_path", [])
            if isinstance(img_paths, str):
                img_paths = [img_paths]
            if not img_paths:
                continue

            llm    = item.get("llm_labels", {})
            labels = np.array([float(llm.get(d, 0)) for d in DISEASES], dtype=np.float32)

            valid_paths = [
                os.path.join(img_dir, p) for p in img_paths
                if os.path.exists(os.path.join(img_dir, p))
            ]
            if valid_paths:
                self.samples.append((valid_paths, labels))

        print(f"  Loaded {len(self.samples)} patient studies.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        paths, label = self.samples[idx]
        pixel_values_list = []
        valid_mask        = []

        for path in paths[:self.max_views]:
            try:
                img = Image.open(path).convert("RGB")
                if self.augment:
                    img = self.aug_transform(img)
                # Use MedGemma's processor for SigLIP-correct normalisation
                enc = self.processor(images=img, return_tensors="pt")
                pixel_values_list.append(enc["pixel_values"].squeeze(0))  # [C, H, W]
                valid_mask.append(1.0)
            except Exception:
                pixel_values_list.append(torch.zeros(3, 224, 224))
                valid_mask.append(0.0)

        # Pad to max_views
        while len(pixel_values_list) < self.max_views:
            pixel_values_list.append(torch.zeros_like(pixel_values_list[0]))
            valid_mask.append(0.0)

        # Edge-case: all views failed
        if sum(valid_mask) == 0:
            valid_mask[0] = 1.0

        images = torch.stack(pixel_values_list)          # [max_views, C, H, W]
        mask   = torch.tensor(valid_mask, dtype=torch.float32)  # [max_views]
        return images, mask, torch.tensor(label, dtype=torch.float32)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL  — SigLIP backbone + classification head
# ─────────────────────────────────────────────────────────────────────────────
class SigLIPClassifier(nn.Module):
    def __init__(self, medgemma_id, num_classes, unfreeze_blocks=0):
        super().__init__()
        print(f"Loading MedGemma vision tower from {medgemma_id} ...")

        # Load full MedGemma in bfloat16 and extract only the vision tower
        full_model = AutoModelForImageTextToText.from_pretrained(
            medgemma_id,
            torch_dtype=torch.bfloat16,
            device_map="cpu",   # keep on CPU first; move to GPU after extraction
        )
        self.vision_tower = full_model.model.vision_tower
        # Infer feature dim from the tower's config
        self.feature_dim  = self.vision_tower.config.hidden_size
        del full_model  # free memory from LLM weights

        # Freeze entire backbone by default
        for p in self.vision_tower.parameters():
            p.requires_grad = False

        if unfreeze_blocks > 0:
            self._unfreeze_last_blocks(unfreeze_blocks)

        # Simple 2-layer classification head
        self.head = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes),
        )

    def _unfreeze_last_blocks(self, n):
        """Unfreeze the last n transformer blocks of the SigLIP ViT."""
        vit = self.vision_tower.vision_model
        # SigLIP ViT stores blocks in encoder.layers
        layers = vit.encoder.layers
        total  = len(layers)
        start  = max(0, total - n)
        for layer in layers[start:]:
            for p in layer.parameters():
                p.requires_grad = True
        # Also unfreeze the final LayerNorm
        for p in vit.post_layernorm.parameters():
            p.requires_grad = True
        print(f"  Unfroze SigLIP blocks [{start}:{total}] ({n} blocks) + post_layernorm")

    def encode_image(self, pixel_values):
        """
        pixel_values: [B, C, H, W] float
        Returns CLS-pooled feature: [B, feature_dim]
        """
        out = self.vision_tower(
            pixel_values=pixel_values.to(dtype=torch.bfloat16),
            interpolate_pos_encoding=True,   # handles any resolution safely
        )
        # SigLIP returns pooler_output (CLS token) as the compact representation
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            return out.pooler_output.float()
        # Fallback: mean-pool over patch tokens
        return out.last_hidden_state.mean(dim=1).float()

    def forward(self, images, mask):
        """
        images : [B, V, C, H, W]  (V = max_views)
        mask   : [B, V]            (1 = valid view; 0 = padding)
        """
        B, V, C, H, W = images.shape
        flat_imgs = images.view(B * V, C, H, W)

        with torch.set_grad_enabled(
            any(p.requires_grad for p in self.vision_tower.parameters())
        ):
            feats = self.encode_image(flat_imgs)          # [B*V, D]

        feats = feats.view(B, V, -1)                      # [B, V, D]
        feats = feats * mask.unsqueeze(-1)                 # zero-out invalid views

        valid_counts = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        pooled = feats.sum(dim=1) / valid_counts          # [B, D]
        pooled = F.normalize(pooled, dim=-1)

        return self.head(pooled)                           # [B, num_classes]

    def get_param_groups(self, head_lr, backbone_lr):
        head_params     = list(self.head.parameters())
        backbone_params = [p for p in self.vision_tower.parameters() if p.requires_grad]
        groups = [{"params": head_params, "lr": head_lr}]
        if backbone_params:
            groups.append({"params": backbone_params, "lr": backbone_lr})
        return groups


# ─────────────────────────────────────────────────────────────────────────────
# ASYMMETRIC LOSS  (same as biomed encoder)
# ─────────────────────────────────────────────────────────────────────────────
class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_neg, self.gamma_pos = gamma_neg, gamma_pos
        self.clip, self.eps = clip, eps

    def forward(self, x, y):
        xs = torch.sigmoid(x)
        xs_neg = (1 - xs + self.clip).clamp(max=1)
        loss = y * torch.log(xs.clamp(min=self.eps)) + \
               (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        if self.gamma_pos > 0 or self.gamma_neg > 0:
            pt   = xs * y + (1 - xs) * (1 - y)
            w    = torch.pow(1 - pt, self.gamma_pos * y + self.gamma_neg * (1 - y))
            loss = loss * w
        return -loss.mean()


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(model, loader, device, criterion=None):
    model.eval()
    if criterion is None:
        criterion = nn.BCEWithLogitsLoss()
    total_loss, all_preds, all_labels = 0.0, [], []

    with torch.no_grad():
        for imgs, masks, lbls in loader:
            imgs, masks, lbls = imgs.to(device), masks.to(device), lbls.to(device)
            logits = model(imgs, masks)
            total_loss += criterion(logits, lbls).item()
            all_preds.append(torch.sigmoid(logits).cpu().numpy())
            all_labels.append(lbls.cpu().numpy())

    all_preds  = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    aucs = [
        roc_auc_score(all_labels[:, i], all_preds[:, i])
        for i in range(all_labels.shape[1])
        if len(np.unique(all_labels[:, i])) > 1
    ]
    macro_auc = float(np.mean(aucs)) if aucs else 0.5

    f1s = [
        max(f1_score(all_labels[:, i], (all_preds[:, i] > t).astype(int), zero_division=0)
            for t in [0.3, 0.4, 0.5])
        for i in range(all_labels.shape[1])
    ]
    macro_f1 = float(np.mean(f1s))

    return total_loss / len(loader), macro_auc, macro_f1, all_preds, all_labels


def train_one_epoch(model, loader, criterion, optimizer, device, dataset=None):
    model.train()
    if dataset:
        dataset.augment = True
    total_loss = 0.0
    for imgs, masks, lbls in loader:
        imgs, masks, lbls = imgs.to(device), masks.to(device), lbls.to(device)
        optimizer.zero_grad()
        loss = criterion(model(imgs, masks), lbls)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def save_checkpoint(model, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    head_path = os.path.join(save_dir, "siglip_head_best.pth")
    torch.save(model.head.state_dict(), head_path)

    backbone_path = os.path.join(save_dir, "siglip_backbone_finetuned.pth")
    unfrozen = {n: p.data.clone() for n, p in model.vision_tower.named_parameters()
                if p.requires_grad}
    if unfrozen:
        torch.save(unfrozen, backbone_path)
    print(f"  Checkpoint saved → {save_dir}")


def optimize_thresholds(all_preds, all_labels):
    thresholds = {}
    print("\nOptimising per-disease thresholds on validation set...")
    for i, disease in enumerate(DISEASES):
        best_t, best_f1 = 0.5, 0.0
        for t in np.arange(0.01, 1.0, 0.02):
            f1 = f1_score(all_labels[:, i], (all_preds[:, i] > t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, float(round(t, 2))
        thresholds[disease] = best_t
        print(f"   {disease:>30s}  threshold={best_t:.2f}  (F1={best_f1:.3f})")
    return thresholds


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(CONFIG["save_dir"], exist_ok=True)

    # ── Processor (SigLIP preprocessing) ───────────────────────────────────
    processor = AutoProcessor.from_pretrained(CONFIG["medgemma_id"])

    # ── Load data ───────────────────────────────────────────────────────────
    print(f"Loading {CONFIG['annotation_file']} ...")
    with open(CONFIG["annotation_file"], "r") as f:
        all_data = json.load(f)

    train_data = [x for x in all_data if x.get("split") == "train"]
    val_data   = [x for x in all_data if x.get("split") in ("val", "test")]
    print(f"  Train: {len(train_data)}  |  Val: {len(val_data)}")

    train_dataset = CXRDataset(train_data, CONFIG["img_dir"], processor, augment=False)
    val_dataset   = CXRDataset(val_data,   CONFIG["img_dir"], processor, augment=False)

    # ── Class-balanced pos_weight ───────────────────────────────────────────
    train_labels = np.array([s[1] for s in train_dataset.samples])
    pos_counts   = train_labels.sum(axis=0)
    neg_counts   = len(train_dataset) - pos_counts
    pos_weight   = torch.tensor(
        np.sqrt(neg_counts / np.clip(pos_counts, 1, None)), dtype=torch.float32
    ).to(DEVICE)

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"],
                              shuffle=True, num_workers=CONFIG["num_workers"], pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=CONFIG["batch_size"],
                              shuffle=False, num_workers=CONFIG["num_workers"], pin_memory=True)

    # ── Model ───────────────────────────────────────────────────────────────
    model = SigLIPClassifier(CONFIG["medgemma_id"], NUM_CLASSES, unfreeze_blocks=0).to(DEVICE)

    # ── PHASE 1: Linear probing (frozen backbone) ───────────────────────────
    print("\n" + "="*50)
    print("PHASE 1 — Linear Probing (frozen SigLIP backbone)")
    print("="*50)
    criterion_p1 = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer_p1 = optim.AdamW(model.head.parameters(),
                               lr=CONFIG["head_lr"], weight_decay=CONFIG["weight_decay"])
    scheduler_p1 = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_p1, mode="max", factor=0.5, patience=4
    )

    best_auc, patience = 0.0, 0
    for epoch in range(20):
        train_loss = train_one_epoch(model, train_loader, criterion_p1,
                                     optimizer_p1, DEVICE, train_dataset)
        train_dataset.augment = False
        val_loss, val_auc, val_f1, _, _ = evaluate(model, val_loader, DEVICE, criterion_p1)
        scheduler_p1.step(val_auc)
        lr = optimizer_p1.param_groups[0]["lr"]
        print(f"  P1 Ep{epoch+1:2d} | Train={train_loss:.4f} Val={val_loss:.4f} "
              f"AUC={val_auc:.4f} F1={val_f1:.4f} LR={lr:.2e}")
        if val_auc > best_auc:
            best_auc, patience = val_auc, 0
            save_checkpoint(model, CONFIG["save_dir"])
            print(f"  ✅ New best AUC={val_auc:.4f}")
        else:
            patience += 1
            if patience >= 6:
                print(f"  Early stop at epoch {epoch+1}")
                break
    print(f"\nPhase 1 complete. Best AUC={best_auc:.4f}")

    # Restore best head weights before phase 2
    model.head.load_state_dict(
        torch.load(os.path.join(CONFIG["save_dir"], "siglip_head_best.pth"), map_location=DEVICE)
    )

    # ── PHASE 2: Fine-tune last N blocks ───────────────────────────────────
    print("\n" + "="*50)
    print(f"PHASE 2 — Fine-tuning last {CONFIG['unfreeze_blocks']} SigLIP blocks")
    print("="*50)
    model._unfreeze_last_blocks(CONFIG["unfreeze_blocks"])
    criterion_p2 = AsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=0.05)
    param_groups = model.get_param_groups(
        head_lr=CONFIG["head_lr"],
        backbone_lr=CONFIG["backbone_lr"]
    )
    optimizer_p2 = optim.AdamW(param_groups, weight_decay=CONFIG["weight_decay"])
    scheduler_p2 = optim.lr_scheduler.CosineAnnealingLR(optimizer_p2, T_max=15, eta_min=1e-7)

    best_auc, patience = 0.0, 0
    for epoch in range(15):
        train_loss = train_one_epoch(model, train_loader, criterion_p2,
                                     optimizer_p2, DEVICE, train_dataset)
        train_dataset.augment = False
        val_loss, val_auc, val_f1, _, _ = evaluate(model, val_loader, DEVICE, criterion_p2)
        scheduler_p2.step()
        lrs = [f"{g['lr']:.2e}" for g in optimizer_p2.param_groups]
        print(f"  P2 Ep{epoch+1:2d} | Train={train_loss:.4f} Val={val_loss:.4f} "
              f"AUC={val_auc:.4f} F1={val_f1:.4f} LRs={lrs}")
        if val_auc > best_auc:
            best_auc, patience = val_auc, 0
            save_checkpoint(model, CONFIG["save_dir"])
            print(f"  ✅ New best AUC={val_auc:.4f}")
        else:
            patience += 1
            if patience >= 6:
                print(f"  Early stop at epoch {epoch+1}")
                break
    print(f"\nPhase 2 complete. Best AUC={best_auc:.4f}")

    # ── Final evaluation ────────────────────────────────────────────────────
    print("\nFINAL EVALUATION")
    model.head.load_state_dict(
        torch.load(os.path.join(CONFIG["save_dir"], "siglip_head_best.pth"), map_location=DEVICE)
    )
    backbone_path = os.path.join(CONFIG["save_dir"], "siglip_backbone_finetuned.pth")
    if os.path.exists(backbone_path):
        unfrozen = torch.load(backbone_path, map_location=DEVICE)
        for name, param in model.vision_tower.named_parameters():
            if name in unfrozen:
                param.data = unfrozen[name]

    val_dataset.augment = False
    _, _, _, all_preds, all_labels = evaluate(model, val_loader, DEVICE)
    bin_preds = (all_preds > 0.5).astype(int)
    print(classification_report(all_labels, bin_preds, target_names=DISEASES, zero_division=0))

    thresholds = optimize_thresholds(all_preds, all_labels)
    threshold_path = os.path.join(CONFIG["save_dir"], "thresholds.json")
    with open(threshold_path, "w") as f:
        json.dump(thresholds, f, indent=2)
    print(f"\nThresholds saved → {threshold_path}")