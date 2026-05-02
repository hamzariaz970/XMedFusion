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
import csv
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import open_clip
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, classification_report
from torchvision import transforms

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
CONFIG = {
    "data_dir": "data/iu_xray/images",
    "annotation_file": "data/iu_xray/annotation.json",
    "save_dir": "model_weights/KG_Agent/biomed_clip_multiview",
    "model_name": "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
    
    "batch_size": 32,
    "head_lr": 1e-3,
    "backbone_lr": 1e-5,
    "weight_decay": 1e-4,
    "seed": 42,
    "num_workers": 0,
    "unfreeze_blocks": 2,
    "phase1_epochs": 20,
    "phase2_epochs": 15,
    "early_stopping_patience": 6,
    "decision_threshold": 0.5,
    "max_pos_weight": 20.0,
    "prediction_dump_top_k": 25,
    "stable_metric_min_positives": 5,
    "checkpoint_variants": {
        "selected": "",
        "best_by_macro_ap": "best_by_macro_ap",
        "best_by_stable_ap": "best_by_stable_ap",
    },
    
    # Focal loss (Phase 2 only)
    "focal_gamma": 2.0,
}
PRIMARY_VIEW_WEIGHT = 0.7

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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
        "stable enlargement of the heart", "stable enlarged heart",
        "heart size enlarged", "mild cardiomegaly",
        "moderate cardiomegaly", "stable cardiomegaly",
        "mildly enlarged heart", "moderately enlarged heart",
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
        "airspace process", "airspace processes",
        "airspace infiltrate", "focal airspace disease",
        "alveolar consolidation",
    ],
    "Lung Opacity": [
        "opacity", "opacities", "opacification", "haziness",
        "hazy opacity", "airspace opacity",
    ],
    "Nodule": [
        "nodule", "nodules", "mass", "granuloma", "calcified nodule",
        "pulmonary nodule", "lung nodule", "calcified granuloma",
        "solitary pulmonary nodule", "pulmonary mass",
        "lung mass", "coin lesion",
    ],
    "Atelectasis": [
        "atelectasis", "atelectatic", "volume loss",
    ],
    "Fracture": [
        "fracture", "fractures", "rib fracture",
        "compression fracture", "wedge fracture",
        "wedge-shaped fracture", "wedge deformity",
        "healing deformity", "rib deformity",
        "healed rib fracture", "remote fracture",
        "minimally displaced fracture", "displaced fracture",
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
    def __init__(self, annotation_path, img_dir, preprocess, split="train", augment=False):
        self.img_dir = img_dir
        self.preprocess = preprocess
        self.augment = augment
        self.split = split
        self.samples = []
        self.invalid_samples = 0
        self.duplicated_single_view_samples = 0
        
        # Mild geometric perturbations only. Avoid flips/color jitter for radiographs.
        self.aug_transform = transforms.Compose([
            transforms.RandomRotation(degrees=5),
            transforms.RandomAffine(degrees=0, translate=(0.02, 0.02), scale=(0.98, 1.02)),
        ])
        
        with open(annotation_path, "r") as f:
            raw_data = json.load(f)
            if isinstance(raw_data, dict):
                if split not in raw_data:
                    raise ValueError(f"Split '{split}' not found in {annotation_path}. Available: {sorted(raw_data.keys())}")
                data = raw_data[split]
            else:
                data = raw_data
            
        print(f"Parsing {len(data)} {split} reports to generate Ground Truth labels...")
        
        for item in tqdm(data):
            img_path = item.get("image_path", [])
            if isinstance(img_path, list):
                rel_paths = [rel for rel in img_path if rel]
            elif img_path:
                rel_paths = [img_path]
            else:
                rel_paths = []

            full_paths = [os.path.join(self.img_dir, rel_path) for rel_path in rel_paths]
            full_paths = self._filter_valid_image_paths(full_paths)
            if not full_paths:
                self.invalid_samples += 1
                continue
            if len(full_paths) == 1:
                self.duplicated_single_view_samples += 1
                
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
            
            sample_id = item.get("id", f"{split}_{len(self.samples)}")
            self.samples.append(
                {
                    "id": sample_id,
                    "paths": full_paths,
                    "labels": labels,
                    "report": item.get("report", ""),
                }
            )
            
        print(f"✅ Loaded {len(self.samples)} valid {split} image-report pairs.")
        if self.invalid_samples:
            print(f"⚠️ Skipped {self.invalid_samples} {split} samples with no readable image.")
        if self.duplicated_single_view_samples:
            print(f"ℹ️ {self.duplicated_single_view_samples} {split} samples have one readable image and will be duplicated to 2 views.")
        
        all_labels = np.array([s["labels"] for s in self.samples])
        print(f"📊 {split.capitalize()} label distribution:")
        for idx, disease in enumerate(DISEASES):
            count = int(all_labels[:, idx].sum())
            pct = 100 * count / len(self.samples)
            print(f"   {disease:>20s}: {count:4d} ({pct:5.1f}%)")

    def _filter_valid_image_paths(self, paths):
        valid_paths = []
        for path in paths:
            if not os.path.exists(path):
                continue
            try:
                with Image.open(path) as img:
                    img.verify()
                valid_paths.append(path)
            except Exception:
                continue
        return valid_paths

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        paths = sample["paths"]
        label = sample["labels"]
        try:
            view_tensors = []
            for path in paths:
                image = Image.open(path).convert("RGB")
                if self.augment:
                    image = self.aug_transform(image)
                view_tensors.append(self.preprocess(image))

            while len(view_tensors) < 2:
                view_tensors.append(view_tensors[-1].clone())

            return torch.stack(view_tensors[:2], dim=0), torch.tensor(label, dtype=torch.float32)
        except Exception as exc:
            raise RuntimeError(f"Failed to load sample idx={idx} with paths={paths}") from exc


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

    def fuse_view_features(self, features):
        if features.dim() == 2:
            return features

        batch_size, num_views, hidden_dim = features.shape
        if num_views == 1:
            return features[:, 0, :]

        aux_weight = max(0.0, 1.0 - PRIMARY_VIEW_WEIGHT)
        weights = torch.full(
            (batch_size, num_views),
            aux_weight / max(1, num_views - 1),
            dtype=features.dtype,
            device=features.device,
        )
        weights[:, 0] = PRIMARY_VIEW_WEIGHT
        fused = (features * weights.unsqueeze(-1)).sum(dim=1)
        return F.normalize(fused, dim=-1)

    def forward(self, x):
        batch_size = x.shape[0]
        if x.dim() == 5:
            num_views = x.shape[1]
            x = x.view(batch_size * num_views, *x.shape[2:])
        else:
            num_views = 1

        with torch.set_grad_enabled(any(p.requires_grad for p in self.backbone.parameters())):
            features = self.backbone.encode_image(x)
            features = F.normalize(features, dim=-1)
        if num_views > 1:
            features = features.view(batch_size, num_views, -1)
            features = self.fuse_view_features(features)
        return self.classifier(features)


# ---------------------------------------------------------
# FOCAL LOSS (Phase 2 only)
# ---------------------------------------------------------
class WeightedFocalLoss(nn.Module):
    def __init__(self, pos_weight=None, gamma=2.0):
        super().__init__()
        self.register_buffer("pos_weight", pos_weight if pos_weight is not None else None)
        self.gamma = gamma
    
    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction='none'
        )
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1.0 - probs) * (1.0 - targets)
        focal = (1.0 - pt).pow(self.gamma) * bce
        return focal.mean()


# ---------------------------------------------------------
# EVALUATION (consistent loss calculation)
# ---------------------------------------------------------
def evaluate(model, loader, device, criterion=None, threshold=0.5):
    """Evaluate with threshold-free metrics for model selection and fixed-threshold F1 for reference."""
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

    per_label_metrics = {}
    aurocs = []
    aps = []
    f1s = []

    for idx, disease in enumerate(DISEASES):
        y_true = all_labels[:, idx]
        y_score = all_preds[:, idx]
        y_pred = (y_score >= threshold).astype(int)

        label_metrics = {
            "positives": int(y_true.sum()),
            "negatives": int(len(y_true) - y_true.sum()),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        }

        if np.unique(y_true).size == 2:
            label_metrics["auroc"] = float(roc_auc_score(y_true, y_score))
            aurocs.append(label_metrics["auroc"])
        else:
            label_metrics["auroc"] = None

        if y_true.sum() > 0:
            label_metrics["average_precision"] = float(average_precision_score(y_true, y_score))
            aps.append(label_metrics["average_precision"])
            f1s.append(label_metrics["f1"])
        else:
            label_metrics["average_precision"] = None

        per_label_metrics[disease] = label_metrics

    metrics = {
        "macro_auroc": float(np.mean(aurocs)) if aurocs else float("nan"),
        "macro_average_precision": float(np.mean(aps)) if aps else float("nan"),
        "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
        "valid_auroc_labels": len(aurocs),
        "valid_ap_labels": len(aps),
        "per_label": per_label_metrics,
    }

    stable_labels = [
        disease for disease, label_metrics in per_label_metrics.items()
        if label_metrics["positives"] >= CONFIG["stable_metric_min_positives"]
    ]
    stable_aurocs = [
        per_label_metrics[disease]["auroc"]
        for disease in stable_labels
        if per_label_metrics[disease]["auroc"] is not None
    ]
    stable_aps = [
        per_label_metrics[disease]["average_precision"]
        for disease in stable_labels
        if per_label_metrics[disease]["average_precision"] is not None
    ]
    stable_f1s = [per_label_metrics[disease]["f1"] for disease in stable_labels]
    metrics["stable_subset"] = {
        "min_positives": CONFIG["stable_metric_min_positives"],
        "labels": stable_labels,
        "macro_auroc": float(np.mean(stable_aurocs)) if stable_aurocs else float("nan"),
        "macro_average_precision": float(np.mean(stable_aps)) if stable_aps else float("nan"),
        "macro_f1": float(np.mean(stable_f1s)) if stable_f1s else 0.0,
        "valid_auroc_labels": len(stable_aurocs),
        "valid_ap_labels": len(stable_aps),
    }

    return total_loss / len(loader), metrics, all_preds, all_labels


def get_selection_metric(metrics):
    stable_metrics = metrics.get("stable_subset", {})
    stable_ap = stable_metrics.get("macro_average_precision", float("nan"))
    if not np.isnan(stable_ap):
        return stable_ap, f"stable AP (pos>={stable_metrics.get('min_positives', '?')})"
    full_ap = metrics.get("macro_average_precision", float("nan"))
    return full_ap, "macro AP"


def optimize_thresholds(all_preds, all_labels, diseases):
    """Per-disease threshold optimization."""
    thresholds = {}
    threshold_metrics = {}
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
        threshold_metrics[disease] = {"threshold": best_thresh, "f1": float(best_f1)}
        print(f"   {disease:>20s}: threshold={best_thresh:.2f}  (F1={best_f1:.3f})")
    
    return thresholds, threshold_metrics


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


def compute_pos_weights(dataset):
    labels = np.array([sample["labels"] for sample in dataset.samples], dtype=np.float32)
    positives = labels.sum(axis=0)
    negatives = len(labels) - positives
    raw_pos_weight = (negatives + 1.0) / (positives + 1.0)
    clipped_pos_weight = np.clip(raw_pos_weight, 1.0, CONFIG["max_pos_weight"])
    return torch.tensor(clipped_pos_weight, dtype=torch.float32), positives.astype(int), negatives.astype(int)


def resolve_checkpoint_dir(base_dir, variant_key="selected"):
    variant_suffix = CONFIG.get("checkpoint_variants", {}).get(variant_key, "")
    return os.path.join(base_dir, variant_suffix) if variant_suffix else base_dir


def save_checkpoint(model, save_dir, variant_key="selected"):
    checkpoint_dir = resolve_checkpoint_dir(save_dir, variant_key)
    os.makedirs(checkpoint_dir, exist_ok=True)

    head_path = os.path.join(checkpoint_dir, "biomed_clip_head_best.pth")
    torch.save(model.classifier.state_dict(), head_path)
    
    backbone_path = os.path.join(checkpoint_dir, "biomed_clip_backbone_finetuned.pth")
    unfrozen = {}
    for name, param in model.backbone.named_parameters():
        if param.requires_grad:
            unfrozen[name] = param.data.clone()
    if unfrozen:
        torch.save(unfrozen, backbone_path)
    elif os.path.exists(backbone_path):
        os.remove(backbone_path)


def load_best_checkpoint(model, save_dir, device, variant_key="selected"):
    checkpoint_dir = resolve_checkpoint_dir(save_dir, variant_key)
    model.classifier.load_state_dict(
        torch.load(os.path.join(checkpoint_dir, "biomed_clip_head_best.pth"), map_location=device)
    )

    backbone_path = os.path.join(checkpoint_dir, "biomed_clip_backbone_finetuned.pth")
    if os.path.exists(backbone_path):
        unfrozen = torch.load(backbone_path, map_location=device)
        for name, param in model.backbone.named_parameters():
            if name in unfrozen:
                param.data = unfrozen[name]


def dump_validation_predictions(dataset, all_preds, all_labels, thresholds, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    rows = []
    per_label_rows = {disease: [] for disease in DISEASES}

    for sample, pred_scores, true_labels in zip(dataset.samples, all_preds, all_labels):
        row = {
            "sample_id": sample["id"],
            "image_paths": "|".join(sample["paths"]),
        }
        for idx, disease in enumerate(DISEASES):
            score = float(pred_scores[idx])
            true_label = int(true_labels[idx])
            threshold = float(thresholds[disease])
            pred_label = int(score >= threshold)
            row[f"{disease}_score"] = score
            row[f"{disease}_true"] = true_label
            row[f"{disease}_pred"] = pred_label
            row[f"{disease}_threshold"] = threshold

            per_label_rows[disease].append(
                {
                    "sample_id": sample["id"],
                    "image_paths": sample["paths"],
                    "score": score,
                    "threshold": threshold,
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "error_type": (
                        "tp" if pred_label == 1 and true_label == 1 else
                        "fp" if pred_label == 1 and true_label == 0 else
                        "fn" if pred_label == 0 and true_label == 1 else
                        "tn"
                    ),
                    "report": sample["report"],
                }
            )
        rows.append(row)

    summary_csv_path = os.path.join(save_dir, "val_predictions_all_labels.csv")
    fieldnames = list(rows[0].keys()) if rows else ["sample_id", "image_paths"]
    with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    label_analysis = {}
    for disease, disease_rows in per_label_rows.items():
        disease_rows.sort(key=lambda item: item["score"], reverse=True)
        positives = [row for row in disease_rows if row["true_label"] == 1]
        false_positives = [row for row in disease_rows if row["error_type"] == "fp"]
        false_negatives = [row for row in disease_rows if row["error_type"] == "fn"]

        label_analysis[disease] = {
            "top_scored_samples": disease_rows[:CONFIG["prediction_dump_top_k"]],
            "top_false_positives": false_positives[:CONFIG["prediction_dump_top_k"]],
            "false_negatives": false_negatives[:CONFIG["prediction_dump_top_k"]],
            "positive_examples": positives[:CONFIG["prediction_dump_top_k"]],
        }

    analysis_path = os.path.join(save_dir, "val_prediction_analysis.json")
    with open(analysis_path, "w", encoding="utf-8") as f:
        json.dump(label_analysis, f, indent=2)

    return summary_csv_path, analysis_path


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
if __name__ == "__main__":
    seed_everything(CONFIG["seed"])
    os.makedirs(CONFIG["save_dir"], exist_ok=True)
    
    # A. Model (start with frozen backbone)
    model = BioMedCLIPClassifier(
        len(DISEASES), CONFIG["model_name"], unfreeze_blocks=0
    ).to(DEVICE)
    preprocess = model.get_preprocess()
    
    # B. Dataset
    train_dataset = CXRDataset(
        CONFIG["annotation_file"], CONFIG["data_dir"], preprocess, split="train", augment=False
    )
    val_dataset = CXRDataset(
        CONFIG["annotation_file"], CONFIG["data_dir"], preprocess, split="val", augment=False
    )

    pos_weight, train_pos_counts, train_neg_counts = compute_pos_weights(train_dataset)
    print("\n⚖️ Clipped per-label positive weights:")
    for idx, disease in enumerate(DISEASES):
        print(
            f"   {disease:>20s}: pos={train_pos_counts[idx]:4d} neg={train_neg_counts[idx]:4d} "
            f"pos_weight={pos_weight[idx].item():5.2f}"
        )
    
    train_loader = DataLoader(
        train_dataset, batch_size=CONFIG["batch_size"],
        shuffle=True, num_workers=CONFIG["num_workers"], pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_dataset, batch_size=CONFIG["batch_size"],
        shuffle=False, num_workers=CONFIG["num_workers"], pin_memory=torch.cuda.is_available()
    )
    
    # =========================================================
    # PHASE 1: Linear Probing — weighted BCE
    # =========================================================
    print(f"\n{'='*60}")
    print(f"📌 PHASE 1: Linear Probing (Frozen Backbone)")
    print(f"   Epochs: {CONFIG['phase1_epochs']} | LR: {CONFIG['head_lr']} | Loss: Weighted BCE")
    print(f"   Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    print(f"{'='*60}")
    
    criterion_p1 = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(DEVICE))
    optimizer_p1 = optim.AdamW(
        model.classifier.parameters(), lr=CONFIG["head_lr"], weight_decay=CONFIG["weight_decay"]
    )
    scheduler_p1 = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_p1, mode='max', factor=0.5, patience=4, verbose=True
    )
    
    best_f1 = 0.0
    best_ap = -1.0
    best_stable_ap = -1.0
    best_selection_value = -1.0
    best_metric_name = "macro AP"
    best_macro_ap_value = -1.0
    best_stable_ap_value = -1.0
    patience = 0
    
    for epoch in range(CONFIG["phase1_epochs"]):
        train_loss = train_one_epoch(
            model, train_loader, criterion_p1, optimizer_p1, DEVICE, train_dataset
        )
        
        train_dataset.augment = False
        val_loss, val_metrics, _, _ = evaluate(
            model, val_loader, DEVICE, criterion_p1, threshold=CONFIG["decision_threshold"]
        )
        val_ap = val_metrics["macro_average_precision"]
        selection_metric, selection_metric_name = get_selection_metric(val_metrics)
        stable_metrics = val_metrics["stable_subset"]
        stable_ap = stable_metrics["macro_average_precision"]
        scheduler_p1.step(selection_metric if not np.isnan(selection_metric) else -val_loss)
        
        lr = optimizer_p1.param_groups[0]['lr']
        val_auc_str = f"{val_metrics['macro_auroc']:.4f}" if not np.isnan(val_metrics["macro_auroc"]) else "nan"
        val_ap_str = f"{val_ap:.4f}" if not np.isnan(val_ap) else "nan"
        stable_ap_str = f"{stable_ap:.4f}" if not np.isnan(stable_ap) else "nan"
        print(
            f"  P1 Epoch {epoch+1:2d}: Train={train_loss:.4f} | Val={val_loss:.4f} | "
            f"AUC={val_auc_str} ({val_metrics['valid_auroc_labels']}) | "
            f"AP={val_ap_str} ({val_metrics['valid_ap_labels']}) | "
            f"StableAP={stable_ap_str} ({stable_metrics['valid_ap_labels']}) | "
            f"F1@0.5={val_metrics['macro_f1']:.4f} | LR={lr:.2e}"
        )

        if val_ap > best_macro_ap_value:
            best_macro_ap_value = val_ap
            save_checkpoint(model, CONFIG["save_dir"], variant_key="best_by_macro_ap")
            print(f"  💾 Updated best_by_macro_ap ({val_ap_str})")

        if stable_ap > best_stable_ap_value:
            best_stable_ap_value = stable_ap
            save_checkpoint(model, CONFIG["save_dir"], variant_key="best_by_stable_ap")
            print(f"  💾 Updated best_by_stable_ap ({stable_ap_str})")
        
        if selection_metric > best_selection_value:
            best_ap = val_ap
            best_stable_ap = stable_ap
            best_selection_value = selection_metric
            best_f1 = val_metrics["macro_f1"]
            best_metric_name = selection_metric_name
            patience = 0
            save_checkpoint(model, CONFIG["save_dir"], variant_key="selected")
            print(
                f"  ⭐ New Best! ({selection_metric_name}: {selection_metric:.4f}, "
                f"macro AP: {val_ap_str}, stable AP: {stable_ap_str}, F1@0.5: {best_f1:.4f})"
            )
        else:
            patience += 1
            if patience >= CONFIG["early_stopping_patience"]:
                print(f"  ⏹️ Phase 1 early stop at epoch {epoch+1}")
                break
    
    print(
        f"\n✅ Phase 1 complete. Best selector={best_metric_name} | "
        f"macro AP: {best_ap:.4f} | stable AP: {best_stable_ap:.4f} | F1@0.5: {best_f1:.4f}"
    )
    
    # Reload best P1 weights
    load_best_checkpoint(model, CONFIG["save_dir"], DEVICE, variant_key="selected")
    
    # =========================================================
    # PHASE 2: Fine-tune backbone + head with Focal Loss
    # =========================================================
    print(f"\n{'='*60}")
    print(f"🔓 PHASE 2: Fine-tuning (Last {CONFIG['unfreeze_blocks']} ViT Blocks)")
    print(f"   Head LR: {CONFIG['backbone_lr']} | Backbone LR: {CONFIG['backbone_lr']/5}")
    print(f"   Loss: Weighted Focal BCE (γ={CONFIG['focal_gamma']})")
    print(f"{'='*60}")
    
    # Unfreeze backbone
    model._unfreeze_blocks(CONFIG["unfreeze_blocks"])
    
    criterion_p2 = WeightedFocalLoss(pos_weight=pos_weight.to(DEVICE), gamma=CONFIG["focal_gamma"])
    
    param_groups = model.get_param_groups(
        head_lr=CONFIG["backbone_lr"],       # 1e-5 for head
        backbone_lr=CONFIG["backbone_lr"]/5   # 2e-6 for backbone
    )
    optimizer_p2 = optim.AdamW(param_groups, weight_decay=CONFIG["weight_decay"])
    scheduler_p2 = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_p2, mode='max', factor=0.5, patience=3, verbose=True
    )
    
    patience = 0
    
    for epoch in range(CONFIG["phase2_epochs"]):
        train_loss = train_one_epoch(
            model, train_loader, criterion_p2, optimizer_p2, DEVICE, train_dataset
        )
        
        train_dataset.augment = False
        val_loss, val_metrics, _, _ = evaluate(
            model, val_loader, DEVICE, criterion_p2, threshold=CONFIG["decision_threshold"]
        )
        val_ap = val_metrics["macro_average_precision"]
        selection_metric, selection_metric_name = get_selection_metric(val_metrics)
        stable_metrics = val_metrics["stable_subset"]
        stable_ap = stable_metrics["macro_average_precision"]
        scheduler_p2.step(selection_metric if not np.isnan(selection_metric) else -val_loss)
        
        lrs = [f"{g['lr']:.2e}" for g in optimizer_p2.param_groups]
        val_auc_str = f"{val_metrics['macro_auroc']:.4f}" if not np.isnan(val_metrics["macro_auroc"]) else "nan"
        val_ap_str = f"{val_ap:.4f}" if not np.isnan(val_ap) else "nan"
        stable_ap_str = f"{stable_ap:.4f}" if not np.isnan(stable_ap) else "nan"
        print(
            f"  P2 Epoch {epoch+1:2d}: Train={train_loss:.4f} | Val={val_loss:.4f} | "
            f"AUC={val_auc_str} ({val_metrics['valid_auroc_labels']}) | "
            f"AP={val_ap_str} ({val_metrics['valid_ap_labels']}) | "
            f"StableAP={stable_ap_str} ({stable_metrics['valid_ap_labels']}) | "
            f"F1@0.5={val_metrics['macro_f1']:.4f} | LRs={lrs}"
        )

        if val_ap > best_macro_ap_value:
            best_macro_ap_value = val_ap
            save_checkpoint(model, CONFIG["save_dir"], variant_key="best_by_macro_ap")
            print(f"  💾 Updated best_by_macro_ap ({val_ap_str})")

        if stable_ap > best_stable_ap_value:
            best_stable_ap_value = stable_ap
            save_checkpoint(model, CONFIG["save_dir"], variant_key="best_by_stable_ap")
            print(f"  💾 Updated best_by_stable_ap ({stable_ap_str})")
        
        if selection_metric > best_selection_value:
            best_ap = val_ap
            best_stable_ap = stable_ap
            best_selection_value = selection_metric
            best_f1 = val_metrics["macro_f1"]
            best_metric_name = selection_metric_name
            patience = 0
            save_checkpoint(model, CONFIG["save_dir"], variant_key="selected")
            print(
                f"  ⭐ New Best! ({selection_metric_name}: {selection_metric:.4f}, "
                f"macro AP: {val_ap_str}, stable AP: {stable_ap_str}, F1@0.5: {best_f1:.4f})"
            )
        else:
            patience += 1
            if patience >= CONFIG["early_stopping_patience"]:
                print(f"  ⏹️ Phase 2 early stop at epoch {epoch+1}")
                break
    
    print(
        f"\n✅ Phase 2 complete. Best selector={best_metric_name} | "
        f"macro AP: {best_ap:.4f} | stable AP: {best_stable_ap:.4f} | F1@0.5: {best_f1:.4f}"
    )
    
    # =========================================================
    # FINAL EVALUATION & THRESHOLDS
    # =========================================================
    print("\n📊 --- FINAL EVALUATION ---")
    load_best_checkpoint(model, CONFIG["save_dir"], DEVICE, variant_key="selected")
    
    train_dataset.augment = False
    _, final_metrics, all_preds, all_labels = evaluate(
        model, val_loader, DEVICE, nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(DEVICE)),
        threshold=CONFIG["decision_threshold"]
    )

    print("\n📈 Validation metrics (fixed threshold 0.5):")
    valid_auroc = f"{final_metrics['macro_auroc']:.4f}" if not np.isnan(final_metrics["macro_auroc"]) else "nan"
    valid_ap = (
        f"{final_metrics['macro_average_precision']:.4f}"
        if not np.isnan(final_metrics["macro_average_precision"]) else "nan"
    )
    print(
        f"   Macro AUROC: {valid_auroc} over {final_metrics['valid_auroc_labels']} labels\n"
        f"   Macro AP:    {valid_ap} over {final_metrics['valid_ap_labels']} labels\n"
        f"   Macro F1:    {final_metrics['macro_f1']:.4f}"
    )
    stable_metrics = final_metrics["stable_subset"]
    stable_auroc = f"{stable_metrics['macro_auroc']:.4f}" if not np.isnan(stable_metrics["macro_auroc"]) else "nan"
    stable_ap = (
        f"{stable_metrics['macro_average_precision']:.4f}"
        if not np.isnan(stable_metrics["macro_average_precision"]) else "nan"
    )
    print(
        f"   Stable AUROC (pos>={stable_metrics['min_positives']}): {stable_auroc} over {stable_metrics['valid_auroc_labels']} labels\n"
        f"   Stable AP    (pos>={stable_metrics['min_positives']}): {stable_ap} over {stable_metrics['valid_ap_labels']} labels\n"
        f"   Stable F1    (pos>={stable_metrics['min_positives']}): {stable_metrics['macro_f1']:.4f}"
    )

    thresholds, threshold_metrics = optimize_thresholds(all_preds, all_labels, DISEASES)
    bin_preds = np.column_stack(
        [(all_preds[:, idx] >= thresholds[disease]).astype(int) for idx, disease in enumerate(DISEASES)]
    )
    print(classification_report(all_labels, bin_preds, target_names=DISEASES, zero_division=0))

    threshold_path = os.path.join(CONFIG["save_dir"], "thresholds.json")
    with open(threshold_path, "w") as f:
        json.dump(thresholds, f, indent=2)
    print(f"\n✅ Thresholds saved to {threshold_path}")

    metrics_path = os.path.join(CONFIG["save_dir"], "validation_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "macro_auroc": None if np.isnan(final_metrics["macro_auroc"]) else final_metrics["macro_auroc"],
                "macro_average_precision": (
                    None if np.isnan(final_metrics["macro_average_precision"])
                    else final_metrics["macro_average_precision"]
                ),
                "selection_metric_name": get_selection_metric(final_metrics)[1],
                "selection_metric_value": get_selection_metric(final_metrics)[0],
                "checkpoint_paths": {
                    "selected": resolve_checkpoint_dir(CONFIG["save_dir"], "selected"),
                    "best_by_macro_ap": resolve_checkpoint_dir(CONFIG["save_dir"], "best_by_macro_ap"),
                    "best_by_stable_ap": resolve_checkpoint_dir(CONFIG["save_dir"], "best_by_stable_ap"),
                },
                "macro_f1_at_0_5": final_metrics["macro_f1"],
                "valid_auroc_labels": final_metrics["valid_auroc_labels"],
                "valid_ap_labels": final_metrics["valid_ap_labels"],
                "stable_subset_metrics": final_metrics["stable_subset"],
                "per_label_metrics": final_metrics["per_label"],
                "optimized_thresholds": threshold_metrics,
            },
            f,
            indent=2,
        )
    print(f"✅ Validation metrics saved to {metrics_path}")

    predictions_csv_path, prediction_analysis_path = dump_validation_predictions(
        val_dataset, all_preds, all_labels, thresholds, CONFIG["save_dir"]
    )
    print(f"✅ Validation predictions saved to {predictions_csv_path}")
    print(f"✅ Validation error analysis saved to {prediction_analysis_path}")
    
    print(f"\n{'='*50}")
    print(f"Training Complete!")
    print(f"   Best Val selector: {best_metric_name}")
    print(f"   Best Val macro AP: {best_ap:.4f}")
    print(f"   Best Val stable AP: {best_stable_ap:.4f}")
    print(f"   Best Val F1@0.5: {best_f1:.4f}")
    print(f"   Model: {CONFIG['save_dir']}/biomed_clip_head_best.pth")
    print(f"   Backbone: {CONFIG['save_dir']}/biomed_clip_backbone_finetuned.pth")
    print(f"   Best-by-macro-AP dir: {resolve_checkpoint_dir(CONFIG['save_dir'], 'best_by_macro_ap')}")
    print(f"   Best-by-stable-AP dir: {resolve_checkpoint_dir(CONFIG['save_dir'], 'best_by_stable_ap')}")
    print(f"   Thresholds: {threshold_path}")
    print(f"   Metrics: {metrics_path}")
    print(f"   Predictions: {predictions_csv_path}")
    print(f"   Analysis: {prediction_analysis_path}")
    print(f"{'='*50}")
