"""
KG Agent RAD-DINO Classifier Training Script
===========================================

RAD-DINO is used as a chest-X-ray-specific ViT encoder. This script trains a
multi-label pathology classifier on the IU X-ray train split, selects/tunes on
the validation split, and reports final comparison artifacts on the test split.

Artifacts are written to:
    model_weights/KG_Agent/rad_dino_multiview

Key differences from the BioMedCLIP KG trainer:
  - Uses microsoft/rad-dino through transformers.AutoModel.
  - Pools both CLS and mean patch tokens so local findings can influence
    labels such as Nodule, Fracture, Pneumothorax, and Atelectasis.
  - Uses LayerNorm instead of BatchNorm because RAD-DINO needs smaller batches.
  - Tunes precision-oriented thresholds on validation and applies those
    thresholds to test.
  - Exports KG-ready present/uncertain/absent statuses with rare-label
    abstention.
"""

from __future__ import annotations

import csv
import json
import os
import random
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from sklearn.metrics import average_precision_score, classification_report, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
CONFIG = {
    "data_dir": "data/iu_xray/images",
    "annotation_file": "data/iu_xray/annotation.json",
    "save_dir": "model_weights/KG_Agent/rad_dino_multiview",
    "model_name": "microsoft/rad-dino",
    "batch_size": 8,
    "head_lr": 7e-4,
    "backbone_lr": 8e-6,
    "weight_decay": 1e-4,
    "seed": 42,
    "num_workers": 0,
    "max_views": 2,
    "primary_view_weight": 0.75,
    "phase1_epochs": 18,
    "phase2_epochs": 8,
    "unfreeze_blocks": 2,
    "early_stopping_patience": 5,
    "decision_threshold": 0.5,
    "max_pos_weight": 25.0,
    "prediction_dump_top_k": 25,
    "stable_metric_min_positives": 5,
    "focal_gamma": 2.0,
    "threshold_grid_start": 0.05,
    "threshold_grid_end": 0.95,
    "threshold_grid_step": 0.025,
    "use_rare_label_sampler": True,
    "rare_label_sampler_boost": 2.0,
    "rare_label_sampler_max_weight": 12.0,
    "rare_label_sampler_labels": [
        "Pleural Effusion",
        "Edema",
        "Pneumothorax",
        "Consolidation",
        "Nodule",
        "Fracture",
    ],
    "selection_metric": "weak_macro_ap",
    "checkpoint_variants": {
        "selected": "",
        "best_by_macro_ap": "best_by_macro_ap",
        "best_by_stable_ap": "best_by_stable_ap",
    },
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def apply_env_overrides() -> None:
    """Allow experiment runners to tune training without editing this file."""
    env_map = {
        "KG_RAD_SAVE_DIR": ("save_dir", str),
        "KG_RAD_BATCH_SIZE": ("batch_size", int),
        "KG_RAD_PHASE1_EPOCHS": ("phase1_epochs", int),
        "KG_RAD_PHASE2_EPOCHS": ("phase2_epochs", int),
        "KG_RAD_HEAD_LR": ("head_lr", float),
        "KG_RAD_BACKBONE_LR": ("backbone_lr", float),
        "KG_RAD_MAX_POS_WEIGHT": ("max_pos_weight", float),
        "KG_RAD_FOCAL_GAMMA": ("focal_gamma", float),
        "KG_RAD_EARLY_STOPPING_PATIENCE": ("early_stopping_patience", int),
        "KG_RAD_SELECTION_METRIC": ("selection_metric", str),
        "KG_RAD_SAMPLER_BOOST": ("rare_label_sampler_boost", float),
        "KG_RAD_SAMPLER_MAX_WEIGHT": ("rare_label_sampler_max_weight", float),
        "KG_RAD_STABLE_METRIC_MIN_POSITIVES": ("stable_metric_min_positives", int),
    }
    for env_name, (config_key, caster) in env_map.items():
        value = os.getenv(env_name)
        if value is not None:
            CONFIG[config_key] = caster(value)

    CONFIG["use_rare_label_sampler"] = _env_bool(
        "KG_RAD_USE_RARE_LABEL_SAMPLER",
        bool(CONFIG["use_rare_label_sampler"]),
    )
    labels = os.getenv("KG_RAD_RARE_LABELS")
    if labels:
        CONFIG["rare_label_sampler_labels"] = [
            item.strip()
            for item in labels.split(",")
            if item.strip()
        ]


apply_env_overrides()


# KG policy is deliberately more conservative than classifier evaluation.
# A wrong KG edge is usually worse than a missed weak visual suspicion.
KG_LABEL_POLICY = {
    "Cardiomegaly": {
        "target_precision": 0.60,
        "min_val_positives": 5,
        "fallback_threshold": 0.85,
        "kg_enabled": True,
    },
    "Pleural Effusion": {
        "target_precision": 0.60,
        "min_val_positives": 5,
        "fallback_threshold": 0.70,
        "kg_enabled": True,
    },
    "Edema": {
        "target_precision": 0.75,
        "min_val_positives": 10,
        "fallback_threshold": 0.90,
        "kg_enabled": False,
    },
    "Pneumothorax": {
        "target_precision": 0.75,
        "min_val_positives": 10,
        "fallback_threshold": 0.90,
        "kg_enabled": False,
    },
    "Infiltrate": {
        "target_precision": 0.65,
        "min_val_positives": 10,
        "fallback_threshold": 0.75,
        "kg_enabled": True,
    },
    "Consolidation": {
        "target_precision": 0.75,
        "min_val_positives": 10,
        "fallback_threshold": 0.90,
        "kg_enabled": False,
    },
    "Lung Opacity": {
        "target_precision": 0.50,
        "min_val_positives": 10,
        "fallback_threshold": 0.75,
        "kg_enabled": True,
    },
    "Nodule": {
        "target_precision": 0.75,
        "min_val_positives": 10,
        "fallback_threshold": 0.85,
        "kg_enabled": True,
    },
    "Atelectasis": {
        "target_precision": 0.50,
        "min_val_positives": 10,
        "fallback_threshold": 0.85,
        "kg_enabled": True,
    },
    "Fracture": {
        "target_precision": 0.75,
        "min_val_positives": 10,
        "fallback_threshold": 0.90,
        "kg_enabled": False,
    },
}


# ---------------------------------------------------------
# DISEASES & LABEL HEURISTICS
# ---------------------------------------------------------
DISEASES = [
    "Cardiomegaly", "Pleural Effusion", "Edema", "Pneumothorax",
    "Infiltrate", "Consolidation", "Lung Opacity", "Nodule",
    "Atelectasis", "Fracture",
]

KEYWORD_MAP = {
    "Cardiomegaly": [
        "cardiomegaly", "cardiac enlargement", "heart is enlarged",
        "heart size is enlarged", "enlarged heart", "heart is large",
        "enlargement of the cardiac silhouette", "enlarged cardiac silhouette",
        "moderate-to-marked enlargement of the cardiac silhouette",
        "heart is moderately enlarged", "heart is mildly enlarged",
        "heart is severely enlarged", "cardiac silhouette is enlarged",
        "cardiomediastinal silhouette is enlarged",
        "stable enlargement of the heart", "stable enlarged heart",
        "heart size enlarged", "mild cardiomegaly",
        "moderate cardiomegaly", "stable cardiomegaly",
        "mildly enlarged heart", "moderately enlarged heart",
    ],
    "Pleural Effusion": [
        "pleural effusion", "pleural effusions", "effusions",
        "pleural fluid", "posterior pleural effusion",
        "costophrenic blunting", "costophrenic angle blunting",
        "costophrenic sulcus blunting", "costophrenic recess blunting",
        "blunting of the costophrenic", "blunting of bilateral costophrenic",
        "blunting of the bilateral costophrenic", "blunted costophrenic",
        "blunted posterior costophrenic",
    ],
    "Edema": [
        "edema", "pulmonary edema", "vascular congestion",
        "pulmonary congestion", "fluid overload", "interstitial edema",
        "central vascular congestion", "pulmonary xxxx are engorged",
    ],
    "Pneumothorax": [
        "pneumothorax", "pneumothoraces", "pleural air",
        "pleural air collection",
    ],
    "Infiltrate": [
        "infiltrate", "infiltrates", "airspace disease",
        "airspace opacity", "air space opacity", "air space opacities",
        "airspace consolidation",
    ],
    "Consolidation": [
        "consolidation", "consolidations", "focal consolidation",
        "lobar consolidation", "airspace process", "airspace processes",
        "airspace infiltrate", "focal airspace disease",
        "alveolar consolidation",
    ],
    "Lung Opacity": [
        "opacity", "opacities", "opacification", "haziness",
        "hazy opacity", "airspace opacity",
    ],
    "Nodule": [
        "nodule", "nodules", "mass", "calcified nodule",
        "pulmonary nodule", "lung nodule", "solitary pulmonary nodule",
        "pulmonary mass", "lung mass", "coin lesion",
    ],
    "Atelectasis": [
        "atelectasis", "atelectatic", "volume loss",
    ],
    "Fracture": [
        "fracture", "fractures", "rib fracture", "compression fracture",
        "wedge fracture", "wedge-shaped fracture", "wedge deformity",
        "healing deformity", "rib deformity", "healed rib fracture",
        "remote fracture", "minimally displaced fracture", "displaced fracture",
    ],
}

NEGATION_WINDOW = 120
NEGATION_PHRASES = [
    "no ", "no evidence", "without ", "negative for", "free of",
    "clear of", "absent", "not ", "denies ", "ruled out",
    "resolution of", "resolved", "removed", "no definite",
    "no definitive", "no visible", "no acute", "no focal",
    "no large", "no obvious", "no significant", "no suspicious",
    "no displaced", "nondisplaced",
]

IGNORE_CONTEXT = {
    "Nodule": [
        "granuloma", "granulomas", "granulomatous", "calcified granuloma",
        "calcified granulomas", "calcified hilar", "calcified mediastinal",
    ],
    "Fracture": [
        "remote", "old", "healed", "healing", "chronic", "stable",
        "age-indeterminate", "deformity", "degenerative",
    ],
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


def _keyword_context(report: str, pos: int) -> str:
    context_start = max(0, pos - NEGATION_WINDOW)
    context = report[context_start:pos]
    last_boundary = max(context.rfind("."), context.rfind("!"), context.rfind("?"))
    if last_boundary >= 0:
        context = context[last_boundary + 1:]
    return context


def collect_keyword_evidence(report: str, disease: str, keywords: Iterable[str]) -> Dict:
    evidence = {
        "positive_keywords": [],
        "negated_keywords": [],
        "ignored_keywords": [],
    }
    for keyword in keywords:
        if keyword not in report:
            continue
        start = 0
        while True:
            pos = report.find(keyword, start)
            if pos == -1:
                break
            context = _keyword_context(report, pos)
            context_after = report[pos: min(len(report), pos + NEGATION_WINDOW)]
            evidence_item = {
                "keyword": keyword,
                "context_before": context.strip(),
                "context_after": context_after.strip(),
            }
            negated = any(neg in context for neg in NEGATION_PHRASES)
            ignored = any(
                ignore in context or ignore in context_after
                for ignore in IGNORE_CONTEXT.get(disease, [])
            )
            if negated:
                evidence["negated_keywords"].append(evidence_item)
            elif ignored:
                evidence["ignored_keywords"].append(evidence_item)
            else:
                evidence["positive_keywords"].append(evidence_item)
            start = pos + len(keyword)
    return evidence


def label_audit_from_report(report: str) -> Tuple[np.ndarray, Dict]:
    normalized_report = report.lower()
    labels = np.zeros(len(DISEASES), dtype=np.float32)
    audit = {}
    for idx, disease in enumerate(DISEASES):
        evidence = collect_keyword_evidence(normalized_report, disease, KEYWORD_MAP[disease])
        if evidence["positive_keywords"]:
            labels[idx] = 1.0
        audit[disease] = {
            "label": int(labels[idx]),
            **evidence,
        }
    return labels, audit


def labels_from_report(report: str) -> np.ndarray:
    labels, _ = label_audit_from_report(report)
    return labels


try:
    from report_labels import (
        label_audit_from_report as _shared_label_audit_from_report,
        labels_from_report as _shared_labels_from_report,
    )

    label_audit_from_report = _shared_label_audit_from_report
    labels_from_report = _shared_labels_from_report
except ImportError:
    pass


# ---------------------------------------------------------
# DATASET
# ---------------------------------------------------------
class CXRDataset(Dataset):
    def __init__(self, annotation_path: str, img_dir: str, processor, split: str, augment: bool = False):
        self.img_dir = img_dir
        self.processor = processor
        self.split = split
        self.augment = augment
        self.samples: List[Dict] = []
        self.invalid_samples = 0
        self.single_view_samples = 0
        self.max_views = int(CONFIG["max_views"])

        self.aug_transform = transforms.Compose([
            transforms.RandomRotation(degrees=4),
            transforms.RandomAffine(degrees=0, translate=(0.015, 0.015), scale=(0.985, 1.015)),
        ])

        with open(annotation_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        if not isinstance(raw_data, dict) or split not in raw_data:
            raise ValueError(f"Expected split '{split}' in {annotation_path}.")

        print(f"Parsing {len(raw_data[split])} {split} reports to generate labels...")
        for item in tqdm(raw_data[split]):
            rel_paths = item.get("image_path", [])
            if isinstance(rel_paths, str):
                rel_paths = [rel_paths]
            full_paths = [
                os.path.join(self.img_dir, rel_path)
                for rel_path in rel_paths
                if rel_path
            ]
            full_paths = self._filter_valid_image_paths(full_paths)
            if not full_paths:
                self.invalid_samples += 1
                continue
            if len(full_paths) == 1:
                self.single_view_samples += 1

            report = item.get("report", "")
            labels, label_audit = label_audit_from_report(report)
            self.samples.append({
                "id": item.get("id", f"{split}_{len(self.samples)}"),
                "paths": full_paths[:self.max_views],
                "labels": labels,
                "label_audit": label_audit,
                "report": report,
            })

        print(f"Loaded {len(self.samples)} valid {split} studies.")
        if self.invalid_samples:
            print(f"Skipped {self.invalid_samples} {split} samples with no readable image.")
        if self.single_view_samples:
            print(f"{self.single_view_samples} {split} samples have one readable view.")
        self.print_label_distribution()

    @staticmethod
    def _filter_valid_image_paths(paths: List[str]) -> List[str]:
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

    def print_label_distribution(self) -> None:
        labels = np.array([sample["labels"] for sample in self.samples], dtype=np.float32)
        print(f"{self.split.capitalize()} label distribution:")
        for idx, disease in enumerate(DISEASES):
            count = int(labels[:, idx].sum()) if len(labels) else 0
            pct = 100.0 * count / max(1, len(labels))
            print(f"  {disease:>20s}: {count:4d} ({pct:5.1f}%)")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        pixel_values = []
        for path in sample["paths"]:
            try:
                image = Image.open(path).convert("RGB")
                if self.augment:
                    image = self.aug_transform(image)
                encoded = self.processor(images=image, return_tensors="pt")
                pixel_values.append(encoded["pixel_values"].squeeze(0))
            except Exception as exc:
                raise RuntimeError(f"Failed to read image {path}") from exc

        while len(pixel_values) < self.max_views:
            pixel_values.append(pixel_values[-1].clone())

        images = torch.stack(pixel_values[:self.max_views], dim=0)
        labels = torch.tensor(sample["labels"], dtype=torch.float32)
        return images, labels


# ---------------------------------------------------------
# MODEL
# ---------------------------------------------------------
class RadDinoClassifierHead(nn.Module):
    def __init__(self, num_classes: int, input_dim: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(256, num_classes),
        )
        self.residual = nn.Linear(input_dim, num_classes)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(features) + self.residual(features)


class RadDinoKGClassifier(nn.Module):
    def __init__(self, model_name: str, num_classes: int, unfreeze_blocks: int = 0):
        super().__init__()
        print(f"Loading RAD-DINO encoder: {model_name}")
        self.backbone = AutoModel.from_pretrained(model_name)
        self.hidden_size = int(getattr(self.backbone.config, "hidden_size", 768))
        self.feature_dim = self.hidden_size * 2

        for param in self.backbone.parameters():
            param.requires_grad = False
        if unfreeze_blocks > 0:
            self.unfreeze_last_blocks(unfreeze_blocks)

        self.classifier = RadDinoClassifierHead(num_classes=num_classes, input_dim=self.feature_dim)

    def unfreeze_last_blocks(self, n: int) -> None:
        layers = self._get_transformer_layers()
        if layers is None:
            print("Could not find DINO transformer layers; keeping backbone frozen.")
            return
        total = len(layers)
        start = max(0, total - n)
        for layer in layers[start:]:
            for param in layer.parameters():
                param.requires_grad = True
        for module_name in ("layernorm", "norm", "final_layernorm"):
            module = getattr(self.backbone, module_name, None)
            if module is not None:
                for param in module.parameters():
                    param.requires_grad = True
        print(f"Unfroze RAD-DINO blocks [{start}:{total}] ({n} blocks).")

    def _get_transformer_layers(self):
        for path in (
            ("encoder", "layer"),
            ("encoder", "layers"),
            ("vit", "encoder", "layer"),
            ("vision_model", "encoder", "layers"),
        ):
            obj = self.backbone
            for attr in path:
                obj = getattr(obj, attr, None)
                if obj is None:
                    break
            if obj is not None:
                return obj
        return None

    def get_param_groups(self, head_lr: float, backbone_lr: float) -> List[Dict]:
        head_params = list(self.classifier.parameters())
        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        groups = [{"params": head_params, "lr": head_lr}]
        if backbone_params:
            groups.append({"params": backbone_params, "lr": backbone_lr})
        return groups

    @staticmethod
    def _pool_tokens(outputs) -> torch.Tensor:
        hidden = outputs.last_hidden_state
        cls_token = hidden[:, 0, :]
        patch_mean = hidden[:, 1:, :].mean(dim=1) if hidden.shape[1] > 1 else cls_token
        return F.normalize(torch.cat([cls_token, patch_mean], dim=-1), dim=-1)

    def fuse_view_features(self, features: torch.Tensor) -> torch.Tensor:
        if features.dim() == 2:
            return features
        batch_size, num_views, _ = features.shape
        if num_views == 1:
            return features[:, 0, :]
        primary_weight = float(CONFIG["primary_view_weight"])
        aux_weight = max(0.0, 1.0 - primary_weight)
        weights = torch.full(
            (batch_size, num_views),
            aux_weight / max(1, num_views - 1),
            dtype=features.dtype,
            device=features.device,
        )
        weights[:, 0] = primary_weight
        return F.normalize((features * weights.unsqueeze(-1)).sum(dim=1), dim=-1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        batch_size = images.shape[0]
        if images.dim() == 5:
            num_views = images.shape[1]
            images = images.view(batch_size * num_views, *images.shape[2:])
        else:
            num_views = 1

        grad_backbone = any(param.requires_grad for param in self.backbone.parameters())
        with torch.set_grad_enabled(grad_backbone):
            outputs = self.backbone(pixel_values=images, return_dict=True)
            features = self._pool_tokens(outputs)
        if num_views > 1:
            features = features.view(batch_size, num_views, -1)
            features = self.fuse_view_features(features)
        return self.classifier(features)


# ---------------------------------------------------------
# LOSSES AND METRICS
# ---------------------------------------------------------
class WeightedFocalLoss(nn.Module):
    def __init__(self, pos_weight: torch.Tensor | None = None, gamma: float = 2.0):
        super().__init__()
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight)
        else:
            self.pos_weight = None
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none"
        )
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1.0 - probs) * (1.0 - targets)
        return ((1.0 - pt).pow(self.gamma) * bce).mean()


def compute_pos_weights(dataset: CXRDataset) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
    labels = np.array([sample["labels"] for sample in dataset.samples], dtype=np.float32)
    positives = labels.sum(axis=0)
    negatives = len(labels) - positives
    raw_weights = (negatives + 1.0) / (positives + 1.0)
    clipped_weights = np.clip(raw_weights, 1.0, CONFIG["max_pos_weight"])
    return torch.tensor(clipped_weights, dtype=torch.float32), positives.astype(int), negatives.astype(int)


def build_rare_label_sampler(dataset: CXRDataset) -> WeightedRandomSampler | None:
    if not CONFIG.get("use_rare_label_sampler", False):
        return None
    labels = np.array([sample["labels"] for sample in dataset.samples], dtype=np.float32)
    if labels.size == 0:
        return None

    rare_indices = [
        DISEASES.index(disease)
        for disease in CONFIG["rare_label_sampler_labels"]
        if disease in DISEASES
    ]
    if not rare_indices:
        return None

    positives = labels[:, rare_indices].sum(axis=0)
    max_positive = max(1.0, float(positives.max()))
    per_label_boost = np.zeros(len(DISEASES), dtype=np.float32)
    for idx, positive_count in zip(rare_indices, positives):
        if positive_count <= 0:
            continue
        per_label_boost[idx] = float(CONFIG["rare_label_sampler_boost"]) * max_positive / float(positive_count)

    sample_weights = 1.0 + (labels * per_label_boost.reshape(1, -1)).sum(axis=1)
    sample_weights = np.clip(sample_weights, 1.0, float(CONFIG["rare_label_sampler_max_weight"]))
    print("\nRare-label sampler enabled:")
    for idx in rare_indices:
        print(f"  {DISEASES[idx]:>20s}: positives={int(labels[:, idx].sum()):4d} boost={per_label_boost[idx]:5.2f}")
    print(f"  sample weight range: {sample_weights.min():.2f} - {sample_weights.max():.2f}")
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )


def threshold_array(thresholds: Dict[str, float] | None = None, default: float = 0.5) -> np.ndarray:
    if thresholds is None:
        return np.full(len(DISEASES), default, dtype=np.float32)
    return np.array([thresholds.get(disease, default) for disease in DISEASES], dtype=np.float32)


def calculate_metrics(all_preds: np.ndarray, all_labels: np.ndarray, thresholds: Dict[str, float] | None = None) -> Dict:
    thresh = threshold_array(thresholds, CONFIG["decision_threshold"])
    per_label = {}
    aurocs, aps, f1s = [], [], []

    for idx, disease in enumerate(DISEASES):
        y_true = all_labels[:, idx]
        y_score = all_preds[:, idx]
        y_pred = (y_score >= thresh[idx]).astype(int)
        metrics = {
            "positives": int(y_true.sum()),
            "negatives": int(len(y_true) - y_true.sum()),
            "threshold": float(thresh[idx]),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        }
        if np.unique(y_true).size == 2:
            metrics["auroc"] = float(roc_auc_score(y_true, y_score))
            aurocs.append(metrics["auroc"])
        else:
            metrics["auroc"] = None
        if y_true.sum() > 0:
            metrics["average_precision"] = float(average_precision_score(y_true, y_score))
            aps.append(metrics["average_precision"])
            f1s.append(metrics["f1"])
        else:
            metrics["average_precision"] = None
        per_label[disease] = metrics

    stable_labels = [
        disease for disease, metrics in per_label.items()
        if metrics["positives"] >= CONFIG["stable_metric_min_positives"]
    ]
    stable_aurocs = [
        per_label[disease]["auroc"]
        for disease in stable_labels
        if per_label[disease]["auroc"] is not None
    ]
    stable_aps = [
        per_label[disease]["average_precision"]
        for disease in stable_labels
        if per_label[disease]["average_precision"] is not None
    ]
    stable_f1s = [per_label[disease]["f1"] for disease in stable_labels]

    return {
        "macro_auroc": float(np.mean(aurocs)) if aurocs else float("nan"),
        "macro_average_precision": float(np.mean(aps)) if aps else float("nan"),
        "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
        "valid_auroc_labels": len(aurocs),
        "valid_ap_labels": len(aps),
        "stable_subset": {
            "min_positives": CONFIG["stable_metric_min_positives"],
            "labels": stable_labels,
            "macro_auroc": float(np.mean(stable_aurocs)) if stable_aurocs else float("nan"),
            "macro_average_precision": float(np.mean(stable_aps)) if stable_aps else float("nan"),
            "macro_f1": float(np.mean(stable_f1s)) if stable_f1s else 0.0,
            "valid_auroc_labels": len(stable_aurocs),
            "valid_ap_labels": len(stable_aps),
        },
        "per_label": per_label,
    }


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    criterion: nn.Module | None = None,
    thresholds: Dict[str, float] | None = None,
) -> Tuple[float, Dict, np.ndarray, np.ndarray]:
    model.eval()
    if criterion is None:
        criterion = nn.BCEWithLogitsLoss()

    all_preds, all_labels = [], []
    total_loss = 0.0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            all_preds.append(torch.sigmoid(logits).cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    preds_np = np.vstack(all_preds)
    labels_np = np.vstack(all_labels)
    return total_loss / max(1, len(loader)), calculate_metrics(preds_np, labels_np, thresholds), preds_np, labels_np


def get_selection_metric(metrics: Dict) -> Tuple[float, str]:
    if CONFIG.get("selection_metric") == "weak_macro_ap":
        weak_labels = [
            disease for disease in CONFIG["rare_label_sampler_labels"]
            if disease in metrics.get("per_label", {})
        ]
        weak_aps = [
            metrics["per_label"][disease]["average_precision"]
            for disease in weak_labels
            if metrics["per_label"][disease]["average_precision"] is not None
        ]
        if weak_aps:
            return float(np.mean(weak_aps)), "weak-label macro AP"

    if CONFIG.get("selection_metric") == "macro_ap":
        macro_ap = metrics.get("macro_average_precision", float("nan"))
        if not np.isnan(macro_ap):
            return macro_ap, "macro AP"

    stable_ap = metrics.get("stable_subset", {}).get("macro_average_precision", float("nan"))
    if not np.isnan(stable_ap):
        return stable_ap, f"stable AP (pos>={CONFIG['stable_metric_min_positives']})"
    return metrics.get("macro_average_precision", float("nan")), "macro AP"


def optimize_thresholds(all_preds: np.ndarray, all_labels: np.ndarray) -> Tuple[Dict[str, float], Dict[str, Dict]]:
    thresholds = {}
    threshold_metrics = {}
    print("\nOptimizing precision-oriented per-label thresholds on validation set...")
    candidate_thresholds = np.arange(
        CONFIG["threshold_grid_start"],
        CONFIG["threshold_grid_end"] + 1e-8,
        CONFIG["threshold_grid_step"],
    )
    for idx, disease in enumerate(DISEASES):
        y_true = all_labels[:, idx]
        positives = int(y_true.sum())
        policy = KG_LABEL_POLICY[disease]

        if positives < policy["min_val_positives"]:
            thresholds[disease] = float(policy["fallback_threshold"])
            threshold_metrics[disease] = {
                "threshold": thresholds[disease],
                "selection": "fallback_rare_label",
                "val_positives": positives,
                "min_val_positives": policy["min_val_positives"],
                "target_precision": policy["target_precision"],
                "precision": None,
                "recall": None,
                "f1": 0.0,
                "kg_enabled": bool(policy["kg_enabled"]),
            }
            print(
                f"  {disease:>20s}: threshold={thresholds[disease]:0.3f} "
                f"fallback rare label (val positives={positives})"
            )
            continue

        best = None
        best_any = None
        for threshold in candidate_thresholds:
            y_pred = (all_preds[:, idx] >= threshold).astype(int)
            tp = int(((y_pred == 1) & (y_true == 1)).sum())
            fp = int(((y_pred == 1) & (y_true == 0)).sum())
            fn = int(((y_pred == 0) & (y_true == 1)).sum())
            predicted = int(y_pred.sum())
            precision = tp / predicted if predicted else 0.0
            recall = tp / positives if positives else 0.0
            f1 = f1_score(y_true, y_pred, zero_division=0)
            candidate = {
                "threshold": float(round(threshold, 3)),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "predicted_positives": predicted,
            }
            if predicted > 0 and (best_any is None or (f1, precision, recall) > (best_any["f1"], best_any["precision"], best_any["recall"])):
                best_any = candidate
            if predicted > 0 and precision >= policy["target_precision"]:
                if best is None or (recall, f1, precision) > (best["recall"], best["f1"], best["precision"]):
                    best = candidate

        if best is None:
            fallback = best_any or {
                "threshold": float(policy["fallback_threshold"]),
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "tp": 0,
                "fp": 0,
                "fn": positives,
                "predicted_positives": 0,
            }
            thresholds[disease] = max(float(policy["fallback_threshold"]), float(fallback["threshold"]))
            threshold_metrics[disease] = {
                **fallback,
                "threshold": thresholds[disease],
                "selection": "fallback_precision_target_not_met",
                "val_positives": positives,
                "min_val_positives": policy["min_val_positives"],
                "target_precision": policy["target_precision"],
                "kg_enabled": bool(policy["kg_enabled"]),
            }
        else:
            thresholds[disease] = float(best["threshold"])
            threshold_metrics[disease] = {
                **best,
                "selection": "precision_target_met",
                "val_positives": positives,
                "min_val_positives": policy["min_val_positives"],
                "target_precision": policy["target_precision"],
                "kg_enabled": bool(policy["kg_enabled"]),
            }

        tm = threshold_metrics[disease]
        print(
            f"  {disease:>20s}: threshold={thresholds[disease]:0.3f} "
            f"precision={tm['precision'] if tm['precision'] is not None else 'n/a'} "
            f"recall={tm['recall'] if tm['recall'] is not None else 'n/a'} "
            f"selection={tm['selection']}"
        )
    return thresholds, threshold_metrics


# ---------------------------------------------------------
# TRAINING AND CHECKPOINTS
# ---------------------------------------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str,
    dataset: CXRDataset,
) -> float:
    model.train()
    dataset.augment = True
    total_loss = 0.0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    dataset.augment = False
    return total_loss / max(1, len(loader))


def resolve_checkpoint_dir(base_dir: str, variant_key: str = "selected") -> str:
    suffix = CONFIG["checkpoint_variants"].get(variant_key, "")
    return os.path.join(base_dir, suffix) if suffix else base_dir


def save_checkpoint(model: RadDinoKGClassifier, save_dir: str, variant_key: str = "selected") -> None:
    checkpoint_dir = resolve_checkpoint_dir(save_dir, variant_key)
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(model.classifier.state_dict(), os.path.join(checkpoint_dir, "rad_dino_head_best.pth"))

    unfrozen = {
        name: param.detach().cpu().clone()
        for name, param in model.backbone.named_parameters()
        if param.requires_grad
    }
    backbone_path = os.path.join(checkpoint_dir, "rad_dino_backbone_finetuned.pth")
    if unfrozen:
        torch.save(unfrozen, backbone_path)
    elif os.path.exists(backbone_path):
        os.remove(backbone_path)


def load_checkpoint(model: RadDinoKGClassifier, save_dir: str, device: str, variant_key: str = "selected") -> None:
    checkpoint_dir = resolve_checkpoint_dir(save_dir, variant_key)
    head_path = os.path.join(checkpoint_dir, "rad_dino_head_best.pth")
    model.classifier.load_state_dict(torch.load(head_path, map_location=device))

    backbone_path = os.path.join(checkpoint_dir, "rad_dino_backbone_finetuned.pth")
    if os.path.exists(backbone_path):
        unfrozen = torch.load(backbone_path, map_location=device)
        for name, param in model.backbone.named_parameters():
            if name in unfrozen:
                param.data.copy_(unfrozen[name].to(device))


def kg_status_for_score(
    disease: str,
    score: float,
    threshold: float,
    policy: Dict[str, Dict] | None = None,
) -> Tuple[str, bool]:
    policy = policy or KG_LABEL_POLICY
    label_policy = policy[disease]
    if score >= threshold and label_policy["kg_enabled"]:
        return "present", True
    if score >= threshold and not label_policy["kg_enabled"]:
        return "uncertain", False
    return "absent", False


def build_kg_predictions(
    dataset: CXRDataset,
    all_preds: np.ndarray,
    thresholds: Dict[str, float],
    policy: Dict[str, Dict] | None = None,
) -> List[Dict]:
    policy = policy or KG_LABEL_POLICY
    kg_predictions = []
    for sample, scores in zip(dataset.samples, all_preds):
        findings = {}
        present_labels = []
        uncertain_labels = []
        for idx, disease in enumerate(DISEASES):
            score = float(scores[idx])
            threshold = float(thresholds[disease])
            status, create_edge = kg_status_for_score(disease, score, threshold, policy)
            findings[disease] = {
                "score": score,
                "threshold": threshold,
                "status": status,
                "create_kg_edge": create_edge,
                "target_precision": policy[disease]["target_precision"],
                "kg_enabled": policy[disease]["kg_enabled"],
                "kg_enable_reason": policy[disease].get("kg_enable_reason", "static_policy"),
            }
            if create_edge:
                present_labels.append(disease)
            elif status == "uncertain":
                uncertain_labels.append(disease)
        kg_predictions.append({
            "sample_id": sample["id"],
            "image_paths": sample["paths"],
            "present_for_kg": present_labels,
            "uncertain_not_in_kg": uncertain_labels,
            "findings": findings,
        })
    return kg_predictions


def dump_predictions(
    dataset: CXRDataset,
    all_preds: np.ndarray,
    all_labels: np.ndarray,
    thresholds: Dict[str, float],
    save_dir: str,
    prefix: str,
    policy: Dict[str, Dict] | None = None,
) -> Tuple[str, str, str]:
    policy = policy or KG_LABEL_POLICY
    os.makedirs(save_dir, exist_ok=True)
    rows = []
    per_label_rows = {disease: [] for disease in DISEASES}

    for sample, scores, labels in zip(dataset.samples, all_preds, all_labels):
        row = {
            "sample_id": sample["id"],
            "image_paths": "|".join(sample["paths"]),
        }
        for idx, disease in enumerate(DISEASES):
            score = float(scores[idx])
            true_label = int(labels[idx])
            threshold = float(thresholds[disease])
            pred_label = int(score >= threshold)
            kg_status, create_kg_edge = kg_status_for_score(disease, score, threshold, policy)
            error_type = (
                "tp" if pred_label == 1 and true_label == 1 else
                "fp" if pred_label == 1 and true_label == 0 else
                "fn" if pred_label == 0 and true_label == 1 else
                "tn"
            )
            row[f"{disease}_score"] = score
            row[f"{disease}_true"] = true_label
            row[f"{disease}_pred"] = pred_label
            row[f"{disease}_threshold"] = threshold
            row[f"{disease}_kg_status"] = kg_status
            row[f"{disease}_create_kg_edge"] = int(create_kg_edge)
            per_label_rows[disease].append({
                "sample_id": sample["id"],
                "image_paths": sample["paths"],
                "score": score,
                "threshold": threshold,
                "true_label": true_label,
                "pred_label": pred_label,
                "kg_status": kg_status,
                "create_kg_edge": create_kg_edge,
                "error_type": error_type,
                "report": sample["report"],
            })
        rows.append(row)

    csv_path = os.path.join(save_dir, f"{prefix}_predictions_all_labels.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    analysis = {}
    for disease, disease_rows in per_label_rows.items():
        disease_rows.sort(key=lambda item: item["score"], reverse=True)
        positives = [row for row in disease_rows if row["true_label"] == 1]
        false_positives = [row for row in disease_rows if row["error_type"] == "fp"]
        false_negatives = [row for row in disease_rows if row["error_type"] == "fn"]
        analysis[disease] = {
            "top_scored_samples": disease_rows[:CONFIG["prediction_dump_top_k"]],
            "top_false_positives": false_positives[:CONFIG["prediction_dump_top_k"]],
            "false_negatives": false_negatives[:CONFIG["prediction_dump_top_k"]],
            "positive_examples": positives[:CONFIG["prediction_dump_top_k"]],
        }

    analysis_path = os.path.join(save_dir, f"{prefix}_prediction_analysis.json")
    with open(analysis_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2)

    kg_path = os.path.join(save_dir, f"{prefix}_kg_predictions.json")
    with open(kg_path, "w", encoding="utf-8") as f:
        json.dump(build_kg_predictions(dataset, all_preds, thresholds, policy), f, indent=2)

    return csv_path, analysis_path, kg_path


def json_safe_metrics(metrics: Dict) -> Dict:
    def clean(value):
        if isinstance(value, float) and np.isnan(value):
            return None
        if isinstance(value, dict):
            return {key: clean(val) for key, val in value.items()}
        if isinstance(value, list):
            return [clean(item) for item in value]
        return value
    return clean(metrics)


def write_metrics_file(
    path: str,
    split: str,
    metrics_at_05: Dict,
    metrics_at_tuned: Dict,
    threshold_metrics: Dict[str, Dict] | None = None,
) -> None:
    selection_value, selection_name = get_selection_metric(metrics_at_05)
    payload = {
        "split": split,
        "model_name": CONFIG["model_name"],
        "pooling": "concat(cls_token, mean_patch_token), weighted multiview fusion",
        "selection_metric_name": selection_name,
        "selection_metric_value": None if np.isnan(selection_value) else selection_value,
        "fixed_threshold_metrics": metrics_at_05,
        "tuned_threshold_metrics": metrics_at_tuned,
        "optimized_thresholds": threshold_metrics,
        "checkpoint_paths": {
            "selected": resolve_checkpoint_dir(CONFIG["save_dir"], "selected"),
            "best_by_macro_ap": resolve_checkpoint_dir(CONFIG["save_dir"], "best_by_macro_ap"),
            "best_by_stable_ap": resolve_checkpoint_dir(CONFIG["save_dir"], "best_by_stable_ap"),
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_safe_metrics(payload), f, indent=2)


def build_effective_kg_policy(threshold_metrics: Dict[str, Dict]) -> Dict[str, Dict]:
    effective_policy = {}
    for disease in DISEASES:
        base_policy = dict(KG_LABEL_POLICY[disease])
        threshold_detail = threshold_metrics.get(disease, {})
        meets_precision_target = threshold_detail.get("selection") == "precision_target_met"
        static_enabled = bool(base_policy.get("kg_enabled", False))
        base_policy["kg_enabled"] = bool(static_enabled and meets_precision_target)
        base_policy["kg_enable_reason"] = (
            "validation_precision_target_met"
            if base_policy["kg_enabled"]
            else (
                "disabled_by_static_policy"
                if not static_enabled
                else threshold_detail.get("selection", "validation_precision_target_not_met")
            )
        )
        base_policy["validation_threshold_detail"] = threshold_detail
        effective_policy[disease] = base_policy
    return effective_policy


def dump_label_audit(datasets: Iterable[CXRDataset], save_dir: str) -> str:
    rows = []
    for dataset in datasets:
        for sample in dataset.samples:
            rows.append({
                "split": dataset.split,
                "sample_id": sample["id"],
                "image_paths": sample["paths"],
                "labels": {
                    disease: int(sample["labels"][idx])
                    for idx, disease in enumerate(DISEASES)
                },
                "label_audit": sample.get("label_audit", {}),
                "report": sample["report"],
            })
    audit_path = os.path.join(save_dir, "iu_xray_labels_audit.json")
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    return audit_path


def dump_label_summary(datasets: Iterable[CXRDataset], save_dir: str) -> str:
    summary = {
        disease: {
            "positive_count": 0,
            "positive_keywords": {},
            "negated_keywords": {},
            "ignored_keywords": {},
            "positive_examples": [],
            "ignored_examples": [],
        }
        for disease in DISEASES
    }
    split_counts = {}

    for dataset in datasets:
        split_counts[dataset.split] = len(dataset.samples)
        for sample in dataset.samples:
            for idx, disease in enumerate(DISEASES):
                disease_summary = summary[disease]
                audit = sample.get("label_audit", {}).get(disease, {})
                if int(sample["labels"][idx]) == 1:
                    disease_summary["positive_count"] += 1
                    if len(disease_summary["positive_examples"]) < 10:
                        disease_summary["positive_examples"].append({
                            "split": dataset.split,
                            "sample_id": sample["id"],
                            "report": sample["report"],
                        })

                for bucket in ("positive_keywords", "negated_keywords", "ignored_keywords"):
                    for item in audit.get(bucket, []):
                        keyword = item.get("keyword", "")
                        disease_summary[bucket][keyword] = disease_summary[bucket].get(keyword, 0) + 1
                        if bucket == "ignored_keywords" and len(disease_summary["ignored_examples"]) < 10:
                            disease_summary["ignored_examples"].append({
                                "split": dataset.split,
                                "sample_id": sample["id"],
                                "keyword": keyword,
                                "context_before": item.get("context_before", ""),
                                "context_after": item.get("context_after", ""),
                                "report": sample["report"],
                            })

    for disease_summary in summary.values():
        for bucket in ("positive_keywords", "negated_keywords", "ignored_keywords"):
            disease_summary[bucket] = dict(
                sorted(disease_summary[bucket].items(), key=lambda kv: kv[1], reverse=True)
            )

    summary_path = os.path.join(save_dir, "iu_xray_label_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"split_counts": split_counts, "labels": summary}, f, indent=2)
    return summary_path


def fresh_model_for_variant(variant_key: str) -> RadDinoKGClassifier:
    model = RadDinoKGClassifier(
        CONFIG["model_name"],
        len(DISEASES),
        unfreeze_blocks=CONFIG["unfreeze_blocks"],
    ).to(DEVICE)
    load_checkpoint(model, CONFIG["save_dir"], DEVICE, variant_key)
    model.eval()
    return model


def evaluate_checkpoint_variants(
    val_loader: DataLoader,
    test_loader: DataLoader,
    criterion: nn.Module,
) -> Tuple[str, str]:
    comparison = {}
    recommended_variant = "selected"
    best_validation_policy_ap = -1.0

    for variant_key in CONFIG["checkpoint_variants"]:
        checkpoint_dir = resolve_checkpoint_dir(CONFIG["save_dir"], variant_key)
        head_path = os.path.join(checkpoint_dir, "rad_dino_head_best.pth")
        if not os.path.exists(head_path):
            continue

        model = fresh_model_for_variant(variant_key)
        _, val_metrics_05, val_preds, val_labels = evaluate(model, val_loader, DEVICE, criterion)
        thresholds, threshold_metrics = optimize_thresholds(val_preds, val_labels)
        val_metrics_policy = calculate_metrics(val_preds, val_labels, thresholds)
        _, test_metrics_05, test_preds, test_labels = evaluate(model, test_loader, DEVICE, criterion)
        test_metrics_policy = calculate_metrics(test_preds, test_labels, thresholds)

        validation_policy_ap = val_metrics_policy["stable_subset"]["macro_average_precision"]
        if validation_policy_ap > best_validation_policy_ap:
            best_validation_policy_ap = validation_policy_ap
            recommended_variant = variant_key

        comparison[variant_key] = {
            "checkpoint_dir": checkpoint_dir,
            "validation_fixed_threshold_metrics": val_metrics_05,
            "validation_precision_policy_metrics": val_metrics_policy,
            "test_fixed_threshold_metrics": test_metrics_05,
            "test_precision_policy_metrics": test_metrics_policy,
            "precision_thresholds": thresholds,
            "threshold_details": threshold_metrics,
        }

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    comparison_path = os.path.join(CONFIG["save_dir"], "checkpoint_comparison.json")
    payload = {
        "recommended_variant_by_validation_policy": recommended_variant,
        "selection_rule": "highest validation stable AP after precision-oriented threshold policy",
        "variants": comparison,
    }
    with open(comparison_path, "w", encoding="utf-8") as f:
        json.dump(json_safe_metrics(payload), f, indent=2)
    return recommended_variant, comparison_path


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main() -> None:
    seed_everything(CONFIG["seed"])
    os.makedirs(CONFIG["save_dir"], exist_ok=True)
    with open(os.path.join(CONFIG["save_dir"], "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(CONFIG, f, indent=2)

    processor = AutoImageProcessor.from_pretrained(CONFIG["model_name"], use_fast=False)

    train_dataset = CXRDataset(CONFIG["annotation_file"], CONFIG["data_dir"], processor, split="train")
    val_dataset = CXRDataset(CONFIG["annotation_file"], CONFIG["data_dir"], processor, split="val")
    test_dataset = CXRDataset(CONFIG["annotation_file"], CONFIG["data_dir"], processor, split="test")
    audit_path = dump_label_audit([train_dataset, val_dataset, test_dataset], CONFIG["save_dir"])
    label_summary_path = dump_label_summary([train_dataset, val_dataset, test_dataset], CONFIG["save_dir"])
    print(f"Label audit saved to {audit_path}")
    print(f"Label summary saved to {label_summary_path}")

    pos_weight, train_pos, train_neg = compute_pos_weights(train_dataset)
    print("\nClipped per-label positive weights:")
    for idx, disease in enumerate(DISEASES):
        print(f"  {disease:>20s}: pos={train_pos[idx]:4d} neg={train_neg[idx]:4d} weight={pos_weight[idx].item():5.2f}")

    train_sampler = build_rare_label_sampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=CONFIG["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )

    model = RadDinoKGClassifier(CONFIG["model_name"], len(DISEASES), unfreeze_blocks=0).to(DEVICE)

    criterion_p1 = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(DEVICE))
    optimizer_p1 = optim.AdamW(model.classifier.parameters(), lr=CONFIG["head_lr"], weight_decay=CONFIG["weight_decay"])
    scheduler_p1 = optim.lr_scheduler.ReduceLROnPlateau(optimizer_p1, mode="max", factor=0.5, patience=3)

    best_selection = -1.0
    best_macro_ap = -1.0
    best_stable_ap = -1.0
    best_metric_name = "stable AP"
    best_macro_variant = -1.0
    best_stable_variant = -1.0
    patience = 0

    print("\n" + "=" * 72)
    print("PHASE 1: RAD-DINO frozen encoder + classifier head")
    print(f"Epochs={CONFIG['phase1_epochs']} batch_size={CONFIG['batch_size']} head_lr={CONFIG['head_lr']}")
    print("=" * 72)

    for epoch in range(CONFIG["phase1_epochs"]):
        train_loss = train_one_epoch(model, train_loader, criterion_p1, optimizer_p1, DEVICE, train_dataset)
        val_loss, val_metrics, _, _ = evaluate(model, val_loader, DEVICE, criterion_p1)
        selection, metric_name = get_selection_metric(val_metrics)
        scheduler_p1.step(selection if not np.isnan(selection) else -val_loss)

        macro_ap = val_metrics["macro_average_precision"]
        stable_ap = val_metrics["stable_subset"]["macro_average_precision"]
        print(
            f"P1 epoch {epoch + 1:02d}: train={train_loss:.4f} val={val_loss:.4f} "
            f"macroAP={macro_ap:.4f} stableAP={stable_ap:.4f} "
            f"macroAUC={val_metrics['macro_auroc']:.4f} f1@0.5={val_metrics['macro_f1']:.4f}"
        )

        if macro_ap > best_macro_variant:
            best_macro_variant = macro_ap
            save_checkpoint(model, CONFIG["save_dir"], "best_by_macro_ap")
        if stable_ap > best_stable_variant:
            best_stable_variant = stable_ap
            save_checkpoint(model, CONFIG["save_dir"], "best_by_stable_ap")
        if selection > best_selection:
            best_selection = selection
            best_macro_ap = macro_ap
            best_stable_ap = stable_ap
            best_metric_name = metric_name
            patience = 0
            save_checkpoint(model, CONFIG["save_dir"], "selected")
            print(f"  saved selected checkpoint ({metric_name}={selection:.4f})")
        else:
            patience += 1
            if patience >= CONFIG["early_stopping_patience"]:
                print(f"Phase 1 early stop at epoch {epoch + 1}.")
                break

    load_checkpoint(model, CONFIG["save_dir"], DEVICE, "selected")
    model.unfreeze_last_blocks(CONFIG["unfreeze_blocks"])

    criterion_p2 = WeightedFocalLoss(pos_weight=pos_weight.to(DEVICE), gamma=CONFIG["focal_gamma"])
    optimizer_p2 = optim.AdamW(
        model.get_param_groups(CONFIG["backbone_lr"] * 4.0, CONFIG["backbone_lr"]),
        weight_decay=CONFIG["weight_decay"],
    )
    scheduler_p2 = optim.lr_scheduler.ReduceLROnPlateau(optimizer_p2, mode="max", factor=0.5, patience=2)
    patience = 0

    print("\n" + "=" * 72)
    print(f"PHASE 2: Fine-tune last {CONFIG['unfreeze_blocks']} RAD-DINO blocks + head")
    print(f"Epochs={CONFIG['phase2_epochs']} head_lr={CONFIG['backbone_lr'] * 4.0} backbone_lr={CONFIG['backbone_lr']}")
    print("=" * 72)

    for epoch in range(CONFIG["phase2_epochs"]):
        train_loss = train_one_epoch(model, train_loader, criterion_p2, optimizer_p2, DEVICE, train_dataset)
        val_loss, val_metrics, _, _ = evaluate(model, val_loader, DEVICE, criterion_p2)
        selection, metric_name = get_selection_metric(val_metrics)
        scheduler_p2.step(selection if not np.isnan(selection) else -val_loss)

        macro_ap = val_metrics["macro_average_precision"]
        stable_ap = val_metrics["stable_subset"]["macro_average_precision"]
        print(
            f"P2 epoch {epoch + 1:02d}: train={train_loss:.4f} val={val_loss:.4f} "
            f"macroAP={macro_ap:.4f} stableAP={stable_ap:.4f} "
            f"macroAUC={val_metrics['macro_auroc']:.4f} f1@0.5={val_metrics['macro_f1']:.4f}"
        )

        if macro_ap > best_macro_variant:
            best_macro_variant = macro_ap
            save_checkpoint(model, CONFIG["save_dir"], "best_by_macro_ap")
        if stable_ap > best_stable_variant:
            best_stable_variant = stable_ap
            save_checkpoint(model, CONFIG["save_dir"], "best_by_stable_ap")
        if selection > best_selection:
            best_selection = selection
            best_macro_ap = macro_ap
            best_stable_ap = stable_ap
            best_metric_name = metric_name
            patience = 0
            save_checkpoint(model, CONFIG["save_dir"], "selected")
            print(f"  saved selected checkpoint ({metric_name}={selection:.4f})")
        else:
            patience += 1
            if patience >= CONFIG["early_stopping_patience"]:
                print(f"Phase 2 early stop at epoch {epoch + 1}.")
                break

    print("\nEvaluating checkpoint variants with the precision policy.")
    bce_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(DEVICE))
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    recommended_variant, comparison_path = evaluate_checkpoint_variants(val_loader, test_loader, bce_criterion)
    print(f"Checkpoint comparison saved to {comparison_path}")
    print(f"Validation-policy recommended checkpoint: {recommended_variant}")

    print("\nFinal evaluation using validation-policy recommended checkpoint.")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model = RadDinoKGClassifier(
        CONFIG["model_name"],
        len(DISEASES),
        unfreeze_blocks=CONFIG["unfreeze_blocks"],
    ).to(DEVICE)
    load_checkpoint(model, CONFIG["save_dir"], DEVICE, recommended_variant)

    _, val_metrics_05, val_preds, val_labels = evaluate(model, val_loader, DEVICE, bce_criterion)
    thresholds, threshold_metrics = optimize_thresholds(val_preds, val_labels)
    effective_kg_policy = build_effective_kg_policy(threshold_metrics)
    val_metrics_tuned = calculate_metrics(val_preds, val_labels, thresholds)

    _, test_metrics_05, test_preds, test_labels = evaluate(model, test_loader, DEVICE, bce_criterion)
    test_metrics_tuned = calculate_metrics(test_preds, test_labels, thresholds)

    threshold_path = os.path.join(CONFIG["save_dir"], "thresholds.json")
    with open(threshold_path, "w", encoding="utf-8") as f:
        json.dump(thresholds, f, indent=2)
    policy_path = os.path.join(CONFIG["save_dir"], "kg_label_policy.json")
    with open(policy_path, "w", encoding="utf-8") as f:
        json.dump(effective_kg_policy, f, indent=2)

    write_metrics_file(
        os.path.join(CONFIG["save_dir"], "validation_metrics.json"),
        "val",
        val_metrics_05,
        val_metrics_tuned,
        threshold_metrics,
    )
    write_metrics_file(
        os.path.join(CONFIG["save_dir"], "test_metrics.json"),
        "test",
        test_metrics_05,
        test_metrics_tuned,
        threshold_metrics,
    )

    val_csv, val_analysis, val_kg = dump_predictions(
        val_dataset, val_preds, val_labels, thresholds, CONFIG["save_dir"], "val", effective_kg_policy
    )
    test_csv, test_analysis, test_kg = dump_predictions(
        test_dataset, test_preds, test_labels, thresholds, CONFIG["save_dir"], "test", effective_kg_policy
    )

    val_bin = np.column_stack([(val_preds[:, i] >= thresholds[disease]).astype(int) for i, disease in enumerate(DISEASES)])
    test_bin = np.column_stack([(test_preds[:, i] >= thresholds[disease]).astype(int) for i, disease in enumerate(DISEASES)])
    print("\nValidation classification report with tuned thresholds:")
    print(classification_report(val_labels, val_bin, target_names=DISEASES, zero_division=0))
    print("\nTest classification report with validation-tuned thresholds:")
    print(classification_report(test_labels, test_bin, target_names=DISEASES, zero_division=0))

    print("\n" + "=" * 72)
    print("RAD-DINO KG Agent training complete.")
    print(f"Best validation selector: {best_metric_name}={best_selection:.4f}")
    print(f"Best validation macro AP: {best_macro_ap:.4f}")
    print(f"Best validation stable AP: {best_stable_ap:.4f}")
    print(f"Recommended checkpoint variant: {recommended_variant}")
    print(f"Artifacts: {CONFIG['save_dir']}")
    print(f"Thresholds: {threshold_path}")
    print(f"KG policy: {policy_path}")
    print(f"Label audit: {audit_path}")
    print(f"Label summary: {label_summary_path}")
    print(f"Checkpoint comparison: {comparison_path}")
    print(f"Validation predictions: {val_csv}")
    print(f"Validation analysis: {val_analysis}")
    print(f"Validation KG statuses: {val_kg}")
    print(f"Test predictions: {test_csv}")
    print(f"Test analysis: {test_analysis}")
    print(f"Test KG statuses: {test_kg}")
    print("=" * 72)


if __name__ == "__main__":
    main()
