"""
shared_models.py  –  Single source of truth for BioMedCLIP classifier architecture,
disease label sets, scoring utilities, and threshold helpers.

Both vision.py and kg_agent.py import from here instead of maintaining
duplicate copies.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------
# DISEASE LABELS (Runtime)
# ---------------------------------------------------------
DISEASES = [
    "Cardiomegaly", "Pleural Effusion", "Edema", "Pneumothorax",
    "Infiltrate", "Consolidation", "Lung Opacity", "Nodule",
    "Atelectasis", "Fracture"
]

# Earlier BioMedCLIP checkpoints in this project were trained against the
# 14-label CheXpert-style target set saved alongside thresholds.json.
LEGACY_CHEXPERT_LABELS = [
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
    "Lung Lesion", "Edema", "Consolidation", "Pneumonia",
    "Atelectasis", "Pneumothorax", "Pleural Effusion", "Pleural Other",
    "Fracture", "Support Devices", "No Finding"
]

CLASSIFIER_LABEL_ALIASES = {
    "Cardiomegaly": ["Cardiomegaly", "Enlarged Cardiomediastinum"],
    "Pleural Effusion": ["Pleural Effusion", "Pleural Other"],
    "Infiltrate": ["Infiltrate", "Pneumonia"],
    "Nodule": ["Nodule", "Lung Lesion"],
}

# Zero-Shot Prompts (The "Verifier")
PATHOLOGY_CONFIG = {
    "Cardiomegaly":     {"pos": "enlarged heart cardiomegaly", "neg": "normal heart size"},
    "Pleural Effusion": {"pos": "pleural effusion fluid",      "neg": "no pleural effusion"},
    "Edema":            {"pos": "pulmonary edema",             "neg": "no pulmonary edema"},
    "Pneumothorax":     {"pos": "pneumothorax air",            "neg": "no pneumothorax"},
    "Infiltrate":       {"pos": "lung infiltrate pneumonia",   "neg": "no infiltrates"},
    "Consolidation":    {"pos": "lung consolidation",          "neg": "no consolidation"},
    "Lung Opacity":     {"pos": "lung opacity",                "neg": "clear lungs"},
    "Nodule":           {"pos": "lung nodule mass",            "neg": "no nodules"},
    "Atelectasis":      {"pos": "lung atelectasis",            "neg": "no atelectasis"},
    "Fracture":         {"pos": "bone fracture broken rib",    "neg": "normal ribs no fractures intact bones"}
}


# ---------------------------------------------------------
# CLASSIFIER ARCHITECTURES
# ---------------------------------------------------------
class BioMedCLIPClassifierHead(nn.Module):
    """Deeper classifier head with residual connection. Must match training script."""
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


class LegacyBioMedCLIPClassifierHead(nn.Module):
    """Older single-linear-layer head used by earlier saved checkpoints."""
    def __init__(self, num_classes, input_dim=512):
        super().__init__()
        self.head = nn.Linear(input_dim, num_classes)

    def forward(self, features):
        return self.head(features)


# ---------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------
def _infer_classifier_output_count(state_dict):
    for key in ("head.weight", "head.8.weight", "residual.weight"):
        if key in state_dict:
            return int(state_dict[key].shape[0])
    return len(DISEASES)


def _labels_for_output_count(output_count):
    if output_count == len(DISEASES):
        return DISEASES
    if output_count == len(LEGACY_CHEXPERT_LABELS):
        return LEGACY_CHEXPERT_LABELS
    return [f"class_{idx}" for idx in range(output_count)]


def build_classifier_for_state_dict(state_dict, device=None):
    """Build the correct classifier head architecture from a saved state_dict."""
    output_count = _infer_classifier_output_count(state_dict)
    if "head.weight" in state_dict and "head.0.weight" not in state_dict:
        head = LegacyBioMedCLIPClassifierHead(num_classes=output_count)
    else:
        head = BioMedCLIPClassifierHead(num_classes=output_count)
    if device is not None:
        head = head.to(device)
    head.label_names = _labels_for_output_count(output_count)
    return head


def _classifier_probs_by_disease(raw_probs, label_names):
    """Map raw classifier probabilities to the 10-disease runtime label set."""
    raw_probs = np.atleast_1d(raw_probs)
    label_scores = {
        label: float(raw_probs[idx])
        for idx, label in enumerate(label_names[:len(raw_probs)])
    }

    disease_scores = {}
    for disease in DISEASES:
        aliases = CLASSIFIER_LABEL_ALIASES.get(disease, [disease])
        alias_scores = [label_scores[label] for label in aliases if label in label_scores]
        disease_scores[disease] = max(alias_scores) if alias_scores else 0.0
    return disease_scores


def _threshold_for_disease(disease, learned_thresholds):
    """Get the learned threshold for a disease, with hardcoded fallbacks."""
    aliases = CLASSIFIER_LABEL_ALIASES.get(disease, [disease])
    for label in aliases:
        if label in learned_thresholds:
            return learned_thresholds[label]
    if disease == "Fracture":
        return 0.50
    if disease == "Nodule":
        return 0.45
    return 0.40


def load_thresholds(thresholds_path):
    """Load learned thresholds from disk, falling back to hardcoded defaults."""
    if os.path.exists(thresholds_path):
        try:
            with open(thresholds_path, 'r') as f:
                thresholds = json.load(f)
            print(f"✅ Loaded learned thresholds from {thresholds_path}")
            return thresholds
        except Exception:
            pass
    # Fallback hardcoded thresholds
    return {
        "Cardiomegaly": 0.40, "Pleural Effusion": 0.40, "Edema": 0.40,
        "Pneumothorax": 0.40, "Infiltrate": 0.40, "Consolidation": 0.40,
        "Lung Opacity": 0.40, "Nodule": 0.45, "Atelectasis": 0.40, "Fracture": 0.50
    }


# ---------------------------------------------------------
# VIEW FUSION
# ---------------------------------------------------------
FUSION_WEIGHTS_PATH = "model_weights/KG_Agent/biomed_clip/fusion_weights.json"
_FUSION_WEIGHTS = None


def _load_fusion_weights():
    """Lazy-load per-disease fusion weights for frontal/lateral views."""
    global _FUSION_WEIGHTS
    if _FUSION_WEIGHTS is not None:
        return _FUSION_WEIGHTS
    if os.path.exists(FUSION_WEIGHTS_PATH):
        try:
            with open(FUSION_WEIGHTS_PATH, 'r') as f:
                _FUSION_WEIGHTS = json.load(f)
            print(f"✅ Loaded fusion weights from {FUSION_WEIGHTS_PATH}")
            return _FUSION_WEIGHTS
        except Exception:
            pass
    # Default: slightly favor frontal view (standard diagnostic view)
    _FUSION_WEIGHTS = {d: {"frontal": 0.65, "lateral": 0.35} for d in DISEASES}
    return _FUSION_WEIGHTS


def fuse_view_findings(frontal_findings: dict, lateral_findings: dict) -> dict:
    """
    Fuse disease scores from frontal and lateral views using learned
    per-disease weights. If only one view is available, returns that view's
    findings directly.
    """
    if not lateral_findings:
        return frontal_findings
    if not frontal_findings:
        return lateral_findings

    weights = _load_fusion_weights()
    combined = {}
    all_diseases = set(frontal_findings.keys()) | set(lateral_findings.keys())

    for disease in all_diseases:
        f_score = frontal_findings.get(disease, 0.0)
        l_score = lateral_findings.get(disease, 0.0)
        w = weights.get(disease, {"frontal": 0.65, "lateral": 0.35})
        combined[disease] = w["frontal"] * f_score + w["lateral"] * l_score

    return combined
