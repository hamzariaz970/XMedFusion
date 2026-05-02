"""
RAD-DINO + TorchXRayVision Ensemble KG Agent
===========================================

This script does not train another deep model. It calibrates an ensemble between:

1. The IU-finetuned RAD-DINO classifier from train_rad_dino_kg_agent.py
2. TorchXRayVision densenet121-res224-all, a supervised multi-dataset CXR
   pathology classifier

The output is a KG-oriented ensemble policy:
  - label-specific RAD-DINO/XRV weights selected on IU validation AP
  - precision-oriented validation thresholds
  - strict KG gating: a label must meet its validation precision target to
    create a KG edge
  - validation/test metrics, prediction analysis, and KG-ready JSON dumps

Run from backend/:
    python train_xrv_rad_dino_ensemble_kg_agent.py
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import average_precision_score, classification_report, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoImageProcessor

from train_rad_dino_kg_agent import (
    CONFIG as RAD_CONFIG,
    DISEASES,
    KG_LABEL_POLICY,
    RadDinoKGClassifier,
    build_effective_kg_policy,
    calculate_metrics,
    dump_label_summary,
    dump_predictions,
    json_safe_metrics,
    label_audit_from_report,
    load_checkpoint,
    optimize_thresholds,
    resolve_checkpoint_dir,
)


ENSEMBLE_CONFIG = {
    "annotation_file": RAD_CONFIG["annotation_file"],
    "data_dir": RAD_CONFIG["data_dir"],
    "save_dir": "model_weights/KG_Agent/rad_dino_xrv_ensemble",
    "rad_dino_save_dir": RAD_CONFIG["save_dir"],
    "rad_dino_model_name": RAD_CONFIG["model_name"],
    "xrv_model_name": "densenet121-res224-all",
    "batch_size": 8,
    "num_workers": 0,
    "max_views": int(RAD_CONFIG["max_views"]),
    "primary_view_weight": float(RAD_CONFIG["primary_view_weight"]),
    "per_view_mean_weight": 0.75,
    "per_view_max_weight": 0.25,
    "weight_grid": [0.0, 0.25, 0.5, 0.75, 1.0],
    "rare_label_rad_weight": 0.25,
    "stable_metric_min_positives": int(RAD_CONFIG["stable_metric_min_positives"]),
    "prediction_dump_top_k": int(RAD_CONFIG["prediction_dump_top_k"]),
}


def apply_env_overrides() -> None:
    env_map = {
        "KG_ENSEMBLE_SAVE_DIR": ("save_dir", str),
        "KG_RAD_SAVE_DIR": ("rad_dino_save_dir", str),
        "KG_ENSEMBLE_BATCH_SIZE": ("batch_size", int),
        "KG_ENSEMBLE_RARE_LABEL_RAD_WEIGHT": ("rare_label_rad_weight", float),
    }
    for env_name, (config_key, caster) in env_map.items():
        value = os.getenv(env_name)
        if value is not None:
            ENSEMBLE_CONFIG[config_key] = caster(value)


apply_env_overrides()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


ANATOMY_BY_DISEASE = {
    "Cardiomegaly": "mediastinum",
    "Pleural Effusion": "pleural space",
    "Edema": "lungs",
    "Pneumothorax": "pleural space",
    "Infiltrate": "lungs",
    "Consolidation": "lungs",
    "Lung Opacity": "lungs",
    "Nodule": "lungs",
    "Atelectasis": "lungs",
    "Fracture": "bones",
}


def import_xrv():
    try:
        import torchxrayvision as xrv
    except ImportError as exc:
        raise ImportError(
            "TorchXRayVision is required for the ensemble. Install it in fyp_env with:\n"
            "  pip install torchxrayvision"
        ) from exc
    return xrv


def recommended_rad_variant() -> str:
    comparison_path = Path(ENSEMBLE_CONFIG["rad_dino_save_dir"]) / "checkpoint_comparison.json"
    if comparison_path.exists():
        try:
            data = json.loads(comparison_path.read_text(encoding="utf-8"))
            variant = data.get("recommended_variant_by_validation_policy")
            if variant in RAD_CONFIG["checkpoint_variants"]:
                return variant
        except Exception:
            pass
    return "selected"


def load_rad_model(variant: str | None = None) -> RadDinoKGClassifier:
    variant = variant or recommended_rad_variant()
    checkpoint_dir = resolve_checkpoint_dir(ENSEMBLE_CONFIG["rad_dino_save_dir"], variant)
    head_path = Path(checkpoint_dir) / "rad_dino_head_best.pth"
    if not head_path.exists():
        raise FileNotFoundError(
            f"RAD-DINO checkpoint not found at {head_path}. Run train_rad_dino_kg_agent.py first."
        )
    model = RadDinoKGClassifier(
        ENSEMBLE_CONFIG["rad_dino_model_name"],
        len(DISEASES),
        unfreeze_blocks=int(RAD_CONFIG["unfreeze_blocks"]),
    ).to(DEVICE)
    load_checkpoint(model, ENSEMBLE_CONFIG["rad_dino_save_dir"], DEVICE, variant)
    model.eval()
    return model


def load_xrv_model():
    xrv = import_xrv()
    try:
        get_model = getattr(xrv.models, "get_model")
        model = get_model(ENSEMBLE_CONFIG["xrv_model_name"], from_hf_hub=True)
    except Exception:
        model = xrv.models.DenseNet(weights=ENSEMBLE_CONFIG["xrv_model_name"])
    model = model.to(DEVICE)
    model.eval()
    return xrv, model


def normalize_xrv_key(name: str) -> str:
    return name.lower().replace("_", " ").replace("-", " ").strip()


def build_xrv_mapping(pathologies: Sequence[str]) -> Dict[str, List[int]]:
    normalized = {normalize_xrv_key(name): idx for idx, name in enumerate(pathologies)}

    def indices(*names: str) -> List[int]:
        return [normalized[name] for name in names if name in normalized]

    return {
        "Cardiomegaly": indices("cardiomegaly", "enlarged cardiomediastinum"),
        "Pleural Effusion": indices("effusion"),
        "Edema": indices("edema"),
        "Pneumothorax": indices("pneumothorax"),
        "Infiltrate": indices("infiltration", "pneumonia"),
        "Consolidation": indices("consolidation"),
        "Lung Opacity": indices("lung opacity", "infiltration", "consolidation"),
        "Nodule": indices("nodule", "mass", "lung lesion"),
        "Atelectasis": indices("atelectasis"),
        "Fracture": indices("fracture"),
    }


def map_xrv_scores(raw_scores: np.ndarray, mapping: Dict[str, List[int]]) -> np.ndarray:
    mapped = np.zeros((raw_scores.shape[0], len(DISEASES)), dtype=np.float32)
    for idx, disease in enumerate(DISEASES):
        source_indices = mapping.get(disease, [])
        if source_indices:
            mapped[:, idx] = raw_scores[:, source_indices].max(axis=1)
    return mapped


class EnsembleCXRDataset(Dataset):
    def __init__(self, split: str, rad_processor, xrv_module):
        self.split = split
        self.rad_processor = rad_processor
        self.samples: List[Dict] = []
        self.max_views = int(ENSEMBLE_CONFIG["max_views"])
        self.xrv = xrv_module
        self.xrv_transform = transforms.Compose([
            self.xrv.datasets.XRayCenterCrop(),
            self.xrv.datasets.XRayResizer(224),
        ])

        annotation_path = Path(ENSEMBLE_CONFIG["annotation_file"])
        raw_data = json.loads(annotation_path.read_text(encoding="utf-8"))
        if split not in raw_data:
            raise ValueError(f"Split '{split}' missing from {annotation_path}")

        img_dir = Path(ENSEMBLE_CONFIG["data_dir"])
        print(f"Loading {len(raw_data[split])} {split} studies for ensemble evaluation...")
        for item in tqdm(raw_data[split]):
            rel_paths = item.get("image_path", [])
            if isinstance(rel_paths, str):
                rel_paths = [rel_paths]
            full_paths = [img_dir / rel_path for rel_path in rel_paths if rel_path]
            full_paths = [path for path in full_paths if self._is_valid_image(path)]
            if not full_paths:
                continue
            report = item.get("report", "")
            labels, audit = label_audit_from_report(report)
            self.samples.append({
                "id": item.get("id", f"{split}_{len(self.samples)}"),
                "paths": [str(path) for path in full_paths[: self.max_views]],
                "labels": labels,
                "label_audit": audit,
                "report": report,
            })
        self.print_label_distribution()

    @staticmethod
    def _is_valid_image(path: Path) -> bool:
        if not path.exists():
            return False
        try:
            with Image.open(path) as image:
                image.verify()
            return True
        except Exception:
            return False

    def _rad_tensor(self, image: Image.Image) -> torch.Tensor:
        return self.rad_processor(images=image.convert("RGB"), return_tensors="pt")["pixel_values"].squeeze(0)

    def _xrv_tensor(self, image: Image.Image) -> torch.Tensor:
        img = np.asarray(image.convert("L"), dtype=np.float32)
        img = self.xrv.datasets.normalize(img, 255)
        img = img[None, :, :]
        img = self.xrv_transform(img)
        if isinstance(img, torch.Tensor):
            return img.float()
        return torch.from_numpy(np.asarray(img)).float()

    def print_label_distribution(self) -> None:
        labels = np.array([sample["labels"] for sample in self.samples], dtype=np.float32)
        print(f"{self.split.capitalize()} label distribution:")
        for idx, disease in enumerate(DISEASES):
            count = int(labels[:, idx].sum()) if len(labels) else 0
            pct = 100.0 * count / max(1, len(labels))
            print(f"  {disease:>20s}: {count:4d} ({pct:5.1f}%)")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        rad_tensors = []
        xrv_tensors = []
        for path in sample["paths"]:
            image = Image.open(path).convert("RGB")
            rad_tensors.append(self._rad_tensor(image))
            xrv_tensors.append(self._xrv_tensor(image))
        while len(rad_tensors) < self.max_views:
            rad_tensors.append(rad_tensors[-1].clone())
            xrv_tensors.append(xrv_tensors[-1].clone())
        return (
            torch.stack(rad_tensors[: self.max_views], dim=0),
            torch.stack(xrv_tensors[: self.max_views], dim=0),
            torch.tensor(sample["labels"], dtype=torch.float32),
        )


def fuse_view_scores(view_scores: torch.Tensor) -> torch.Tensor:
    if view_scores.dim() == 2:
        return view_scores
    batch_size, num_views, _ = view_scores.shape
    primary_weight = float(ENSEMBLE_CONFIG["primary_view_weight"])
    aux_weight = max(0.0, 1.0 - primary_weight)
    weights = torch.full(
        (batch_size, num_views),
        aux_weight / max(1, num_views - 1),
        dtype=view_scores.dtype,
        device=view_scores.device,
    )
    weights[:, 0] = primary_weight
    weighted_mean = (view_scores * weights.unsqueeze(-1)).sum(dim=1)
    max_score = view_scores.max(dim=1).values
    return (
        float(ENSEMBLE_CONFIG["per_view_mean_weight"]) * weighted_mean
        + float(ENSEMBLE_CONFIG["per_view_max_weight"]) * max_score
    )


def evaluate_components(
    rad_model: RadDinoKGClassifier,
    xrv_model,
    xrv_mapping: Dict[str, List[int]],
    loader: DataLoader,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rad_model.eval()
    xrv_model.eval()
    rad_scores, xrv_scores, labels = [], [], []

    with torch.no_grad():
        for rad_images, xrv_images, y in loader:
            batch_size = rad_images.shape[0]
            num_views = rad_images.shape[1]

            rad_images = rad_images.to(DEVICE)
            rad_logits = rad_model(rad_images)
            rad_scores.append(torch.sigmoid(rad_logits).cpu().numpy())

            xrv_images = xrv_images.to(DEVICE)
            flat_xrv = xrv_images.reshape(batch_size * num_views, *xrv_images.shape[2:])
            raw = xrv_model(flat_xrv)
            if raw.min().item() < 0.0 or raw.max().item() > 1.0:
                raw = torch.sigmoid(raw)
            raw = raw.view(batch_size, num_views, -1)
            fused_raw = fuse_view_scores(raw).cpu().numpy()
            xrv_scores.append(map_xrv_scores(fused_raw, xrv_mapping))
            labels.append(y.numpy())

    return np.vstack(rad_scores), np.vstack(xrv_scores), np.vstack(labels)


def select_ensemble_weights(rad_scores: np.ndarray, xrv_scores: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, Dict]:
    weights = np.zeros(len(DISEASES), dtype=np.float32)
    details = {}
    grid = [float(w) for w in ENSEMBLE_CONFIG["weight_grid"]]

    for idx, disease in enumerate(DISEASES):
        positives = int(labels[:, idx].sum())
        if positives < int(KG_LABEL_POLICY[disease]["min_val_positives"]):
            weights[idx] = float(ENSEMBLE_CONFIG["rare_label_rad_weight"])
            details[disease] = {
                "rad_weight": float(weights[idx]),
                "selection": "rare_label_default",
                "val_positives": positives,
                "ap_by_weight": None,
            }
            continue

        ap_by_weight = {}
        best_weight = 0.5
        best_ap = -1.0
        for weight in grid:
            score = weight * rad_scores[:, idx] + (1.0 - weight) * xrv_scores[:, idx]
            ap = average_precision_score(labels[:, idx], score) if positives else 0.0
            ap_by_weight[str(weight)] = float(ap)
            if ap > best_ap:
                best_ap = float(ap)
                best_weight = float(weight)
        weights[idx] = best_weight
        details[disease] = {
            "rad_weight": best_weight,
            "xrv_weight": 1.0 - best_weight,
            "selection": "best_validation_ap",
            "val_positives": positives,
            "best_validation_ap": best_ap,
            "ap_by_weight": ap_by_weight,
        }
    return weights, details


def apply_ensemble_weights(rad_scores: np.ndarray, xrv_scores: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return weights.reshape(1, -1) * rad_scores + (1.0 - weights.reshape(1, -1)) * xrv_scores


def write_metrics(
    path: str,
    split: str,
    rad_scores: np.ndarray,
    xrv_scores: np.ndarray,
    ensemble_scores: np.ndarray,
    labels: np.ndarray,
    thresholds: Dict[str, float],
    threshold_details: Dict,
    policy: Dict,
) -> None:
    payload = {
        "split": split,
        "models": {
            "rad_dino": ENSEMBLE_CONFIG["rad_dino_model_name"],
            "torchxrayvision": ENSEMBLE_CONFIG["xrv_model_name"],
        },
        "component_metrics": {
            "rad_dino": calculate_metrics(rad_scores, labels),
            "torchxrayvision": calculate_metrics(xrv_scores, labels),
            "ensemble_fixed_0_5": calculate_metrics(ensemble_scores, labels),
            "ensemble_precision_policy": calculate_metrics(ensemble_scores, labels, thresholds),
        },
        "thresholds": thresholds,
        "threshold_details": threshold_details,
        "kg_policy": policy,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_safe_metrics(payload), f, indent=2)


def dump_component_scores(
    dataset: EnsembleCXRDataset,
    rad_scores: np.ndarray,
    xrv_scores: np.ndarray,
    ensemble_scores: np.ndarray,
    save_dir: str,
    prefix: str,
) -> str:
    path = os.path.join(save_dir, f"{prefix}_component_scores.csv")
    fieldnames = ["sample_id", "image_paths"]
    for disease in DISEASES:
        fieldnames.extend([f"{disease}_rad", f"{disease}_xrv", f"{disease}_ensemble"])
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for sample, rad_row, xrv_row, ensemble_row in zip(dataset.samples, rad_scores, xrv_scores, ensemble_scores):
            row = {
                "sample_id": sample["id"],
                "image_paths": "|".join(sample["paths"]),
            }
            for idx, disease in enumerate(DISEASES):
                row[f"{disease}_rad"] = float(rad_row[idx])
                row[f"{disease}_xrv"] = float(xrv_row[idx])
                row[f"{disease}_ensemble"] = float(ensemble_row[idx])
            writer.writerow(row)
    return path


def write_kg_reliability_audit(
    save_dir: str,
    labels: np.ndarray,
    scores: np.ndarray,
    thresholds: Dict[str, float],
    policy: Dict,
) -> str:
    audit = {}
    for idx, disease in enumerate(DISEASES):
        threshold = float(thresholds[disease])
        pred = (scores[:, idx] >= threshold).astype(int)
        truth = labels[:, idx].astype(int)
        enabled = bool(policy[disease]["kg_enabled"])
        edge_pred = pred if enabled else np.zeros_like(pred)

        tp = int(((edge_pred == 1) & (truth == 1)).sum())
        fp = int(((edge_pred == 1) & (truth == 0)).sum())
        fn = int(((edge_pred == 0) & (truth == 1)).sum())
        predicted_edges = int(edge_pred.sum())
        support = int(truth.sum())
        precision = tp / (tp + fp) if (tp + fp) else None
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision is not None and precision + recall > 0
            else 0.0
        )

        audit[disease] = {
            "kg_enabled": enabled,
            "threshold": threshold,
            "support": support,
            "predicted_edges": predicted_edges,
            "true_positive_edges": tp,
            "false_positive_edges": fp,
            "false_negative_labels": fn,
            "edge_precision": precision,
            "edge_recall": recall,
            "edge_f1": f1,
            "target_precision": float(policy[disease]["target_precision"]),
            "meets_target_on_this_split": (
                precision is not None and precision >= float(policy[disease]["target_precision"])
            ),
        }

    path = os.path.join(save_dir, "kg_reliability_audit.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_safe_metrics(audit), f, indent=2)
    return path


def print_report(title: str, labels: np.ndarray, scores: np.ndarray, thresholds: Dict[str, float]) -> None:
    pred = np.column_stack([
        (scores[:, idx] >= thresholds[disease]).astype(int)
        for idx, disease in enumerate(DISEASES)
    ])
    print(f"\n{title}")
    print(classification_report(labels, pred, target_names=DISEASES, zero_division=0))


def main() -> None:
    os.makedirs(ENSEMBLE_CONFIG["save_dir"], exist_ok=True)
    with open(os.path.join(ENSEMBLE_CONFIG["save_dir"], "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(ENSEMBLE_CONFIG, f, indent=2)

    xrv, xrv_model = load_xrv_model()
    xrv_pathologies = list(getattr(xrv_model, "pathologies", getattr(xrv.datasets, "default_pathologies", [])))
    xrv_mapping = build_xrv_mapping(xrv_pathologies)
    with open(os.path.join(ENSEMBLE_CONFIG["save_dir"], "xrv_mapping.json"), "w", encoding="utf-8") as f:
        json.dump({"pathologies": xrv_pathologies, "mapping": xrv_mapping}, f, indent=2)

    rad_processor = AutoImageProcessor.from_pretrained(ENSEMBLE_CONFIG["rad_dino_model_name"], use_fast=False)
    train_dataset = EnsembleCXRDataset("train", rad_processor, xrv)
    val_dataset = EnsembleCXRDataset("val", rad_processor, xrv)
    test_dataset = EnsembleCXRDataset("test", rad_processor, xrv)
    label_summary_path = dump_label_summary([train_dataset, val_dataset, test_dataset], ENSEMBLE_CONFIG["save_dir"])
    print(f"Label summary saved to {label_summary_path}")

    val_loader = DataLoader(
        val_dataset,
        batch_size=ENSEMBLE_CONFIG["batch_size"],
        shuffle=False,
        num_workers=ENSEMBLE_CONFIG["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=ENSEMBLE_CONFIG["batch_size"],
        shuffle=False,
        num_workers=ENSEMBLE_CONFIG["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )

    rad_variant = recommended_rad_variant()
    print(f"Loading RAD-DINO variant for ensemble: {rad_variant}")
    rad_model = load_rad_model(rad_variant)

    print("\nScoring validation split...")
    val_rad, val_xrv, val_labels = evaluate_components(rad_model, xrv_model, xrv_mapping, val_loader)
    print("Scoring test split...")
    test_rad, test_xrv, test_labels = evaluate_components(rad_model, xrv_model, xrv_mapping, test_loader)

    weights, weight_details = select_ensemble_weights(val_rad, val_xrv, val_labels)
    val_ensemble = apply_ensemble_weights(val_rad, val_xrv, weights)
    test_ensemble = apply_ensemble_weights(test_rad, test_xrv, weights)

    thresholds, threshold_details = optimize_thresholds(val_ensemble, val_labels)
    effective_policy = build_effective_kg_policy(threshold_details)

    thresholds_path = os.path.join(ENSEMBLE_CONFIG["save_dir"], "thresholds.json")
    weights_path = os.path.join(ENSEMBLE_CONFIG["save_dir"], "ensemble_weights.json")
    policy_path = os.path.join(ENSEMBLE_CONFIG["save_dir"], "kg_label_policy.json")
    with open(thresholds_path, "w", encoding="utf-8") as f:
        json.dump(thresholds, f, indent=2)
    with open(weights_path, "w", encoding="utf-8") as f:
        json.dump({"weights": {disease: float(weights[idx]) for idx, disease in enumerate(DISEASES)}, "details": weight_details}, f, indent=2)
    with open(policy_path, "w", encoding="utf-8") as f:
        json.dump(effective_policy, f, indent=2)

    write_metrics(
        os.path.join(ENSEMBLE_CONFIG["save_dir"], "validation_metrics.json"),
        "val",
        val_rad,
        val_xrv,
        val_ensemble,
        val_labels,
        thresholds,
        threshold_details,
        effective_policy,
    )
    write_metrics(
        os.path.join(ENSEMBLE_CONFIG["save_dir"], "test_metrics.json"),
        "test",
        test_rad,
        test_xrv,
        test_ensemble,
        test_labels,
        thresholds,
        threshold_details,
        effective_policy,
    )

    val_csv, val_analysis, val_kg = dump_predictions(
        val_dataset,
        val_ensemble,
        val_labels,
        thresholds,
        ENSEMBLE_CONFIG["save_dir"],
        "val",
        effective_policy,
    )
    test_csv, test_analysis, test_kg = dump_predictions(
        test_dataset,
        test_ensemble,
        test_labels,
        thresholds,
        ENSEMBLE_CONFIG["save_dir"],
        "test",
        effective_policy,
    )
    val_component_csv = dump_component_scores(val_dataset, val_rad, val_xrv, val_ensemble, ENSEMBLE_CONFIG["save_dir"], "val")
    test_component_csv = dump_component_scores(test_dataset, test_rad, test_xrv, test_ensemble, ENSEMBLE_CONFIG["save_dir"], "test")
    kg_audit_path = write_kg_reliability_audit(
        ENSEMBLE_CONFIG["save_dir"],
        test_labels,
        test_ensemble,
        thresholds,
        effective_policy,
    )

    print_report("Validation ensemble report with precision-policy thresholds", val_labels, val_ensemble, thresholds)
    print_report("Test ensemble report with validation precision-policy thresholds", test_labels, test_ensemble, thresholds)

    print("\n" + "=" * 72)
    print("RAD-DINO + TorchXRayVision ensemble complete.")
    print(f"Artifacts: {ENSEMBLE_CONFIG['save_dir']}")
    print(f"Thresholds: {thresholds_path}")
    print(f"Ensemble weights: {weights_path}")
    print(f"KG policy: {policy_path}")
    print(f"Validation predictions: {val_csv}")
    print(f"Validation analysis: {val_analysis}")
    print(f"Validation KG statuses: {val_kg}")
    print(f"Validation component scores: {val_component_csv}")
    print(f"Test predictions: {test_csv}")
    print(f"Test analysis: {test_analysis}")
    print(f"Test KG statuses: {test_kg}")
    print(f"Test component scores: {test_component_csv}")
    print(f"KG reliability audit: {kg_audit_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
