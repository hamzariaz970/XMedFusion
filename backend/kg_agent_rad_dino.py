from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor

from train_rad_dino_kg_agent import (
    CONFIG,
    DISEASES,
    KG_LABEL_POLICY,
    RadDinoKGClassifier,
    load_checkpoint,
    resolve_checkpoint_dir,
)


SAVE_DIR = Path(CONFIG["save_dir"])
THRESHOLDS_PATH = SAVE_DIR / "thresholds.json"
POLICY_PATH = SAVE_DIR / "kg_label_policy.json"
CHECKPOINT_COMPARISON_PATH = SAVE_DIR / "checkpoint_comparison.json"

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


def _load_json(path: Path, fallback: Dict) -> Dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return fallback


def load_thresholds() -> Dict[str, float]:
    fallback = {
        disease: float(KG_LABEL_POLICY[disease]["fallback_threshold"])
        for disease in DISEASES
    }
    return _load_json(THRESHOLDS_PATH, fallback)


def load_policy() -> Dict:
    return _load_json(POLICY_PATH, KG_LABEL_POLICY)


def _recommended_variant() -> str:
    if CHECKPOINT_COMPARISON_PATH.exists():
        try:
            data = json.loads(CHECKPOINT_COMPARISON_PATH.read_text(encoding="utf-8"))
            variant = data.get("recommended_variant_by_validation_policy")
            if variant in CONFIG["checkpoint_variants"]:
                return variant
        except Exception:
            pass
    return "selected"


def _normalize_image_inputs(image: Union[str, Path, Sequence[Union[str, Path]]]) -> List[Path]:
    if isinstance(image, (str, Path)):
        paths = [Path(image)]
    else:
        paths = [Path(path) for path in image]
    if not paths:
        raise ValueError("At least one image path is required.")
    normalized = [path.expanduser() for path in paths]
    missing = [str(path) for path in normalized if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Image path(s) not found: {missing}")
    return normalized


def _load_images(image_paths: Sequence[Path]) -> List[Image.Image]:
    return [Image.open(path).convert("RGB") for path in image_paths[: int(CONFIG["max_views"])]]


def load_rad_dino_agent(variant: str | None = None, device: str | None = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    variant = variant or _recommended_variant()
    checkpoint_dir = resolve_checkpoint_dir(str(SAVE_DIR), variant)
    head_path = Path(checkpoint_dir) / "rad_dino_head_best.pth"
    if not head_path.exists():
        raise FileNotFoundError(
            f"RAD-DINO checkpoint not found at {head_path}. Run train_rad_dino_kg_agent.py first."
        )

    processor = AutoImageProcessor.from_pretrained(CONFIG["model_name"], use_fast=False)
    model = RadDinoKGClassifier(
        CONFIG["model_name"],
        len(DISEASES),
        unfreeze_blocks=int(CONFIG["unfreeze_blocks"]),
    ).to(device)
    load_checkpoint(model, str(SAVE_DIR), device, variant)
    model.eval()
    return model, processor, device, variant


def _predict_scores(
    image_paths: Sequence[Path],
    model: RadDinoKGClassifier,
    processor,
    device: str,
) -> Dict[str, float]:
    images = _load_images(image_paths)
    pixel_values = [
        processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
        for image in images
    ]
    while len(pixel_values) < int(CONFIG["max_views"]):
        pixel_values.append(pixel_values[-1].clone())
    batch = torch.stack(pixel_values[: int(CONFIG["max_views"])], dim=0).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(batch)
        probs = torch.sigmoid(logits).squeeze(0).detach().cpu().numpy()
    return {disease: float(probs[idx]) for idx, disease in enumerate(DISEASES)}


def _kg_status_for_score(disease: str, score: float, threshold: float, policy: Dict) -> Tuple[str, bool]:
    if score >= threshold and policy[disease]["kg_enabled"]:
        return "present", True
    if score >= threshold and not policy[disease]["kg_enabled"]:
        return "uncertain", False
    return "absent", False


def predict_pathologies(
    image: Union[str, Path, Sequence[Union[str, Path]]],
    *,
    model: RadDinoKGClassifier | None = None,
    processor=None,
    device: str | None = None,
    variant: str | None = None,
) -> Dict:
    owns_model = model is None or processor is None
    if owns_model:
        model, processor, device, variant = load_rad_dino_agent(variant=variant, device=device)
    else:
        device = device or next(model.parameters()).device.type
        variant = variant or "external"

    thresholds = load_thresholds()
    policy = load_policy()
    image_paths = _normalize_image_inputs(image)
    scores = _predict_scores(image_paths, model, processor, device)

    findings = {}
    present_for_kg = []
    uncertain_not_in_kg = []
    for disease in DISEASES:
        score = scores[disease]
        threshold = float(thresholds.get(disease, policy[disease]["fallback_threshold"]))
        status, create_edge = _kg_status_for_score(disease, score, threshold, policy)
        finding = {
            "score": score,
            "threshold": threshold,
            "status": status,
            "create_kg_edge": create_edge,
            "kg_enabled": bool(policy[disease]["kg_enabled"]),
            "target_precision": float(policy[disease]["target_precision"]),
        }
        findings[disease] = finding
        if create_edge:
            present_for_kg.append(disease)
        elif status == "uncertain":
            uncertain_not_in_kg.append(disease)

    if owns_model and device == "cuda":
        model.to("cpu")
        torch.cuda.empty_cache()

    return {
        "variant": variant,
        "image_paths": [str(path) for path in image_paths],
        "present_for_kg": present_for_kg,
        "uncertain_not_in_kg": uncertain_not_in_kg,
        "findings": findings,
    }


def predictions_to_kg(prediction: Dict) -> Dict:
    entities = []
    relations = []
    anatomy_to_idx = {}

    for disease in prediction["present_for_kg"]:
        anatomy = ANATOMY_BY_DISEASE.get(disease, "chest")
        if anatomy not in anatomy_to_idx:
            anatomy_to_idx[anatomy] = len(entities)
            entities.append([anatomy, "Anatomy"])

        obs_idx = len(entities)
        entities.append([disease.lower(), "Observation"])
        relations.append([obs_idx, anatomy_to_idx[anatomy], "located_at"])

    if not entities:
        entities.append(["chest", "Anatomy"])
        entities.append(["clear", "Observation"])
        relations.append([1, 0, "modify"])

    return {
        "entities": entities,
        "relations": relations,
        "metadata": {
            "rad_dino_variant": prediction["variant"],
            "present_for_kg": prediction["present_for_kg"],
            "uncertain_not_in_kg": prediction["uncertain_not_in_kg"],
            "findings": prediction["findings"],
        },
    }


def infer_kg(
    image: Union[str, Path, Sequence[Union[str, Path]]],
    *,
    model: RadDinoKGClassifier | None = None,
    processor=None,
    device: str | None = None,
    variant: str | None = None,
    debug: bool = True,
) -> Dict:
    if debug:
        print("[RAD-DINO KG] infer_kg called", flush=True)
    prediction = predict_pathologies(
        image,
        model=model,
        processor=processor,
        device=device,
        variant=variant,
    )
    if debug:
        print(f"[RAD-DINO KG] present_for_kg={prediction['present_for_kg']}", flush=True)
        print(f"[RAD-DINO KG] uncertain_not_in_kg={prediction['uncertain_not_in_kg']}", flush=True)
    return predictions_to_kg(prediction)


if __name__ == "__main__":
    annotation_path = Path(CONFIG["annotation_file"])
    image_root = Path(CONFIG["data_dir"])
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    sample = data["test"][0]
    image_paths = [image_root / rel_path for rel_path in sample["image_path"]]
    print(json.dumps(infer_kg(image_paths, debug=True), indent=2))
