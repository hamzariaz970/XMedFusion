from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np
import torch
from dotenv import load_dotenv
from PIL import Image
from transformers import AutoImageProcessor
try:
    import cv2
except ImportError:
    cv2 = None

load_dotenv(dotenv_path=Path(__file__).with_name(".env"))

from train_rad_dino_kg_agent import DISEASES, KG_LABEL_POLICY, RadDinoKGClassifier
from train_xrv_rad_dino_ensemble_kg_agent import (
    ANATOMY_BY_DISEASE,
    ENSEMBLE_CONFIG,
    apply_ensemble_weights,
    build_xrv_mapping,
    fuse_view_scores,
    load_rad_model,
    load_xrv_model,
    map_xrv_scores,
    recommended_rad_variant,
)


SAVE_DIR = Path(ENSEMBLE_CONFIG["save_dir"])
THRESHOLDS_PATH = SAVE_DIR / "thresholds.json"
POLICY_PATH = SAVE_DIR / "kg_label_policy.json"
WEIGHTS_PATH = SAVE_DIR / "ensemble_weights.json"
AUDIT_PATH = SAVE_DIR / "kg_reliability_audit.json"
SUPPORT_POLICY_PATH = Path(os.getenv("KG_SUPPORT_POLICY_PATH", SAVE_DIR / "kg_support_policy.json"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PREPROCESS_VARIANTS = {
    "baseline": {
        "apply_clahe": False,
        "square_pad": False,
        "rad_size": None,
        "xrv_center_crop": True,
    },
    "clahe": {
        "apply_clahe": True,
        "square_pad": False,
        "rad_size": None,
        "xrv_center_crop": True,
    },
    "square_clahe": {
        "apply_clahe": True,
        "square_pad": True,
        "rad_size": None,
        "xrv_center_crop": True,
    },
    "rad_384": {
        "apply_clahe": False,
        "square_pad": False,
        "rad_size": 384,
        "xrv_center_crop": True,
    },
    "xrv_resize_only": {
        "apply_clahe": False,
        "square_pad": False,
        "rad_size": None,
        "xrv_center_crop": False,
    },
}


def _load_json(path: Path, fallback: Dict, *, required: bool = False) -> Dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    if required:
        raise FileNotFoundError(
            f"Required ensemble artifact is missing or unreadable: {path}. "
            "Run train_xrv_rad_dino_ensemble_kg_agent.py first."
        )
    return fallback


def load_thresholds() -> Dict[str, float]:
    fallback = {
        disease: float(KG_LABEL_POLICY[disease]["fallback_threshold"])
        for disease in DISEASES
    }
    return _load_json(THRESHOLDS_PATH, fallback, required=True)


def load_policy(policy_mode: str = "strict") -> Dict:
    policy = _load_json(POLICY_PATH, KG_LABEL_POLICY, required=True)
    if policy_mode == "validation":
        return policy
    if policy_mode != "strict":
        raise ValueError("policy_mode must be either 'strict' or 'validation'.")

    audit = _load_json(AUDIT_PATH, {}, required=True)
    strict_policy = json.loads(json.dumps(policy))
    for disease in DISEASES:
        disease_audit = audit.get(disease, {})
        target_met = bool(disease_audit.get("meets_target_on_this_split", False))
        strict_policy[disease]["heldout_target_met"] = target_met
        strict_policy[disease]["heldout_edge_precision"] = disease_audit.get("edge_precision")
        strict_policy[disease]["kg_enabled"] = bool(strict_policy[disease]["kg_enabled"] and target_met)
    return strict_policy


def load_ensemble_weights() -> np.ndarray:
    payload = _load_json(WEIGHTS_PATH, {}, required=True)
    weights_by_label = payload.get("weights", {}) if isinstance(payload, dict) else {}
    weights = [
        float(weights_by_label.get(disease, 0.5))
        for disease in DISEASES
    ]
    return np.asarray(weights, dtype=np.float32)


def load_support_policy() -> Dict:
    """Load optional recall-oriented thresholds for uncertain KG evidence.

    Strict present edges still use the precision policy. This artifact only
    allows a finding to be surfaced as uncertain/candidate evidence when the
    high-confidence KG edge policy would otherwise mark it absent.
    """
    return _load_json(SUPPORT_POLICY_PATH, {}, required=False)


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
    return normalized[: int(ENSEMBLE_CONFIG["max_views"])]


def _square_pad(image: Image.Image) -> Image.Image:
    width, height = image.size
    if width == height:
        return image
    side = max(width, height)
    canvas = Image.new("L", (side, side), color=0)
    left = (side - width) // 2
    top = (side - height) // 2
    canvas.paste(image.convert("L"), (left, top))
    return canvas.convert("RGB")


def _clahe_image(image: Image.Image) -> Image.Image:
    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    if cv2 is None:
        return Image.fromarray(gray).convert("RGB")
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return Image.fromarray(clahe.apply(gray)).convert("RGB")


def _apply_variant(image: Image.Image, variant: str) -> Image.Image:
    config = PREPROCESS_VARIANTS[variant]
    working = image
    if config["square_pad"]:
        working = _square_pad(working)
    if config["apply_clahe"]:
        working = _clahe_image(working)
    return working.convert("RGB")


def _load_images(image_paths: Sequence[Path], preprocess_variant: str) -> List[Image.Image]:
    return [_apply_variant(Image.open(path).convert("RGB"), preprocess_variant) for path in image_paths]


def _pad_views(tensors: List[torch.Tensor], max_views: int) -> List[torch.Tensor]:
    while len(tensors) < max_views:
        tensors.append(tensors[-1].clone())
    return tensors[:max_views]


def _rad_batch(images: Sequence[Image.Image], processor, preprocess_variant: str) -> torch.Tensor:
    rad_size = PREPROCESS_VARIANTS[preprocess_variant]["rad_size"]
    tensors = [
        processor(
            images=image.resize((rad_size, rad_size), Image.BICUBIC) if rad_size else image,
            return_tensors="pt",
        )["pixel_values"].squeeze(0)
        for image in images
    ]
    tensors = _pad_views(tensors, int(ENSEMBLE_CONFIG["max_views"]))
    return torch.stack(tensors, dim=0).unsqueeze(0).to(DEVICE)


def _xrv_batch(images: Sequence[Image.Image], xrv, preprocess_variant: str) -> torch.Tensor:
    # TorchXRayVision transforms are numpy-callable objects, so keep them as
    # plain callables rather than torch modules.
    crop = xrv.datasets.XRayCenterCrop() if PREPROCESS_VARIANTS[preprocess_variant]["xrv_center_crop"] else None
    resize = xrv.datasets.XRayResizer(224)

    tensors = []
    for image in images:
        img = np.asarray(image.convert("L"), dtype=np.float32)
        img = xrv.datasets.normalize(img, 255)
        img = img[None, :, :]
        if crop is not None:
            img = crop(img)
        img = resize(img)
        if isinstance(img, torch.Tensor):
            tensor = img.float()
        else:
            tensor = torch.from_numpy(np.asarray(img)).float()
        tensors.append(tensor)

    tensors = _pad_views(tensors, int(ENSEMBLE_CONFIG["max_views"]))
    return torch.stack(tensors, dim=0).unsqueeze(0).to(DEVICE)


def load_ensemble_agent(rad_variant: str | None = None, preprocess_variant: str = "baseline"):
    rad_variant = rad_variant or recommended_rad_variant()
    if preprocess_variant not in PREPROCESS_VARIANTS:
        raise ValueError(f"Unknown preprocess variant: {preprocess_variant}")
    rad_processor = AutoImageProcessor.from_pretrained(
        ENSEMBLE_CONFIG["rad_dino_model_name"],
        use_fast=False,
    )
    rad_model = load_rad_model(rad_variant)
    xrv, xrv_model = load_xrv_model()
    xrv_pathologies = list(getattr(xrv_model, "pathologies", getattr(xrv.datasets, "default_pathologies", [])))
    xrv_mapping = build_xrv_mapping(xrv_pathologies)
    return {
        "rad_variant": rad_variant,
        "rad_processor": rad_processor,
        "rad_model": rad_model,
        "xrv": xrv,
        "xrv_model": xrv_model,
        "xrv_mapping": xrv_mapping,
        "weights": load_ensemble_weights(),
        "preprocess_variant": preprocess_variant,
    }


def _predict_component_scores(image_paths: Sequence[Path], agent: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    images = _load_images(image_paths, agent["preprocess_variant"])

    with torch.no_grad():
        rad_logits = agent["rad_model"](_rad_batch(images, agent["rad_processor"], agent["preprocess_variant"]))
        rad_scores = torch.sigmoid(rad_logits).detach().cpu().numpy()

        xrv_images = _xrv_batch(images, agent["xrv"], agent["preprocess_variant"])
        batch_size, num_views = xrv_images.shape[:2]
        raw = agent["xrv_model"](xrv_images.reshape(batch_size * num_views, *xrv_images.shape[2:]))
        if raw.min().item() < 0.0 or raw.max().item() > 1.0:
            raw = torch.sigmoid(raw)
        raw = raw.reshape(batch_size, num_views, -1)
        fused_raw = fuse_view_scores(raw).detach().cpu().numpy()
        xrv_scores = map_xrv_scores(fused_raw, agent["xrv_mapping"])

    ensemble_scores = apply_ensemble_weights(rad_scores, xrv_scores, agent["weights"])
    return rad_scores.squeeze(0), xrv_scores.squeeze(0), ensemble_scores.squeeze(0)


def _kg_status_for_score(
    disease: str,
    score: float,
    threshold: float,
    policy: Dict,
    support_policy: Dict | None = None,
) -> Tuple[str, bool, str]:
    if score >= threshold and policy[disease]["kg_enabled"]:
        return "present", True, "precision_policy_present"
    support_detail = (support_policy or {}).get(disease, {})
    support_threshold = support_detail.get("support_threshold")
    if bool(support_detail.get("support_enabled", False)) and support_threshold is not None:
        if score >= float(support_threshold):
            return "uncertain", False, "recall_support_threshold"
    if score >= threshold and not policy[disease]["kg_enabled"]:
        return "uncertain", False, "precision_policy_blocked"
    return "absent", False, "below_threshold"


def predict_pathologies(
    image: Union[str, Path, Sequence[Union[str, Path]]],
    *,
    agent: Dict | None = None,
    rad_variant: str | None = None,
    policy_mode: str = "strict",
    preprocess_variant: str = "baseline",
) -> Dict:
    owns_agent = agent is None
    agent = agent or load_ensemble_agent(rad_variant, preprocess_variant=preprocess_variant)

    thresholds = load_thresholds()
    policy = load_policy(policy_mode)
    support_policy = load_support_policy()
    image_paths = _normalize_image_inputs(image)
    rad_scores, xrv_scores, ensemble_scores = _predict_component_scores(image_paths, agent)

    findings = {}
    present_for_kg = []
    uncertain_not_in_kg = []
    for idx, disease in enumerate(DISEASES):
        score = float(ensemble_scores[idx])
        threshold = float(thresholds.get(disease, policy[disease]["fallback_threshold"]))
        support_detail = support_policy.get(disease, {}) if isinstance(support_policy, dict) else {}
        status, create_edge, status_reason = _kg_status_for_score(
            disease,
            score,
            threshold,
            policy,
            support_policy,
        )
        finding = {
            "score": score,
            "threshold": threshold,
            "support_threshold": support_detail.get("support_threshold"),
            "support_enabled": bool(support_detail.get("support_enabled", False)),
            "support_policy": support_detail,
            "status": status,
            "status_reason": status_reason,
            "create_kg_edge": create_edge,
            "kg_enabled": bool(policy[disease]["kg_enabled"]),
            "target_precision": float(policy[disease]["target_precision"]),
            "heldout_target_met": bool(policy[disease].get("heldout_target_met", False)),
            "heldout_edge_precision": policy[disease].get("heldout_edge_precision"),
            "rad_dino_score": float(rad_scores[idx]),
            "xrv_score": float(xrv_scores[idx]),
            "rad_dino_weight": float(agent["weights"][idx]),
            "xrv_weight": float(1.0 - agent["weights"][idx]),
        }
        findings[disease] = finding
        if create_edge:
            present_for_kg.append(disease)
        elif status == "uncertain":
            uncertain_not_in_kg.append(disease)

    if owns_agent and DEVICE == "cuda":
        agent["rad_model"].to("cpu")
        agent["xrv_model"].to("cpu")
        torch.cuda.empty_cache()

    return {
        "variant": "rad_dino_xrv_ensemble",
        "policy_mode": policy_mode,
        "rad_dino_variant": agent["rad_variant"],
        "preprocess_variant": agent["preprocess_variant"],
        "support_policy_path": str(SUPPORT_POLICY_PATH) if SUPPORT_POLICY_PATH.exists() else None,
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

    for disease in prediction.get("uncertain_not_in_kg", []):
        anatomy = ANATOMY_BY_DISEASE.get(disease, "chest")
        if anatomy not in anatomy_to_idx:
            anatomy_to_idx[anatomy] = len(entities)
            entities.append([anatomy, "Anatomy"])

        obs_idx = len(entities)
        entities.append([disease.lower(), "UncertainObservation"])
        relations.append([obs_idx, anatomy_to_idx[anatomy], "possible_at"])

    if not entities:
        entities.append(["chest", "Anatomy"])
        entities.append(["clear", "Observation"])
        relations.append([1, 0, "modify"])

    return {
        "entities": entities,
        "relations": relations,
        "metadata": {
            "kg_agent_variant": prediction["variant"],
            "policy_mode": prediction["policy_mode"],
            "rad_dino_variant": prediction["rad_dino_variant"],
            "preprocess_variant": prediction.get("preprocess_variant", "baseline"),
            "support_policy_path": prediction.get("support_policy_path"),
            "present_for_kg": prediction["present_for_kg"],
            "uncertain_not_in_kg": prediction["uncertain_not_in_kg"],
            "findings": prediction["findings"],
        },
    }


def infer_kg(
    image: Union[str, Path, Sequence[Union[str, Path]]],
    *,
    agent: Dict | None = None,
    rad_variant: str | None = None,
    policy_mode: str = "strict",
    preprocess_variant: str = "baseline",
    debug: bool = True,
) -> Dict:
    if debug:
        print("[RAD-DINO + XRV KG] infer_kg called", flush=True)
    prediction = predict_pathologies(
        image,
        agent=agent,
        rad_variant=rad_variant,
        policy_mode=policy_mode,
        preprocess_variant=preprocess_variant,
    )
    if debug:
        print(f"[RAD-DINO + XRV KG] present_for_kg={prediction['present_for_kg']}", flush=True)
        print(f"[RAD-DINO + XRV KG] uncertain_not_in_kg={prediction['uncertain_not_in_kg']}", flush=True)
    return predictions_to_kg(prediction)


if __name__ == "__main__":
    annotation_path = Path(ENSEMBLE_CONFIG["annotation_file"])
    image_root = Path(ENSEMBLE_CONFIG["data_dir"])
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    sample = data["test"][0]
    image_paths = [image_root / rel_path for rel_path in sample["image_path"]]
    print(json.dumps(infer_kg(image_paths, debug=True), indent=2))
