from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

from report_labels import DISEASES, certainty_labels_from_report, collect_keyword_evidence, KEYWORD_MAP


ANNOTATIONS_PATH = Path("data/iu_xray/annotation.json")
IMAGES_ROOT = Path("data/iu_xray/images")
ENSEMBLE_DIR = Path("model_weights/KG_Agent/rad_dino_xrv_ensemble")
RESULTS_ROOT = Path("out/rare_label_audit")
BASELINE_EXPERIMENT = Path("out/hard_case_experiments/20260502_145619")
WEAK_DISEASES = [
    "Pleural Effusion",
    "Edema",
    "Pneumothorax",
    "Consolidation",
    "Nodule",
    "Fracture",
]

SUPPORT_POLICY_TARGETS = {
    "Pleural Effusion": {"min_precision": 0.18, "target_recall": 0.80, "max_predicted_fraction": 0.16},
    "Edema": {"min_precision": 0.20, "target_recall": 0.80, "max_predicted_fraction": 0.12},
    "Pneumothorax": {"min_precision": 0.02, "target_recall": 0.50, "max_predicted_fraction": 0.22},
    "Consolidation": {"min_precision": 0.03, "target_recall": 0.80, "max_predicted_fraction": 0.18},
    "Nodule": {"min_precision": 0.12, "target_recall": 0.50, "max_predicted_fraction": 0.10},
    "Fracture": {"min_precision": 0.02, "target_recall": 0.50, "max_predicted_fraction": 0.22},
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_annotations() -> Dict[str, List[Dict]]:
    return json.loads(ANNOTATIONS_PATH.read_text(encoding="utf-8"))


def load_thresholds() -> Dict[str, float]:
    path = ENSEMBLE_DIR / "thresholds.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def load_policy() -> Dict[str, Dict]:
    path = ENSEMBLE_DIR / "kg_label_policy.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def load_score_rows(split: str) -> Dict[str, Dict[str, float]]:
    path = ENSEMBLE_DIR / f"{split}_component_scores.csv"
    if not path.exists():
        return {}
    rows = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scores = {}
            for disease in DISEASES:
                for component in ("rad", "xrv", "ensemble"):
                    key = f"{disease}_{component}"
                    if key in row:
                        scores[key] = float(row[key])
            rows[str(row["sample_id"])] = {
                "image_paths": row.get("image_paths", ""),
                **scores,
            }
    return rows


def image_paths_for(item: Dict) -> List[str]:
    rel_paths = item.get("image_path", [])
    if isinstance(rel_paths, str):
        rel_paths = [rel_paths]
    return [str(IMAGES_ROOT / rel_path) for rel_path in rel_paths if (IMAGES_ROOT / rel_path).exists()]


def binary_metrics(truth: np.ndarray, pred: np.ndarray) -> Dict[str, float | int | None]:
    tp = int(((truth == 1) & (pred == 1)).sum())
    fp = int(((truth == 0) & (pred == 1)).sum())
    fn = int(((truth == 1) & (pred == 0)).sum())
    tn = int(((truth == 0) & (pred == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision is not None and precision + recall > 0 else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predicted": int(pred.sum()),
        "support": int(truth.sum()),
    }


def candidate_thresholds(scores: np.ndarray) -> np.ndarray:
    grid = np.arange(0.025, 0.951, 0.025)
    unique_scores = np.unique(np.round(scores, 4))
    return np.unique(np.concatenate([grid, unique_scores]))


def threshold_candidate(scores: np.ndarray, truth: np.ndarray, threshold: float) -> Dict:
    pred = (scores >= threshold).astype(np.int32)
    metrics = binary_metrics(truth, pred)
    metrics["threshold"] = float(round(threshold, 4))
    metrics["predicted_fraction"] = float(pred.mean()) if len(pred) else 0.0
    return metrics


def select_support_threshold(disease: str, scores: np.ndarray, truth: np.ndarray) -> Dict:
    targets = SUPPORT_POLICY_TARGETS[disease]
    candidates = [
        threshold_candidate(scores, truth, threshold)
        for threshold in candidate_thresholds(scores)
    ]
    candidates = [item for item in candidates if item["predicted"] > 0]
    if not candidates:
        return {
            "support_enabled": False,
            "support_threshold": None,
            "selection": "no_predictions",
            "target": targets,
            "val_positives": int(truth.sum()),
        }

    bounded = [
        item for item in candidates
        if item["predicted_fraction"] <= targets["max_predicted_fraction"]
        and (item["precision"] or 0.0) >= targets["min_precision"]
    ]
    recall_target = [
        item for item in bounded
        if float(item["recall"] or 0.0) >= targets["target_recall"]
    ]

    if recall_target:
        selected = max(
            recall_target,
            key=lambda item: (item["recall"], item["precision"] or 0.0, -item["predicted_fraction"]),
        )
        selection = "target_recall_and_precision_met"
    elif bounded:
        selected = max(
            bounded,
            key=lambda item: (
                5.0 * float(item["precision"] or 0.0) * float(item["recall"] or 0.0)
                / max(1e-8, 4.0 * float(item["precision"] or 0.0) + float(item["recall"] or 0.0)),
                item["recall"],
                item["precision"] or 0.0,
            ),
        )
        selection = "best_f2_with_precision_bound"
    else:
        selected = max(
            candidates,
            key=lambda item: (
                5.0 * float(item["precision"] or 0.0) * float(item["recall"] or 0.0)
                / max(1e-8, 4.0 * float(item["precision"] or 0.0) + float(item["recall"] or 0.0)),
                item["precision"] or 0.0,
                -item["predicted_fraction"],
            ),
        )
        selection = "diagnostic_only_target_not_met"

    support_enabled = selected["tp"] > 0 and selection != "diagnostic_only_target_not_met"
    return {
        "support_enabled": bool(support_enabled),
        "support_threshold": selected["threshold"] if support_enabled else None,
        "selection": selection,
        "target": targets,
        "val_positives": int(truth.sum()),
        "validation_metrics_at_threshold": selected,
        "top_candidates": sorted(
            candidates,
            key=lambda item: (item["f1"], item["recall"], item["precision"] or 0.0),
            reverse=True,
        )[:10],
    }


def build_support_policy(annotations: Dict[str, List[Dict]], score_rows: Dict[str, Dict[str, Dict]]) -> Dict:
    val_by_id = {str(item.get("id")): item for item in annotations.get("val", [])}
    policy = {}
    for disease in WEAK_DISEASES:
        scores, truth = [], []
        for sample_id, row in score_rows.get("val", {}).items():
            item = val_by_id.get(sample_id)
            if not item:
                continue
            status = certainty_labels_from_report(item.get("report", "")).get(disease, "not_mentioned")
            scores.append(float(row.get(f"{disease}_ensemble", 0.0)))
            truth.append(1 if status == "present" else 0)
        policy[disease] = select_support_threshold(
            disease,
            np.asarray(scores, dtype=np.float32),
            np.asarray(truth, dtype=np.int32),
        )
    return policy


def status_for_score(disease: str, score: float, thresholds: Dict, policy: Dict, support_policy: Dict) -> str:
    threshold = float(thresholds.get(disease, 1.0))
    kg_enabled = bool(policy.get(disease, {}).get("kg_enabled", False))
    if score >= threshold and kg_enabled:
        return "present"
    detail = support_policy.get(disease, {})
    support_threshold = detail.get("support_threshold")
    if bool(detail.get("support_enabled", False)) and support_threshold is not None and score >= float(support_threshold):
        return "uncertain"
    if score >= threshold:
        return "uncertain"
    return "absent"


def summarize_split(
    split: str,
    annotations: Dict[str, List[Dict]],
    score_rows: Dict[str, Dict[str, Dict]],
    thresholds: Dict,
    policy: Dict,
    support_policy: Dict,
) -> Dict:
    by_id = {str(item.get("id")): item for item in annotations.get(split, [])}
    summary = {}
    for disease in WEAK_DISEASES:
        truth, strict_pred, support_pred = [], [], []
        for sample_id, row in score_rows.get(split, {}).items():
            item = by_id.get(sample_id)
            if not item:
                continue
            status = certainty_labels_from_report(item.get("report", "")).get(disease, "not_mentioned")
            score = float(row.get(f"{disease}_ensemble", 0.0))
            pred_status = status_for_score(disease, score, thresholds, policy, support_policy)
            truth.append(1 if status == "present" else 0)
            strict_pred.append(1 if pred_status == "present" else 0)
            support_pred.append(1 if pred_status in {"present", "uncertain"} else 0)
        truth_arr = np.asarray(truth, dtype=np.int32)
        summary[disease] = {
            "strict_present": binary_metrics(truth_arr, np.asarray(strict_pred, dtype=np.int32)),
            "present_or_uncertain": binary_metrics(truth_arr, np.asarray(support_pred, dtype=np.int32)),
        }
    return summary


def build_audit_rows(
    annotations: Dict[str, List[Dict]],
    score_rows: Dict[str, Dict[str, Dict]],
    thresholds: Dict,
    policy: Dict,
    support_policy: Dict,
) -> List[Dict]:
    rows = []
    for split, items in annotations.items():
        split_scores = score_rows.get(split, {})
        for item in items:
            sample_id = str(item.get("id"))
            report = item.get("report", "")
            statuses = certainty_labels_from_report(report)
            has_weak_label = any(statuses[disease] in {"present", "uncertain"} for disease in WEAK_DISEASES)
            scored = sample_id in split_scores
            row_scores = split_scores.get(sample_id, {})

            weak_predictions = {}
            weak_errors = {}
            for disease in WEAK_DISEASES:
                score = row_scores.get(f"{disease}_ensemble")
                pred_status = None
                if score is not None:
                    pred_status = status_for_score(disease, float(score), thresholds, policy, support_policy)
                    truth = statuses[disease] == "present"
                    predicted_support = pred_status in {"present", "uncertain"}
                    weak_errors[disease] = (
                        "tp" if truth and predicted_support else
                        "fn" if truth and not predicted_support else
                        "fp" if not truth and predicted_support else
                        "tn"
                    )
                weak_predictions[disease] = {
                    "certainty": statuses[disease],
                    "score": score,
                    "rad_score": row_scores.get(f"{disease}_rad"),
                    "xrv_score": row_scores.get(f"{disease}_xrv"),
                    "current_threshold": thresholds.get(disease),
                    "support_threshold": support_policy.get(disease, {}).get("support_threshold"),
                    "predicted_status": pred_status,
                    "error_type": weak_errors.get(disease),
                    "evidence": collect_keyword_evidence(report.lower(), disease, KEYWORD_MAP[disease]),
                }

            include_scored_error = any(error in {"fn", "fp"} for error in weak_errors.values())
            if has_weak_label or include_scored_error:
                rows.append({
                    "split": split,
                    "sample_id": sample_id,
                    "image_paths": image_paths_for(item),
                    "report": report,
                    "weak_predictions": weak_predictions,
                    "has_scored_classifier_outputs": scored,
                })
    return rows


def write_summary_md(path: Path, payload: Dict) -> None:
    lines = [
        "# Rare Label Audit",
        "",
        f"Created: `{payload['created_at']}`",
        f"Baseline comparison folder: `{payload['baseline_experiment']}`",
        "",
        "## Installed Support Policy",
        "",
        "| Label | Enabled | Threshold | Selection | Val Precision | Val Recall |",
        "| --- | ---: | ---: | --- | ---: | ---: |",
    ]
    for disease, detail in payload["support_policy"].items():
        metrics = detail.get("validation_metrics_at_threshold") or {}
        threshold = detail.get("support_threshold")
        threshold_text = f"{float(threshold):.4f}" if threshold is not None else ""
        precision = metrics.get("precision")
        precision_text = f"{float(precision):.4f}" if precision is not None else ""
        recall_text = f"{float(metrics.get('recall', 0.0)):.4f}"
        lines.append(
            f"| {disease} | {detail.get('support_enabled')} | {threshold_text} | "
            f"{detail.get('selection')} | "
            f"{precision_text} | "
            f"{recall_text} |"
        )

    for split, split_summary in payload["split_metrics"].items():
        lines.extend([
            "",
            f"## {split.capitalize()} Metrics",
            "",
            "| Label | Strict F1 | Support F1 | Support Recall | Support FP | Support FN |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ])
        for disease, metrics in split_summary.items():
            strict = metrics["strict_present"]
            support = metrics["present_or_uncertain"]
            lines.append(
                f"| {disease} | {float(strict['f1'] or 0.0):.4f} | "
                f"{float(support['f1'] or 0.0):.4f} | {float(support['recall'] or 0.0):.4f} | "
                f"{support['fp']} | {support['fn']} |"
            )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Audit and calibrate weak IU X-ray KG labels.")
    parser.add_argument("--install", action="store_true", help="Copy kg_support_policy.json into the ensemble artifact directory.")
    parser.add_argument("--ensemble-dir", default=str(ENSEMBLE_DIR), help="Directory containing ensemble score artifacts.")
    parser.add_argument("--output-dir", default=None, help="Optional output directory. Defaults to out/rare_label_audit/<timestamp>.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    global ENSEMBLE_DIR
    ENSEMBLE_DIR = Path(args.ensemble_dir)
    results_dir = Path(args.output_dir) if args.output_dir else RESULTS_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    ensure_dir(results_dir)

    annotations = load_annotations()
    score_rows = {split: load_score_rows(split) for split in ("val", "test")}
    thresholds = load_thresholds()
    policy = load_policy()
    support_policy = build_support_policy(annotations, score_rows)
    audit_rows = build_audit_rows(annotations, score_rows, thresholds, policy, support_policy)
    split_metrics = {
        split: summarize_split(split, annotations, score_rows, thresholds, policy, support_policy)
        for split in ("val", "test")
    }

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "baseline_experiment": str(BASELINE_EXPERIMENT),
        "weak_diseases": WEAK_DISEASES,
        "support_policy": support_policy,
        "split_metrics": split_metrics,
        "audit_row_count": len(audit_rows),
        "artifacts": {
            "support_policy": str(results_dir / "kg_support_policy.json"),
            "audit_rows": str(results_dir / "rare_label_audit.json"),
            "summary_json": str(results_dir / "summary.json"),
            "summary_md": str(results_dir / "summary.md"),
        },
    }

    (results_dir / "kg_support_policy.json").write_text(json.dumps(support_policy, indent=2), encoding="utf-8")
    (results_dir / "rare_label_audit.json").write_text(json.dumps(audit_rows, indent=2), encoding="utf-8")
    (results_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_summary_md(results_dir / "summary.md", payload)

    if args.install:
        shutil.copyfile(results_dir / "kg_support_policy.json", ENSEMBLE_DIR / "kg_support_policy.json")

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
