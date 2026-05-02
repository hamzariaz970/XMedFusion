from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from kg_agent_xrv_rad_dino_ensemble import DISEASES, load_ensemble_agent, predict_pathologies
from report_labels import case_profile, definite_positive_vector


ANNOTATIONS_PATH = Path("data/iu_xray/annotation.json")
IMAGES_ROOT = Path("data/iu_xray/images")
RESULTS_ROOT = Path("out/hard_case_experiments")
PYTHON_EXE = Path(r"D:\Anaconda3\conda_envs\fyp_env\python.exe")
RARE_DISEASES = {"Edema", "Pneumothorax", "Consolidation", "Nodule", "Fracture"}
KG_VARIANTS = ["baseline", "clahe", "square_clahe", "rad_384", "xrv_resize_only"]
KG_LIMIT = 40
SYNTHESIS_LIMIT = 20


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sanitize_metric(value):
    if value is None:
        return None
    if isinstance(value, (np.floating, np.integer)):
        return float(value)
    return value


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float | int | None]:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision is not None and (precision + recall) > 0
        else 0.0
    )
    accuracy = (tp + tn) / max(1, len(y_true))
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": sanitize_metric(precision),
        "recall": sanitize_metric(recall),
        "f1": sanitize_metric(f1),
        "accuracy": sanitize_metric(accuracy),
    }


def average_metric(values: List[float | None]) -> float | None:
    filtered = [value for value in values if value is not None]
    if not filtered:
        return None
    return float(np.mean(filtered))


def compute_kg_metrics(valid_logs: List[Dict]) -> Dict:
    per_label = {}
    gt_matrix = []
    pred_present_matrix = []
    pred_support_matrix = []
    for disease in DISEASES:
        gt = np.asarray([item["ground_truth_labels"][disease] for item in valid_logs], dtype=np.int32)
        statuses = [item["kg_label_statuses"].get(disease, "not_mentioned") for item in valid_logs]
        pred_present = np.asarray([1 if status == "present" else 0 for status in statuses], dtype=np.int32)
        pred_support = np.asarray([1 if status in {"present", "uncertain"} else 0 for status in statuses], dtype=np.int32)
        per_label[disease] = {
            "support": int(gt.sum()),
            "strict_present_metrics": compute_binary_metrics(gt, pred_present),
            "present_or_uncertain_metrics": compute_binary_metrics(gt, pred_support),
        }
        gt_matrix.append(gt)
        pred_present_matrix.append(pred_present)
        pred_support_matrix.append(pred_support)

    gt_matrix = np.column_stack(gt_matrix)
    pred_present_matrix = np.column_stack(pred_present_matrix)
    pred_support_matrix = np.column_stack(pred_support_matrix)

    return {
        "valid_samples": len(valid_logs),
        "strict_present_macro": {
            "precision": average_metric([per_label[d]["strict_present_metrics"]["precision"] for d in DISEASES]),
            "recall": average_metric([per_label[d]["strict_present_metrics"]["recall"] for d in DISEASES]),
            "f1": average_metric([per_label[d]["strict_present_metrics"]["f1"] for d in DISEASES]),
        },
        "strict_present_micro": compute_binary_metrics(gt_matrix.reshape(-1), pred_present_matrix.reshape(-1)),
        "present_or_uncertain_macro": {
            "precision": average_metric([per_label[d]["present_or_uncertain_metrics"]["precision"] for d in DISEASES]),
            "recall": average_metric([per_label[d]["present_or_uncertain_metrics"]["recall"] for d in DISEASES]),
            "f1": average_metric([per_label[d]["present_or_uncertain_metrics"]["f1"] for d in DISEASES]),
        },
        "present_or_uncertain_micro": compute_binary_metrics(gt_matrix.reshape(-1), pred_support_matrix.reshape(-1)),
        "per_label": per_label,
    }


def load_test_samples() -> List[Dict]:
    data = json.loads(ANNOTATIONS_PATH.read_text(encoding="utf-8"))
    return data.get("test", [])


def extract_image_paths(example: Dict) -> List[str]:
    rel_paths = example.get("image_path", [])
    if isinstance(rel_paths, str):
        rel_paths = [rel_paths]
    return [str(IMAGES_ROOT / rel_path) for rel_path in rel_paths if (IMAGES_ROOT / rel_path).exists()]


def select_hard_cases(limit: int) -> List[Dict]:
    enriched = []
    for example in load_test_samples():
        profile = case_profile(example.get("report", ""))
        definite = definite_positive_vector(example.get("report", "")).astype(np.int32)
        if int(definite.sum()) <= 0:
            continue
        enriched.append({
            "example": example,
            "labels": definite,
            "positive_count": int(definite.sum()),
            "rare_positive_count": int(profile["rare_positive_count"]),
        })

    selected = []
    selected_ids = set()
    uncovered = {
        DISEASES[idx]
        for idx in range(len(DISEASES))
        if any(item["labels"][idx] == 1 for item in enriched)
    }

    while len(selected) < limit and uncovered:
        best = None
        best_gain = -1
        for item in enriched:
            sample_id = str(item["example"].get("id"))
            if sample_id in selected_ids:
                continue
            gain = sum(
                1
                for idx, disease in enumerate(DISEASES)
                if disease in uncovered and item["labels"][idx] == 1
            )
            tie_break = item["positive_count"] + item["rare_positive_count"]
            if best is None or gain > best_gain or (gain == best_gain and tie_break > (best["positive_count"] + best["rare_positive_count"])):
                best = item
                best_gain = gain
        if best is None:
            break
        selected.append(best["example"])
        selected_ids.add(str(best["example"].get("id")))
        for idx, disease in enumerate(DISEASES):
            if best["labels"][idx] == 1 and disease in uncovered:
                uncovered.remove(disease)

    remaining = [item for item in enriched if str(item["example"].get("id")) not in selected_ids]
    remaining.sort(key=lambda item: (item["rare_positive_count"], item["positive_count"]), reverse=True)
    for item in remaining:
        if len(selected) >= limit:
            break
        selected.append(item["example"])
    return selected[:limit]


def run_kg_ablation(variant: str, subset: List[Dict], exp_dir: Path) -> Dict:
    agent = load_ensemble_agent(preprocess_variant=variant)
    rows = []
    for example in subset:
        image_paths = extract_image_paths(example)
        if not image_paths:
            continue
        prediction = predict_pathologies(
            image_paths,
            agent=agent,
            policy_mode="validation",
            preprocess_variant=variant,
        )
        gt = definite_positive_vector(example.get("report", "")).astype(np.int32)
        kg_statuses = {
            disease: payload.get("status", "not_mentioned")
            for disease, payload in prediction.get("findings", {}).items()
        }
        rows.append({
            "id": str(example.get("id")),
            "image_paths": image_paths,
            "ground_truth_labels": {disease: int(gt[idx]) for idx, disease in enumerate(DISEASES)},
            "kg_label_statuses": kg_statuses,
            "prediction": prediction,
        })

    metrics = compute_kg_metrics(rows)
    strict = metrics["strict_present_micro"]
    support = metrics["present_or_uncertain_micro"]
    score = (
        0.45 * float(support["f1"] or 0.0)
        + 0.25 * float(strict["f1"] or 0.0)
        + 0.20 * float(support["recall"] or 0.0)
        + 0.10 * float(strict["precision"] or 0.0)
    )
    result = {
        "variant": variant,
        "selection_score": score,
        "metrics": metrics,
        "samples": rows,
    }
    (exp_dir / f"kg_ablation_{variant}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def _read_text_scores(csv_path: Path) -> Dict[str, float]:
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        row = next(reader)
    result = {}
    for key, value in row.items():
        try:
            result[key] = float(value)
        except Exception:
            result[key] = value
    return result


def run_synthesis_eval(exp_dir: Path, name: str, retrieval_label_weight: float, preprocess_variant: str) -> Dict:
    output_file = exp_dir / f"{name}_generations.json"
    scores_file = exp_dir / f"{name}_scores.csv"
    kg_scores_file = exp_dir / f"{name}_kg_scores.json"
    env = os.environ.copy()
    env["RETRIEVAL_LABEL_WEIGHT"] = str(retrieval_label_weight)
    env["KG_PREPROCESS_VARIANT"] = preprocess_variant
    cmd = [
        str(PYTHON_EXE),
        "evaluate.py",
        "--limit", str(SYNTHESIS_LIMIT),
        "--sample-mode", "hard_stratified",
        "--label-mode", "definite",
        "--output-file", str(output_file),
        "--scores-file", str(scores_file),
        "--kg-scores-file", str(kg_scores_file),
        "--timeout-seconds", "240",
    ]
    subprocess.run(cmd, cwd=Path(__file__).resolve().parent, env=env, check=True)
    return {
        "name": name,
        "retrieval_label_weight": retrieval_label_weight,
        "preprocess_variant": preprocess_variant,
        "files": {
            "generations": str(output_file),
            "text_scores": str(scores_file),
            "kg_scores": str(kg_scores_file),
        },
        "text_scores": _read_text_scores(scores_file),
        "kg_scores": json.loads(kg_scores_file.read_text(encoding="utf-8")),
    }


def summarize(results_dir: Path, kg_results: List[Dict], synthesis_results: List[Dict]) -> Dict:
    best_kg = max(kg_results, key=lambda item: item["selection_score"])
    best_synthesis = max(
        synthesis_results,
        key=lambda item: (
            float(item["text_scores"].get("accuracy", 0.0)),
            float(item["text_scores"].get("Bleu_1", 0.0)),
            float(item["kg_scores"]["present_or_uncertain_micro"].get("f1", 0.0)),
        ),
    )
    summary = {
        "experiment_root": str(results_dir),
        "kg_ablation": {
            "best_variant": best_kg["variant"],
            "results": [
                {
                    "variant": item["variant"],
                    "selection_score": item["selection_score"],
                    "strict_present_micro": item["metrics"]["strict_present_micro"],
                    "present_or_uncertain_micro": item["metrics"]["present_or_uncertain_micro"],
                }
                for item in kg_results
            ],
        },
        "synthesis": {
            "best_run": best_synthesis["name"],
            "results": synthesis_results,
        },
        "recommendation": {
            "kg_preprocess_variant": best_kg["variant"],
            "retrieval_label_weight": best_synthesis["retrieval_label_weight"],
        },
    }
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Hard-Case Experiments",
        "",
        f"Results root: `{results_dir}`",
        "",
        "## KG Preprocessing Ablation",
        "",
        "| Variant | Selection Score | Strict KG F1 | Support KG F1 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for item in kg_results:
        strict = item["metrics"]["strict_present_micro"]
        support = item["metrics"]["present_or_uncertain_micro"]
        lines.append(
            f"| {item['variant']} | {item['selection_score']:.4f} | {float(strict['f1'] or 0.0):.4f} | {float(support['f1'] or 0.0):.4f} |"
        )

    lines.extend([
        "",
        f"Best KG variant: `{best_kg['variant']}`",
        "",
        "## Synthesis on Hard Cases",
        "",
        "| Run | Retrieval Label Weight | BLEU-1 | ROUGE-L | Judge Accuracy | Support KG F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for item in synthesis_results:
        lines.append(
            f"| {item['name']} | {item['retrieval_label_weight']:.2f} | "
            f"{float(item['text_scores'].get('Bleu_1', 0.0)):.4f} | "
            f"{float(item['text_scores'].get('ROUGE_L', 0.0)):.4f} | "
            f"{float(item['text_scores'].get('accuracy', 0.0)):.4f} | "
            f"{float(item['kg_scores']['present_or_uncertain_micro'].get('f1', 0.0)):.4f} |"
        )

    lines.extend([
        "",
        f"Best synthesis run: `{best_synthesis['name']}`",
        f"Recommended runtime settings: `KG_PREPROCESS_VARIANT={best_kg['variant']}` and `RETRIEVAL_LABEL_WEIGHT={best_synthesis['retrieval_label_weight']}`",
    ])
    (results_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def main() -> None:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = RESULTS_ROOT / stamp
    ensure_dir(results_dir)

    kg_subset = select_hard_cases(KG_LIMIT)
    synthesis_subset = select_hard_cases(SYNTHESIS_LIMIT)
    (results_dir / "subset_manifest.json").write_text(
        json.dumps(
            {
                "kg_limit": KG_LIMIT,
                "synthesis_limit": SYNTHESIS_LIMIT,
                "kg_ids": [str(item.get("id")) for item in kg_subset],
                "synthesis_ids": [str(item.get("id")) for item in synthesis_subset],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    kg_results = [run_kg_ablation(variant, kg_subset, results_dir) for variant in KG_VARIANTS]
    best_kg = max(kg_results, key=lambda item: item["selection_score"])["variant"]

    synthesis_results = [
        run_synthesis_eval(results_dir, "synthesis_retrieval_baseline", 0.0, best_kg),
        run_synthesis_eval(results_dir, "synthesis_retrieval_abnormal", 0.35, best_kg),
    ]

    summary = summarize(results_dir, kg_results, synthesis_results)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
