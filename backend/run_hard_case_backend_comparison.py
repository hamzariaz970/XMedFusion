from __future__ import annotations

import csv
import importlib
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

from report_labels import DISEASES, case_profile, definite_positive_vector


ANNOTATIONS_PATH = Path("data/iu_xray/annotation.json")
IMAGES_ROOT = Path("data/iu_xray/images")
RESULTS_ROOT = Path("out/hard_case_backend_comparison")
PYTHON_EXE = Path(r"D:\Anaconda3\conda_envs\fyp_env\python.exe")
KG_LIMIT = 40
SYNTHESIS_LIMIT = 20
PREPROCESS_VARIANT = "baseline"
RETRIEVAL_WEIGHTS = [0.35, 0.50]

BACKENDS = [
    {
        "name": "baseline",
        "kg_rad_save_dir": "model_weights/KG_Agent/rad_dino_multiview",
        "kg_ensemble_save_dir": "model_weights/KG_Agent/rad_dino_xrv_ensemble",
        "kg_support_policy_path": "model_weights/KG_Agent/rad_dino_xrv_ensemble/kg_support_policy.json",
    },
    {
        "name": "retrained",
        "kg_rad_save_dir": "out/rare_label_retraining/20260502_rare_retrain/rad_dino_multiview",
        "kg_ensemble_save_dir": "out/rare_label_retraining/20260502_rare_retrain/rad_dino_xrv_ensemble",
        "kg_support_policy_path": "out/rare_label_retraining/20260502_rare_retrain/rare_label_audit/kg_support_policy.json",
    },
]


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


def backend_env(backend: Dict, retrieval_label_weight: float | None = None) -> Dict[str, str]:
    env = os.environ.copy()
    env["KG_RAD_SAVE_DIR"] = backend["kg_rad_save_dir"]
    env["KG_ENSEMBLE_SAVE_DIR"] = backend["kg_ensemble_save_dir"]
    env["KG_SUPPORT_POLICY_PATH"] = backend["kg_support_policy_path"]
    env["KG_PREPROCESS_VARIANT"] = PREPROCESS_VARIANT
    if retrieval_label_weight is not None:
        env["RETRIEVAL_LABEL_WEIGHT"] = str(retrieval_label_weight)
    return env


def run_kg_eval(backend: Dict, subset: List[Dict], results_dir: Path) -> Dict:
    env = backend_env(backend)
    previous_env = {
        "KG_RAD_SAVE_DIR": os.environ.get("KG_RAD_SAVE_DIR"),
        "KG_ENSEMBLE_SAVE_DIR": os.environ.get("KG_ENSEMBLE_SAVE_DIR"),
        "KG_SUPPORT_POLICY_PATH": os.environ.get("KG_SUPPORT_POLICY_PATH"),
        "KG_PREPROCESS_VARIANT": os.environ.get("KG_PREPROCESS_VARIANT"),
    }
    os.environ.update(env)
    kg_module = importlib.import_module("kg_agent_xrv_rad_dino_ensemble")
    kg_module = importlib.reload(kg_module)
    agent = kg_module.load_ensemble_agent(preprocess_variant=PREPROCESS_VARIANT)
    rows = []
    for example in subset:
        image_paths = extract_image_paths(example)
        if not image_paths:
            continue
        prediction = kg_module.predict_pathologies(
            image_paths,
            agent=agent,
            policy_mode="validation",
            preprocess_variant=PREPROCESS_VARIANT,
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
    result = {
        "backend": backend["name"],
        "kg_env": {
            "KG_RAD_SAVE_DIR": env["KG_RAD_SAVE_DIR"],
            "KG_ENSEMBLE_SAVE_DIR": env["KG_ENSEMBLE_SAVE_DIR"],
            "KG_SUPPORT_POLICY_PATH": env["KG_SUPPORT_POLICY_PATH"],
            "KG_PREPROCESS_VARIANT": env["KG_PREPROCESS_VARIANT"],
        },
        "metrics": metrics,
        "samples": rows,
    }
    (results_dir / f"{backend['name']}_kg_eval.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    for key, value in previous_env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
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


def run_synthesis_eval(backend: Dict, retrieval_label_weight: float, results_dir: Path) -> Dict:
    name = f"{backend['name']}_retrieval_{str(retrieval_label_weight).replace('.', '_')}"
    output_file = results_dir / f"{name}_generations.json"
    scores_file = results_dir / f"{name}_scores.csv"
    kg_scores_file = results_dir / f"{name}_kg_scores.json"
    env = backend_env(backend, retrieval_label_weight=retrieval_label_weight)
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
    subprocess.run(cmd, cwd=BACKEND_DIR, env=env, check=True)
    return {
        "backend": backend["name"],
        "name": name,
        "retrieval_label_weight": retrieval_label_weight,
        "kg_env": {
            "KG_RAD_SAVE_DIR": env["KG_RAD_SAVE_DIR"],
            "KG_ENSEMBLE_SAVE_DIR": env["KG_ENSEMBLE_SAVE_DIR"],
            "KG_SUPPORT_POLICY_PATH": env["KG_SUPPORT_POLICY_PATH"],
            "KG_PREPROCESS_VARIANT": env["KG_PREPROCESS_VARIANT"],
            "RETRIEVAL_LABEL_WEIGHT": env["RETRIEVAL_LABEL_WEIGHT"],
        },
        "files": {
            "generations": str(output_file),
            "text_scores": str(scores_file),
            "kg_scores": str(kg_scores_file),
        },
        "text_scores": _read_text_scores(scores_file),
        "kg_scores": json.loads(kg_scores_file.read_text(encoding="utf-8")),
    }


def coverage_selector(run: Dict) -> tuple:
    text = run["text_scores"]
    kg = run["kg_scores"]
    support = kg["present_or_uncertain_micro"]
    strict = kg["strict_present_micro"]
    return (
        float(text.get("coverage", 0.0)),
        float(text.get("accuracy", 0.0)),
        float(support.get("recall", 0.0) or 0.0),
        float(support.get("f1", 0.0) or 0.0),
        float(strict.get("recall", 0.0) or 0.0),
        -float(text.get("style", 0.0)),
    )


def compare_runs(results_dir: Path, kg_results: List[Dict], synthesis_results: List[Dict]) -> Dict:
    best_run = max(synthesis_results, key=coverage_selector)
    by_backend = {result["backend"]: result for result in kg_results}

    summary = {
        "experiment_root": str(results_dir),
        "preprocess_variant": PREPROCESS_VARIANT,
        "kg_results": kg_results,
        "synthesis_results": synthesis_results,
        "best_coverage_run": best_run,
        "recommendation": {
            "backend": best_run["backend"],
            "retrieval_label_weight": best_run["retrieval_label_weight"],
            "why": "selected by medical coverage/accuracy and KG support recall, not IU style",
        },
        "backend_deltas": {},
    }

    if "baseline" in by_backend and "retrained" in by_backend:
        base = by_backend["baseline"]["metrics"]["present_or_uncertain_micro"]
        new = by_backend["retrained"]["metrics"]["present_or_uncertain_micro"]
        summary["backend_deltas"]["kg_support_micro_f1"] = float((new["f1"] or 0.0) - (base["f1"] or 0.0))
        summary["backend_deltas"]["kg_support_micro_recall"] = float((new["recall"] or 0.0) - (base["recall"] or 0.0))

    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Hard-Case Backend Comparison",
        "",
        f"Results root: `{results_dir}`",
        f"Preprocess variant: `{PREPROCESS_VARIANT}`",
        "",
        "## KG Hard-Case Comparison",
        "",
        "| Backend | Strict KG F1 | Strict KG Recall | Support KG F1 | Support KG Recall |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for result in kg_results:
        strict = result["metrics"]["strict_present_micro"]
        support = result["metrics"]["present_or_uncertain_micro"]
        lines.append(
            f"| {result['backend']} | {float(strict['f1'] or 0.0):.4f} | {float(strict['recall'] or 0.0):.4f} | "
            f"{float(support['f1'] or 0.0):.4f} | {float(support['recall'] or 0.0):.4f} |"
        )

    lines.extend([
        "",
        "## Synthesis Comparison",
        "",
        "| Run | Coverage | Accuracy | BLEU-1 | Support KG Recall | Support KG F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for run in synthesis_results:
        support = run["kg_scores"]["present_or_uncertain_micro"]
        lines.append(
            f"| {run['name']} | {float(run['text_scores'].get('coverage', 0.0)):.4f} | "
            f"{float(run['text_scores'].get('accuracy', 0.0)):.4f} | "
            f"{float(run['text_scores'].get('Bleu_1', 0.0)):.4f} | "
            f"{float(support.get('recall', 0.0) or 0.0):.4f} | "
            f"{float(support.get('f1', 0.0) or 0.0):.4f} |"
        )

    lines.extend([
        "",
        f"Best coverage-oriented run: `{best_run['name']}`",
        f"Recommended runtime backend: `{best_run['backend']}` with `RETRIEVAL_LABEL_WEIGHT={best_run['retrieval_label_weight']}`",
        "Selection rule: maximize medical coverage/accuracy and KG support recall before report style.",
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
                "backends": BACKENDS,
                "retrieval_weights": RETRIEVAL_WEIGHTS,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    kg_results = [run_kg_eval(backend, kg_subset, results_dir) for backend in BACKENDS]
    synthesis_results = []
    for backend in BACKENDS:
        for weight in RETRIEVAL_WEIGHTS:
            synthesis_results.append(run_synthesis_eval(backend, weight, results_dir))

    summary = compare_runs(results_dir, kg_results, synthesis_results)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
