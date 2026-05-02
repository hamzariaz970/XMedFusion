from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List


BACKEND_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = Path("out/rare_label_retraining")
BASELINE_EXPERIMENT = Path("out/hard_case_experiments/20260502_145619")
DEFAULT_PYTHON = Path(r"D:\Anaconda3\conda_envs\fyp_env\python.exe")
WEAK_LABELS = [
    "Pleural Effusion",
    "Edema",
    "Pneumothorax",
    "Consolidation",
    "Nodule",
    "Fracture",
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def run_logged(command: List[str], log_path: Path, env: Dict[str, str]) -> None:
    ensure_dir(log_path.parent)
    print(f"\nRunning: {' '.join(command)}")
    print(f"Log: {log_path}")
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=BACKEND_DIR,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def write_summary(results_dir: Path, payload: Dict) -> None:
    (results_dir / "run_manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# Rare-Label Retraining",
        "",
        f"Created: `{payload['created_at']}`",
        f"Baseline comparison: `{payload['baseline_experiment']}`",
        "",
        "## Artifact Directories",
        "",
        f"- RAD-DINO: `{payload['rad_dir']}`",
        f"- Ensemble: `{payload['ensemble_dir']}`",
        f"- Audit: `{payload['audit_dir']}`",
        "",
        "## Training Focus",
        "",
        "Weak labels: " + ", ".join(payload["weak_labels"]),
        "",
        "## Logs",
        "",
    ]
    for name, path in payload["logs"].items():
        lines.append(f"- {name}: `{path}`")
    (results_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Run rare-label RAD-DINO retraining, ensemble calibration, and audit.")
    parser.add_argument("--name", default=None, help="Experiment folder name. Defaults to timestamp.")
    parser.add_argument("--python", default=str(DEFAULT_PYTHON), help="Python executable from the training environment.")
    parser.add_argument("--phase1-epochs", type=int, default=18)
    parser.add_argument("--phase2-epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--head-lr", type=float, default=7e-4)
    parser.add_argument("--backbone-lr", type=float, default=8e-6)
    parser.add_argument("--sampler-boost", type=float, default=3.0)
    parser.add_argument("--sampler-max-weight", type=float, default=16.0)
    parser.add_argument("--max-pos-weight", type=float, default=40.0)
    parser.add_argument("--skip-rad", action="store_true", help="Skip RAD-DINO retraining.")
    parser.add_argument("--skip-ensemble", action="store_true", help="Skip ensemble recalibration.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stamp = args.name or datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = RESULTS_ROOT / stamp
    rad_dir = results_dir / "rad_dino_multiview"
    ensemble_dir = results_dir / "rad_dino_xrv_ensemble"
    audit_dir = results_dir / "rare_label_audit"
    logs_dir = results_dir / "logs"
    ensure_dir(results_dir)

    python_exe = str(Path(args.python))
    env = os.environ.copy()
    env.update({
        "KG_RAD_SAVE_DIR": str(rad_dir),
        "KG_ENSEMBLE_SAVE_DIR": str(ensemble_dir),
        "KG_RAD_PHASE1_EPOCHS": str(args.phase1_epochs),
        "KG_RAD_PHASE2_EPOCHS": str(args.phase2_epochs),
        "KG_RAD_BATCH_SIZE": str(args.batch_size),
        "KG_ENSEMBLE_BATCH_SIZE": str(args.batch_size),
        "KG_RAD_HEAD_LR": str(args.head_lr),
        "KG_RAD_BACKBONE_LR": str(args.backbone_lr),
        "KG_RAD_SELECTION_METRIC": "weak_macro_ap",
        "KG_RAD_USE_RARE_LABEL_SAMPLER": "1",
        "KG_RAD_SAMPLER_BOOST": str(args.sampler_boost),
        "KG_RAD_SAMPLER_MAX_WEIGHT": str(args.sampler_max_weight),
        "KG_RAD_MAX_POS_WEIGHT": str(args.max_pos_weight),
        "KG_RAD_RARE_LABELS": ",".join(WEAK_LABELS),
    })

    logs = {}
    if not args.skip_rad:
        log_path = logs_dir / "train_rad_dino.log"
        logs["train_rad_dino"] = str(log_path)
        run_logged([python_exe, "train_rad_dino_kg_agent.py"], log_path, env)

    if not args.skip_ensemble:
        log_path = logs_dir / "train_ensemble.log"
        logs["train_ensemble"] = str(log_path)
        run_logged([python_exe, "train_xrv_rad_dino_ensemble_kg_agent.py"], log_path, env)

    audit_log = logs_dir / "rare_label_audit.log"
    logs["rare_label_audit"] = str(audit_log)
    run_logged(
        [
            python_exe,
            "rare_label_audit.py",
            "--ensemble-dir",
            str(ensemble_dir),
            "--output-dir",
            str(audit_dir),
            "--install",
        ],
        audit_log,
        env,
    )

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "baseline_experiment": str(BASELINE_EXPERIMENT),
        "results_dir": str(results_dir),
        "rad_dir": str(rad_dir),
        "ensemble_dir": str(ensemble_dir),
        "audit_dir": str(audit_dir),
        "weak_labels": WEAK_LABELS,
        "training_env": {
            key: env[key]
            for key in sorted(env)
            if key.startswith("KG_RAD_") or key.startswith("KG_ENSEMBLE_")
        },
        "logs": logs,
    }
    write_summary(results_dir, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    sys.exit(main())
