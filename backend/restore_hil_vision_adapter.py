"""
Restore a previous Vision Agent HIL adapter backup.

Usage:
  python restore_hil_vision_adapter.py
  python restore_hil_vision_adapter.py --backup model_weights/Vision_Agent/hil_backups/hil_medgemma_active_YYYYMMDD_HHMMSS
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


ACTIVE_ADAPTER_DIR = Path(os.getenv(
    "HIL_VISION_ACTIVE_ADAPTER_DIR",
    str(Path("model_weights") / "Vision_Agent" / "medgemma_ct_grid_finetuned"),
))
BACKUP_ROOT = Path("model_weights") / "Vision_Agent" / "hil_backups"


def latest_backup() -> Path:
    backups = sorted([path for path in BACKUP_ROOT.glob("hil_medgemma_active_*") if path.is_dir()])
    if not backups:
        raise SystemExit(f"No HIL adapter backups found under {BACKUP_ROOT}")
    return backups[-1]


def restore(backup: Path) -> None:
    if not backup.exists() or not backup.is_dir():
        raise SystemExit(f"Backup does not exist: {backup}")
    if ACTIVE_ADAPTER_DIR.exists():
        rollback_copy = ACTIVE_ADAPTER_DIR.with_name(f"{ACTIVE_ADAPTER_DIR.name}_pre_restore")
        if rollback_copy.exists():
            shutil.rmtree(rollback_copy)
        shutil.copytree(ACTIVE_ADAPTER_DIR, rollback_copy)
        shutil.rmtree(ACTIVE_ADAPTER_DIR)
        print(f"Saved current adapter before restore: {rollback_copy}")
    ACTIVE_ADAPTER_DIR.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(backup, ACTIVE_ADAPTER_DIR)
    print(f"Restored {backup} -> {ACTIVE_ADAPTER_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Restore a backed-up HIL Vision Agent adapter.")
    parser.add_argument("--backup", type=Path, default=None, help="Backup directory. Defaults to the latest backup.")
    args = parser.parse_args()
    restore(args.backup or latest_backup())


if __name__ == "__main__":
    main()
