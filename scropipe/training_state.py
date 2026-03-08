"""Training run state tracking via JSON sidecar files."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


@dataclass
class TrainingRunInfo:
    """Metadata about a training run, persisted as training_run.json."""

    model_name: str
    pool_name: str
    architecture: str
    output_dir: str
    status: str  # "training", "paused", "completed"
    started: str = ""

    def __post_init__(self):
        if not self.started:
            self.started = datetime.now(timezone.utc).isoformat()


def save_training_run(run: TrainingRunInfo, run_dir: Path) -> None:
    """Write training run info to run_dir/training_run.json."""
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "training_run.json"
    path.write_text(json.dumps(asdict(run), indent=2))


def load_training_run(run_dir: Path) -> Optional[TrainingRunInfo]:
    """Load training run info from run_dir/training_run.json."""
    path = run_dir / "training_run.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return TrainingRunInfo(**data)


def reconcile_stale_runs(models_dir: Path) -> None:
    """Mark any runs with status 'training' as 'paused'.

    Called on TUI startup to handle cases where the TUI exited
    without graceful shutdown.
    """
    if not models_dir.exists():
        return
    for d in models_dir.iterdir():
        if not d.is_dir():
            continue
        run = load_training_run(d)
        if run is not None and run.status == "training":
            run.status = "paused"
            save_training_run(run, d)


def find_checkpoint_dir(output_dir: Path) -> Optional[Path]:
    """Find the RAVE checkpoint directory inside a training output dir.

    Searches for runs/*/version_*/checkpoints/ containing .ckpt files.
    Returns the checkpoint directory path, or None if not found.
    """
    if not output_dir.exists():
        return None
    ckpt_files = list(output_dir.rglob("*.ckpt"))
    if not ckpt_files:
        return None
    # Return the parent directory of the most recent checkpoint
    newest = max(ckpt_files, key=lambda p: p.stat().st_mtime)
    return newest.parent


def list_paused_runs(models_dir: Path) -> list[TrainingRunInfo]:
    """Find all training runs with status 'paused' or 'training' (stale)."""
    if not models_dir.exists():
        return []
    runs = []
    for d in sorted(models_dir.iterdir()):
        if not d.is_dir():
            continue
        run = load_training_run(d)
        if run is not None and run.status in ("paused", "training"):
            runs.append(run)
    return runs
