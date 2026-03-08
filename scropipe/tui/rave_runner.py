"""Build RAVE CLI commands for the TUI training worker."""

from __future__ import annotations

from pathlib import Path
from typing import Optional


def build_preprocess_cmd(
    rave_cmd: str,
    input_dir: Path,
    output_dir: Path,
    num_signal: int = 131072,
) -> list[str]:
    """Build the rave preprocess command."""
    return [
        rave_cmd, "preprocess",
        "--input_path", str(input_dir),
        "--output_path", str(output_dir),
        "--num_signal", str(num_signal),
    ]


def build_train_cmd(
    rave_cmd: str,
    config: str,
    data_dir: Path,
    name: str,
    val_every: int = 500,
    max_steps: Optional[int] = None,
    gpu: Optional[int] = None,
    workers: int = 0,
    n_signal: int = 131072,
    ckpt: Optional[Path] = None,
) -> list[str]:
    """Build the rave train command."""
    cmd = [
        rave_cmd, "train",
        "--config", config,
        "--db_path", str(data_dir),
        "--name", name,
        "--n_signal", str(n_signal),
        "--workers", str(workers),
        "--val_every", str(val_every),
    ]
    if gpu is not None:
        cmd.extend(["--gpu", str(gpu)])
    if max_steps is not None:
        cmd.extend(["--max_steps", str(max_steps)])
    if ckpt is not None:
        cmd.extend(["--ckpt", str(ckpt)])
    return cmd
