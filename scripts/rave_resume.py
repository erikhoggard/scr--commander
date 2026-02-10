#!/usr/bin/env python3
"""Resume RAVE training from a checkpoint."""

import argparse
import subprocess
import sys
from pathlib import Path


def find_latest_checkpoint(run_dir: Path) -> Path | None:
    """Find the most recent checkpoint file."""
    checkpoints = list(run_dir.rglob("*.ckpt"))
    if not checkpoints:
        return None
    # Prefer epoch checkpoints over 'best.ckpt' for resuming
    epoch_ckpts = [c for c in checkpoints if "epoch" in c.name]
    if epoch_ckpts:
        return max(epoch_ckpts, key=lambda p: p.stat().st_mtime)
    return max(checkpoints, key=lambda p: p.stat().st_mtime)


def find_config(run_dir: Path) -> Path | None:
    """Find the config.gin file."""
    configs = list(run_dir.rglob("config.gin"))
    return configs[0] if configs else None


def main():
    parser = argparse.ArgumentParser(description="Resume RAVE training")
    parser.add_argument(
        "run_dir",
        nargs="?",
        default="scropipe-output/03-rave-model",
        help="RAVE training run directory (default: scropipe-output/03-rave-model)",
    )
    parser.add_argument(
        "--db", "--db_path",
        default="scropipe-output/02-rave-data",
        help="Preprocessed dataset path",
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU index (-1 for CPU)")
    parser.add_argument("--workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--val-every", type=int, default=500, help="Checkpoint every N steps")
    parser.add_argument("--dry-run", action="store_true", help="Print command without running")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"Error: Run directory not found: {run_dir}")
        return 1

    # Find checkpoint
    ckpt = find_latest_checkpoint(run_dir)
    if not ckpt:
        print(f"Error: No checkpoint found in {run_dir}")
        return 1

    # Find config
    config = find_config(run_dir)
    if not config:
        print(f"Error: No config.gin found in {run_dir}")
        return 1

    # Extract model name from checkpoint path
    # e.g., .../model_e18d54798e/version_0/checkpoints/epoch.ckpt -> model
    name = "model"
    for part in ckpt.parts:
        if part.startswith("model_"):
            name = "model"
            break

    print(f"Config:     {config}")
    print(f"Checkpoint: {ckpt}")
    print(f"Database:   {args.db}")
    print()

    cmd = [
        "rave", "train",
        "--config", str(config),
        "--db_path", str(args.db),
        "--out_path", str(run_dir),
        "--name", name,
        "--ckpt", str(ckpt),
        "--channels", "1",
        "--workers", str(args.workers),
        "--val_every", str(args.val_every),
    ]

    if args.gpu >= 0:
        cmd.extend(["--gpu", str(args.gpu)])

    print("Command:")
    print(" ".join(cmd))
    print()

    if args.dry_run:
        return 0

    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    sys.exit(main())
