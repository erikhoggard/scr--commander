"""RAVE integration for high-quality audio synthesis.

Wraps the acids-rave library for training and generating with RAVE models.
RAVE (Realtime Audio Variational autoEncoder) produces much higher quality
output for melodic/harmonic content compared to the basic VAE + Griffin-Lim.
"""

import subprocess
import shutil
from pathlib import Path
from typing import Optional, List
from rich.console import Console

console = Console()


def check_rave_available() -> bool:
    """Check if RAVE CLI is available."""
    return shutil.which("rave") is not None


def rave_preprocess(
    input_dir: Path,
    output_dir: Path,
    channels: int = 1,
    num_signal: int = 16384,  # ~0.37s at 44100Hz, fits short audio chunks
    lazy: bool = False,
) -> bool:
    """Preprocess audio files for RAVE training.

    Args:
        input_dir: Directory containing audio files.
        output_dir: Directory to save preprocessed dataset.
        channels: Number of audio channels (1 for mono).
        num_signal: Window size in samples (default 65536 ~1.4s at 48kHz).
        lazy: Process raw files without conversion.

    Returns:
        True if preprocessing succeeded.
    """
    if not check_rave_available():
        console.print("[red]RAVE CLI not found. Install with: pip install acids-rave[/red]")
        return False

    output_dir.mkdir(parents=True, exist_ok=True)

    # Note: --channels removed as it's not supported by all RAVE versions
    cmd = [
        "rave", "preprocess",
        "--input_path", str(input_dir),
        "--output_path", str(output_dir),
        "--num_signal", str(num_signal),
    ]

    if lazy:
        cmd.append("--lazy")

    console.print(f"[dim]Running: {' '.join(cmd)}[/dim]")

    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode == 0
    except Exception as e:
        console.print(f"[red]RAVE preprocess failed: {e}[/red]")
        return False


def rave_train(
    db_path: Path,
    output_dir: Path,
    name: str = "model",
    config: str = "v2",
    channels: int = 1,
    epochs: Optional[int] = None,
    val_every: int = 10000,
    n_signal: int = 16384,  # ~0.37s at 44100Hz, fits short audio chunks
) -> bool:
    """Train a RAVE model.

    Args:
        db_path: Path to preprocessed dataset.
        output_dir: Directory to save trained model.
        name: Model name.
        config: RAVE config (v1, v2, v2_small, discrete, onnx, raspberry).
        channels: Number of audio channels.
        epochs: Max epochs (None for default ~3M steps).
        val_every: Validation frequency in steps.
        n_signal: Window size matching preprocessing.

    Returns:
        True if training succeeded.
    """
    if not check_rave_available():
        console.print("[red]RAVE CLI not found. Install with: pip install acids-rave[/red]")
        return False

    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "rave", "train",
        "--config", config,
        "--db_path", str(db_path),
        "--out_path", str(output_dir),
        "--name", name,
        "--val_every", str(val_every),
        "--n_signal", str(n_signal),
    ]

    if epochs is not None:
        cmd.extend(["--max_epochs", str(epochs)])

    console.print(f"[dim]Running: {' '.join(cmd)}[/dim]")
    console.print("[dim]RAVE training takes several hours. Progress will be shown below.[/dim]")
    console.print()

    try:
        # Run training, showing output in real-time
        result = subprocess.run(cmd, check=False)
        return result.returncode == 0
    except KeyboardInterrupt:
        console.print("\n[yellow]Training interrupted. Partial checkpoints may be saved.[/yellow]")
        return False
    except Exception as e:
        console.print(f"[red]RAVE training failed: {e}[/red]")
        return False


def rave_export(
    run_path: Path,
    output_path: Optional[Path] = None,
    streaming: bool = False,
) -> Optional[Path]:
    """Export a trained RAVE model for inference.

    Args:
        run_path: Path to training run directory.
        output_path: Optional output path for exported model.
        streaming: Enable streaming mode for realtime use.

    Returns:
        Path to exported model, or None if failed.
    """
    if not check_rave_available():
        console.print("[red]RAVE CLI not found. Install with: pip install acids-rave[/red]")
        return None

    cmd = ["rave", "export", "--run", str(run_path)]

    if streaming:
        cmd.append("--streaming")

    console.print(f"[dim]Running: {' '.join(cmd)}[/dim]")

    try:
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            return None

        # Find the exported model
        exported = list(run_path.glob("*.ts"))
        if exported:
            return exported[0]

        return None
    except Exception as e:
        console.print(f"[red]RAVE export failed: {e}[/red]")
        return None


def rave_generate(
    model_path: Path,
    input_paths: List[Path],
    output_dir: Path,
    count: int = 10,
) -> bool:
    """Generate new audio using a trained RAVE model.

    RAVE generation works by encoding input audio and decoding with variations.
    For unconditional generation, we generate from random latent vectors.

    Args:
        model_path: Path to exported RAVE model (.ts file).
        input_paths: Input audio files to use as seeds.
        output_dir: Directory to save generated audio.
        count: Number of samples to generate.

    Returns:
        True if generation succeeded.
    """
    if not check_rave_available():
        console.print("[red]RAVE CLI not found. Install with: pip install acids-rave[/red]")
        return False

    output_dir.mkdir(parents=True, exist_ok=True)

    # RAVE generate command
    cmd = [
        "rave", "generate",
        str(model_path),
    ]

    # Add input files
    for p in input_paths[:count]:
        cmd.append(str(p))

    cmd.extend(["--out", str(output_dir)])

    console.print(f"[dim]Running: {' '.join(cmd)}[/dim]")

    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode == 0
    except Exception as e:
        console.print(f"[red]RAVE generation failed: {e}[/red]")
        return False


def find_latest_checkpoint(run_dir: Path) -> Optional[Path]:
    """Find the latest checkpoint in a RAVE training run.

    Args:
        run_dir: Path to training run directory.

    Returns:
        Path to latest checkpoint, or None if not found.
    """
    checkpoints = list(run_dir.glob("**/*.ckpt"))
    if not checkpoints:
        return None

    # Sort by modification time
    checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return checkpoints[0]


def find_exported_model(run_dir: Path) -> Optional[Path]:
    """Find an exported RAVE model in a training run.

    Args:
        run_dir: Path to training run directory.

    Returns:
        Path to exported .ts model, or None if not found.
    """
    models = list(run_dir.glob("*.ts"))
    if models:
        return models[0]

    # Check subdirectories
    models = list(run_dir.glob("**/*.ts"))
    if models:
        return models[0]

    return None
