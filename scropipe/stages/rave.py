"""RAVE stages - wraps RAVE commands for high-quality audio synthesis."""

import subprocess
from pathlib import Path
from typing import Optional

from ..utils.discovery import ToolNotFoundError, find_tool
from .base import Stage, StageResult


def _find_rave() -> str:
    """Find the rave executable.

    Returns:
        Path to the rave command.

    Raises:
        ToolNotFoundError: If rave cannot be found.
    """
    return str(find_tool("rave"))


class RavePreprocessStage(Stage):
    """Stage that preprocesses audio for RAVE training."""

    name = "02-rave-data"
    description = "Preprocess audio for RAVE"

    def run(
        self,
        input_dir: Path,
        channels: int = 1,
        num_signal: int = 131072,  # RAVE default; due to bug needs chunks >= 6s
    ) -> StageResult:
        """Run RAVE preprocessing.

        Args:
            input_dir: Directory containing audio files.
            channels: Number of audio channels.
            num_signal: Window size in samples.

        Returns:
            StageResult with success status.
        """
        input_dir = Path(input_dir)
        if not input_dir.exists():
            return StageResult(
                success=False,
                message=f"Input directory not found: {input_dir}",
            )

        output_dir = self.ensure_output_dir()

        # Find rave executable
        try:
            rave_cmd = _find_rave()
        except ToolNotFoundError:
            return StageResult(
                success=False,
                message="RAVE not found. Set RAVE_PATH or add rave to PATH. "
                        "See README for installation instructions.",
            )

        # Call rave CLI directly
        # Note: --channels removed as it's not supported by all RAVE versions
        cmd = [
            rave_cmd, "preprocess",
            "--input_path", str(input_dir),
            "--output_path", str(output_dir),
            "--num_signal", str(num_signal),
        ]

        try:
            result = subprocess.run(cmd, check=False)

            if result.returncode != 0:
                self.log_error("RAVE preprocess failed")
                return StageResult(
                    success=False,
                    message=f"rave preprocess exited with code {result.returncode}",
                )

            self.log_success(f"RAVE dataset created at {output_dir}")

            return StageResult(
                success=True,
                output_dir=output_dir,
                message=f"RAVE dataset created at {output_dir}",
            )

        except Exception as e:
            return StageResult(
                success=False,
                message=f"Unexpected error: {e}",
            )


def _detect_gpu() -> Optional[int]:
    """Detect available GPU, handling both NVIDIA and AMD ROCm."""
    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return 0
    except ImportError:
        pass
    return None


class RaveTrainStage(Stage):
    """Stage that trains a RAVE model."""

    name = "03-rave-model"
    description = "Train RAVE model"

    def run(
        self,
        data_dir: Path,
        name: str = "model",
        config: str = "v2",
        epochs: Optional[int] = None,
        channels: int = 1,
        n_signal: int = 131072,  # RAVE default, must match preprocessing
        gpu: Optional[int] = None,
        workers: int = 0,  # 0 disables multiprocessing, avoids ROCm segfaults
        val_every: int = 500,  # Checkpoint every N steps (default 10000 is too infrequent)
        ckpt: Optional[Path] = None,
    ) -> StageResult:
        """Run RAVE training.

        Args:
            data_dir: Directory containing preprocessed RAVE dataset.
            name: Model name.
            config: RAVE config (v2, v2_small, etc.).
            epochs: Max epochs.
            channels: Number of audio channels.
            n_signal: Window size in samples (must match preprocessing).
            gpu: GPU index to use (auto-detected if None).
            workers: Number of DataLoader workers (0 to disable multiprocessing).
            val_every: Save checkpoint every N steps (default: 500).
            ckpt: Path to checkpoint to resume training from.

        Returns:
            StageResult with success status.
        """
        data_dir = Path(data_dir)
        if not data_dir.exists():
            return StageResult(
                success=False,
                message=f"Data directory not found: {data_dir}",
            )

        output_dir = self.ensure_output_dir()

        # Find rave executable
        try:
            rave_cmd = _find_rave()
        except ToolNotFoundError:
            return StageResult(
                success=False,
                message="RAVE not found. Set RAVE_PATH or add rave to PATH. "
                        "See README for installation instructions.",
            )

        # Auto-detect GPU if not specified (handles AMD ROCm)
        if gpu is None:
            gpu = _detect_gpu()

        # Call rave CLI directly to ensure correct n_signal
        # Note: rave train has no --out_path flag; it writes to cwd,
        # so we run it from output_dir.
        cmd = [
            rave_cmd, "train",
            "--config", config,
            "--db_path", str(data_dir),
            "--name", name,
            "--n_signal", str(n_signal),
        ]

        # Explicitly specify GPU to bypass GPUtil (which doesn't detect AMD)
        if gpu is not None:
            cmd.extend(["--gpu", str(gpu)])

        # Set workers (0 disables multiprocessing, helps with ROCm stability)
        cmd.extend(["--workers", str(workers)])

        # Checkpoint frequently (default 10000 is too infrequent for short runs)
        cmd.extend(["--val_every", str(val_every)])

        if epochs is not None:
            cmd.extend(["--max_steps", str(epochs)])

        if ckpt is not None:
            cmd.extend(["--ckpt", str(ckpt)])

        try:
            result = subprocess.run(cmd, check=False, cwd=str(output_dir))

            if result.returncode != 0:
                self.log_error("RAVE training failed")
                return StageResult(
                    success=False,
                    output_dir=output_dir,
                    message=f"rave train exited with code {result.returncode}",
                )

            self.log_success(f"RAVE model trained at {output_dir}")

            return StageResult(
                success=True,
                output_dir=output_dir,
                message=f"RAVE model trained at {output_dir}",
                details={"config": config},
            )

        except Exception as e:
            return StageResult(
                success=False,
                message=f"Unexpected error: {e}",
            )


class RaveExportStage(Stage):
    """Stage that exports a trained RAVE model."""

    name = "04-rave-export"
    description = "Export RAVE model"

    def run(
        self,
        run_dir: Path,
        streaming: bool = False,
    ) -> StageResult:
        """Export RAVE model for generation.

        Args:
            run_dir: Path to training run directory.
            streaming: Enable streaming mode.

        Returns:
            StageResult with success status.
        """
        run_dir = Path(run_dir)
        if not run_dir.exists():
            return StageResult(
                success=False,
                message=f"Run directory not found: {run_dir}",
            )

        # Find rave executable
        try:
            rave_cmd = _find_rave()
        except ToolNotFoundError:
            return StageResult(
                success=False,
                message="RAVE not found. Set RAVE_PATH or add rave to PATH. "
                        "See README for installation instructions.",
            )

        # Use rave CLI directly
        cmd = [rave_cmd, "export", "--run", str(run_dir)]

        if streaming:
            cmd.append("--streaming")

        try:
            result = subprocess.run(cmd, check=False)

            if result.returncode != 0:
                self.log_error("RAVE export failed")
                return StageResult(
                    success=False,
                    message=f"rave export exited with code {result.returncode}",
                )

            # Find exported model (searches recursively)
            exported = list(run_dir.glob("**/*.ts"))
            model_path = exported[0] if exported else None

            self.log_success(f"RAVE model exported")

            return StageResult(
                success=True,
                output_dir=run_dir,
                message=f"RAVE model exported",
                details={"model_path": str(model_path) if model_path else None},
            )

        except Exception as e:
            return StageResult(
                success=False,
                message=f"Unexpected error: {e}",
            )


class RaveGenerateStage(Stage):
    """Stage that generates audio using a trained RAVE model."""

    name = "05-rave-generated"
    description = "Generate with RAVE"

    def run(
        self,
        model_path: Path,
        input_dir: Path,
        count: int = 10,
        gpu: Optional[int] = None,
    ) -> StageResult:
        """Generate audio with RAVE.

        Uses Python API directly because rave generate CLI has a bug
        with get_valid_extensions() returning None.

        Args:
            model_path: Path to exported RAVE model (.ts file).
            input_dir: Directory with seed audio files.
            count: Number of samples to generate.
            gpu: GPU index (auto-detected if None).

        Returns:
            StageResult with success status.
        """
        model_path = Path(model_path)
        input_dir = Path(input_dir)

        if not model_path.exists():
            return StageResult(
                success=False,
                message=f"Model not found: {model_path}",
            )

        if not input_dir.exists():
            return StageResult(
                success=False,
                message=f"Input directory not found: {input_dir}",
            )

        output_dir = self.ensure_output_dir()

        # Auto-detect GPU
        if gpu is None:
            gpu = _detect_gpu()

        try:
            import torch
            import torchaudio

            # Load model
            model = torch.jit.load(str(model_path))
            if gpu is not None and gpu >= 0:
                device = torch.device(f'cuda:{gpu}')
                model = model.to(device)
            else:
                device = torch.device('cpu')

            # Get input files
            wav_files = sorted(input_dir.glob("*.wav"))[:count]

            if not wav_files:
                return StageResult(
                    success=False,
                    message=f"No WAV files found in {input_dir}",
                )

            generated_count = 0
            for f in wav_files:
                try:
                    x, sr = torchaudio.load(str(f))

                    # Convert to mono if needed (model expects 1 channel)
                    if x.shape[0] > 1:
                        x = x.mean(dim=0, keepdim=True)

                    # Resample if needed
                    if sr != model.sr:
                        x = torchaudio.functional.resample(x, sr, model.sr)

                    x = x.to(device)

                    # Generate
                    with torch.no_grad():
                        out = model.forward(x[None])

                    # Save
                    out_path = output_dir / f.name
                    torchaudio.save(str(out_path), out[0].cpu(), sample_rate=model.sr)
                    generated_count += 1

                except Exception as e:
                    self.log_error(f"Failed to process {f.name}: {e}")
                    continue

            self.log_success(f"Generated {generated_count} samples with RAVE")

            return StageResult(
                success=True,
                output_dir=output_dir,
                message=f"Generated {generated_count} RAVE samples",
                details={"sample_count": generated_count},
            )

        except ImportError as e:
            return StageResult(
                success=False,
                message=f"ML dependencies not installed. Run: pip install scropipe[ml]\nError: {e}",
            )
        except Exception as e:
            return StageResult(
                success=False,
                message=f"Unexpected error: {e}",
            )
