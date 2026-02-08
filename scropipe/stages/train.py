"""Train stage - wraps scronchler train command."""

import subprocess
from pathlib import Path

from ..utils.discovery import find_tool
from .base import Stage, StageResult


class TrainStage(Stage):
    """Stage that trains a VAE model on spectrograms."""

    name = "03-model"
    description = "Train VAE model on spectrograms"

    def run(
        self,
        data_dir: Path,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        z_dim: int = 64,
        kl_weight: float = 0.001,
    ) -> StageResult:
        """Run the train stage.

        Args:
            data_dir: Directory containing preprocessed spectrograms.
            epochs: Number of training epochs.
            batch_size: Training batch size.
            learning_rate: Learning rate.
            z_dim: Latent space dimension.
            kl_weight: Weight for KL divergence in loss.

        Returns:
            StageResult with success status.
        """
        data_dir = Path(data_dir)
        if not data_dir.exists():
            return StageResult(
                success=False,
                message=f"Data directory not found: {data_dir}",
            )

        # Check for spectrograms
        npy_files = list(data_dir.glob("*.npy"))
        if not npy_files:
            return StageResult(
                success=False,
                message=f"No spectrogram files found in {data_dir}",
            )

        try:
            scronchler = find_tool("scronchler")
        except Exception as e:
            return StageResult(success=False, message=str(e))

        output_dir = self.ensure_output_dir()
        model_path = output_dir / "model.pth"

        # Build command
        cmd = [
            str(scronchler),
            "train",
            "-d", str(data_dir),
            "-m", str(model_path),
            "--epochs", str(epochs),
            "--batch-size", str(batch_size),
            "--lr", str(learning_rate),
            "--z-dim", str(z_dim),
            "--kl-weight", str(kl_weight),
        ]

        try:
            # Training can take a while, don't capture output so user sees progress
            result = subprocess.run(
                cmd,
                check=False,
            )

            if result.returncode != 0:
                self.log_error("scronchler train failed")
                return StageResult(
                    success=False,
                    message=f"scronchler exited with code {result.returncode}",
                )

            if not model_path.exists():
                return StageResult(
                    success=False,
                    output_dir=output_dir,
                    message="Model file not created",
                )

            self.log_success(f"Model saved to {model_path}")

            return StageResult(
                success=True,
                output_dir=output_dir,
                message=f"Model trained and saved to {model_path}",
                details={
                    "model_path": str(model_path),
                    "epochs": epochs,
                    "z_dim": z_dim,
                },
            )

        except subprocess.CalledProcessError as e:
            return StageResult(
                success=False,
                message=f"Command failed: {e}",
            )
        except Exception as e:
            return StageResult(
                success=False,
                message=f"Unexpected error: {e}",
            )
