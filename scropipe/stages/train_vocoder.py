"""Train vocoder stage - wraps scronchler train-vocoder command."""

import subprocess
from pathlib import Path

from ..utils.discovery import find_tool
from .base import Stage, StageResult


class TrainVocoderStage(Stage):
    """Stage that trains a HiFi-GAN vocoder."""

    name = "04-vocoder"
    description = "Train HiFi-GAN vocoder"

    def run(
        self,
        audio_dir: Path,
        spec_dir: Path,
        epochs: int = 100,
        batch_size: int = 8,
        learning_rate: float = 2e-4,
    ) -> StageResult:
        """Run the vocoder training stage.

        Args:
            audio_dir: Directory containing original .wav files.
            spec_dir: Directory containing preprocessed spectrograms.
            epochs: Number of training epochs.
            batch_size: Training batch size.
            learning_rate: Learning rate.

        Returns:
            StageResult with success status.
        """
        audio_dir = Path(audio_dir)
        spec_dir = Path(spec_dir)

        if not audio_dir.exists():
            return StageResult(
                success=False,
                message=f"Audio directory not found: {audio_dir}",
            )

        if not spec_dir.exists():
            return StageResult(
                success=False,
                message=f"Spectrogram directory not found: {spec_dir}",
            )

        try:
            scronchler = find_tool("scronchler")
        except Exception as e:
            return StageResult(success=False, message=str(e))

        output_dir = self.ensure_output_dir()
        vocoder_path = output_dir / "vocoder.pth"

        # Build command
        cmd = [
            str(scronchler),
            "train-vocoder",
            "-a", str(audio_dir),
            "-s", str(spec_dir),
            "-o", str(vocoder_path),
            "--epochs", str(epochs),
            "--batch-size", str(batch_size),
            "--lr", str(learning_rate),
        ]

        try:
            result = subprocess.run(cmd, check=False)

            if result.returncode != 0:
                self.log_error("scronchler train-vocoder failed")
                return StageResult(
                    success=False,
                    message=f"scronchler exited with code {result.returncode}",
                )

            if not vocoder_path.exists():
                return StageResult(
                    success=False,
                    output_dir=output_dir,
                    message="Vocoder file not created",
                )

            self.log_success(f"Vocoder saved to {vocoder_path}")

            return StageResult(
                success=True,
                output_dir=output_dir,
                message=f"Vocoder trained and saved to {vocoder_path}",
                details={
                    "vocoder_path": str(vocoder_path),
                    "epochs": epochs,
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
