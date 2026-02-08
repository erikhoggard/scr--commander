"""Preprocess stage - wraps scronchler preprocess command."""

import subprocess
from pathlib import Path
from typing import Optional

from ..utils.discovery import find_tool
from .base import Stage, StageResult


class PreprocessStage(Stage):
    """Stage that preprocesses audio samples into spectrograms."""

    name = "02-spectrograms"
    description = "Preprocess samples into mel spectrograms"

    def run(
        self,
        input_dir: Path,
        augment: bool = False,
        max_duration: float = 2.0,
    ) -> StageResult:
        """Run the preprocess stage.

        Args:
            input_dir: Directory containing WAV files.
            augment: Whether to generate augmented variations.
            max_duration: Maximum sample duration in seconds.

        Returns:
            StageResult with success status.
        """
        input_dir = Path(input_dir)
        if not input_dir.exists():
            return StageResult(
                success=False,
                message=f"Input directory not found: {input_dir}",
            )

        # Find WAV files (may be in subdirectories from scrumpler)
        wav_files = list(input_dir.rglob("*.wav"))
        if not wav_files:
            return StageResult(
                success=False,
                message=f"No WAV files found in {input_dir}",
            )

        try:
            scronchler = find_tool("scronchler")
        except Exception as e:
            return StageResult(success=False, message=str(e))

        output_dir = self.ensure_output_dir()

        # Build command
        cmd = [
            str(scronchler),
            "preprocess",
            "-i", str(input_dir),
            "-o", str(output_dir),
            "--max-duration", str(max_duration),
        ]

        if augment:
            cmd.append("--augment")

        try:
            result = self.run_command(cmd, check=False)

            if result.returncode != 0:
                self.log_error(f"scronchler preprocess failed: {result.stderr}")
                return StageResult(
                    success=False,
                    message=f"scronchler exited with code {result.returncode}",
                    details={"stderr": result.stderr, "stdout": result.stdout},
                )

            # Count output files
            npy_files = list(output_dir.glob("*.npy"))

            if not npy_files:
                return StageResult(
                    success=False,
                    output_dir=output_dir,
                    message="No spectrograms generated",
                )

            self.log_success(f"Generated {len(npy_files)} spectrograms")

            return StageResult(
                success=True,
                output_dir=output_dir,
                message=f"Generated {len(npy_files)} spectrograms",
                details={
                    "spectrogram_count": len(npy_files),
                    "augmented": augment,
                    "max_duration": max_duration,
                },
            )

        except subprocess.CalledProcessError as e:
            return StageResult(
                success=False,
                message=f"Command failed: {e}",
                details={"stderr": e.stderr if e.stderr else ""},
            )
        except Exception as e:
            return StageResult(
                success=False,
                message=f"Unexpected error: {e}",
            )
