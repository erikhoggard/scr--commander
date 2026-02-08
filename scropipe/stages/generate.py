"""Generate stage - wraps scronchler generate command."""

import subprocess
from pathlib import Path
from typing import Optional

from ..utils.discovery import find_tool
from .base import Stage, StageResult


class GenerateStage(Stage):
    """Stage that generates new audio samples from a trained model."""

    name = "05-generated"
    description = "Generate new samples from trained model"

    def run(
        self,
        model_path: Path,
        count: int = 10,
        seed: Optional[int] = None,
        vocoder_path: Optional[Path] = None,
    ) -> StageResult:
        """Run the generate stage.

        Args:
            model_path: Path to trained model file.
            count: Number of samples to generate.
            seed: Random seed for reproducibility.
            vocoder_path: Optional path to trained HiFi-GAN vocoder.

        Returns:
            StageResult with success status.
        """
        model_path = Path(model_path)
        if not model_path.exists():
            return StageResult(
                success=False,
                message=f"Model file not found: {model_path}",
            )

        try:
            scronchler = find_tool("scronchler")
        except Exception as e:
            return StageResult(success=False, message=str(e))

        output_dir = self.ensure_output_dir()

        # Build command
        cmd = [
            str(scronchler),
            "generate",
            "-m", str(model_path),
            "-o", str(output_dir),
            "--count", str(count),
        ]

        if seed is not None:
            cmd.extend(["--seed", str(seed)])

        if vocoder_path and Path(vocoder_path).exists():
            cmd.extend(["--vocoder", str(vocoder_path)])

        try:
            result = self.run_command(cmd, check=False)

            if result.returncode != 0:
                self.log_error(f"scronchler generate failed: {result.stderr}")
                return StageResult(
                    success=False,
                    message=f"scronchler exited with code {result.returncode}",
                    details={"stderr": result.stderr, "stdout": result.stdout},
                )

            # Count generated files
            wav_files = list(output_dir.glob("*.wav"))

            if not wav_files:
                return StageResult(
                    success=False,
                    output_dir=output_dir,
                    message="No samples generated",
                )

            self.log_success(f"Generated {len(wav_files)} AI samples")

            return StageResult(
                success=True,
                output_dir=output_dir,
                message=f"Generated {len(wav_files)} AI samples",
                details={
                    "sample_count": len(wav_files),
                    "samples": [str(f) for f in wav_files],
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
