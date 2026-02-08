"""Split stage - wraps scrumpler for audio splitting."""

import subprocess
from pathlib import Path
from typing import Literal, Optional

from ..utils.discovery import find_tool
from .base import Stage, StageResult


SplitMode = Literal["grid", "transient", "texture"]


class SplitStage(Stage):
    """Stage that splits audio files using scrumpler."""

    name = "splits"
    description = "Split audio into samples using scrumpler"

    def get_split_output_dir(self, source_name: str) -> Path:
        """Get output directory for a specific source.

        Args:
            source_name: Name of the source (typically input filename stem).

        Returns:
            Path to the output directory for this source.
        """
        return self.output_dir / source_name

    def run(
        self,
        input_file: Path,
        mode: SplitMode = "transient",
        source_name: Optional[str] = None,
        # Grid mode options
        chunk_length: Optional[float] = None,
        bpm: Optional[float] = None,
        bars: int = 4,
        # Transient mode options
        delta: float = 0.07,
        min_length: float = 0.05,
        max_length: float = 10.0,
        # Texture mode options
        min_duration: float = 1.0,
        max_duration: float = 30.0,
        rms_threshold: float = 0.1,
        stability_threshold: float = 0.15,
        # General options
        channel: str = "mono",
        sample_rate: int = 44100,
    ) -> StageResult:
        """Run the split stage.

        Args:
            input_file: Path to input audio file.
            mode: Split mode (grid, transient, texture).
            source_name: Name for this source (default: input filename stem).
            chunk_length: Chunk length for grid mode.
            bpm: BPM for musical grid chopping.
            bars: Number of bars per chunk when using BPM.
            delta: Transient detection sensitivity.
            min_length: Minimum segment length for transient mode.
            max_length: Maximum segment length for transient mode.
            min_duration: Minimum duration for texture mode.
            max_duration: Maximum duration for texture mode.
            rms_threshold: RMS threshold for texture detection.
            stability_threshold: Spectral stability threshold.
            channel: Channel selection (left, right, mono).
            sample_rate: Sample rate for processing.

        Returns:
            StageResult with success status.
        """
        input_file = Path(input_file)
        if not input_file.exists():
            return StageResult(
                success=False,
                message=f"Input file not found: {input_file}",
            )

        # Derive source name from filename if not provided
        if source_name is None:
            source_name = input_file.stem
        # Sanitize the name
        source_name = source_name.replace(" ", "_").replace(".", "_")

        try:
            scrumpler = find_tool("scrumpler")
        except Exception as e:
            return StageResult(success=False, message=str(e))

        # Create output directory for this specific source
        self.ensure_output_dir()  # Ensure base "splits" directory exists
        source_output_dir = self.get_split_output_dir(source_name)
        source_output_dir.mkdir(parents=True, exist_ok=True)

        # Build command
        cmd = [
            str(scrumpler),
            str(input_file),
            "-o", str(source_output_dir),
            f"--{mode}",
            "--channel", channel,
            "--sr", str(sample_rate),
        ]

        # Add mode-specific options
        if mode == "grid":
            if bpm:
                cmd.extend(["--bpm", str(bpm), "--bars", str(bars)])
            elif chunk_length:
                cmd.extend(["--chunk-length", str(chunk_length)])
            else:
                cmd.extend(["--chunk-length", "2.0"])

        elif mode == "transient":
            cmd.extend([
                "--delta", str(delta),
                "--min-length", str(min_length),
                "--max-length", str(max_length),
            ])

        elif mode == "texture":
            cmd.extend([
                "--min-duration", str(min_duration),
                "--max-duration", str(max_duration),
                "--rms-threshold", str(rms_threshold),
                "--stability-threshold", str(stability_threshold),
            ])

        try:
            result = self.run_command(cmd, check=False)

            if result.returncode != 0:
                self.log_error(f"scrumpler failed: {result.stderr}")
                return StageResult(
                    success=False,
                    message=f"scrumpler exited with code {result.returncode}",
                    details={"stderr": result.stderr, "stdout": result.stdout},
                )

            # Find output files - scrumpler creates subdirectories
            wav_files = list(source_output_dir.rglob("*.wav"))

            if not wav_files:
                return StageResult(
                    success=False,
                    output_dir=source_output_dir,
                    message="No samples generated",
                )

            self.log_success(f"Split '{source_name}' into {len(wav_files)} samples")

            return StageResult(
                success=True,
                output_dir=source_output_dir,
                message=f"Generated {len(wav_files)} samples",
                details={
                    "sample_count": len(wav_files),
                    "source_name": source_name,
                    "mode": mode,
                    "samples": [str(f) for f in wav_files[:10]],
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
