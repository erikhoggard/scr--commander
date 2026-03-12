"""Split stage - uses scropipe.splitter for audio splitting."""

from pathlib import Path
from typing import Literal, Optional

from .base import Stage, StageResult


SplitMode = Literal["grid", "transient", "texture"]


class _Args:
    """Simple namespace to hold arguments for SampleProcessor."""

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class SplitStage(Stage):
    """Stage that splits audio files using the built-in splitter."""

    name = "splits"
    description = "Split audio into samples"

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
        # Sanitize the name — must match SampleProcessor's sanitization
        # (filepath.stem.replace('.', '_')) so we can find the output dir.
        source_name = source_name.replace(".", "_")

        # Import the splitter module
        try:
            from ..splitter import SampleProcessor
        except ImportError as e:
            return StageResult(
                success=False,
                message=f"Failed to import splitter: {e}",
            )

        # SampleProcessor internally creates a subdirectory named after the
        # file stem, so we pass self.output_dir (the "splits" directory) and
        # let the processor create the source_name level itself.
        self.ensure_output_dir()
        source_output_dir = self.output_dir

        # Calculate chunk_length from BPM if specified
        final_chunk_length = chunk_length
        if bpm:
            final_chunk_length = (60.0 / bpm) * bars * 4
        elif final_chunk_length is None:
            final_chunk_length = 2.0

        # Build args object
        args = _Args(
            channel=channel,
            chunk_length=final_chunk_length,
            bpm=bpm,
            bars=bars,
            delta=delta,
            min_length=min_length,
            max_length=max_length,
            min_duration=min_duration,
            max_duration=max_duration,
            rms_threshold=rms_threshold,
            stability_threshold=stability_threshold,
        )

        try:
            # Create processor and run
            processor = SampleProcessor(
                input_dir=None,
                output_dir=source_output_dir,
                sr=sample_rate
            )
            processor.process_single_file(input_file, modes=[mode], args=args)

            # Find output files under the source subdirectory that
            # SampleProcessor created (output_dir / sanitized_stem / ...).
            # The sanitized stem matches source_name since we already sanitized it.
            actual_output = source_output_dir / source_name
            wav_files = list(actual_output.rglob("*.wav")) if actual_output.exists() else []

            if not wav_files:
                return StageResult(
                    success=False,
                    output_dir=actual_output,
                    message="No samples generated",
                )

            self.log_success(f"Split '{source_name}' into {len(wav_files)} samples")

            return StageResult(
                success=True,
                output_dir=actual_output,
                message=f"Generated {len(wav_files)} samples",
                details={
                    "sample_count": len(wav_files),
                    "source_name": source_name,
                    "mode": mode,
                    "samples": [str(f) for f in wav_files[:10]],
                },
            )

        except Exception as e:
            return StageResult(
                success=False,
                message=f"Unexpected error: {e}",
            )
