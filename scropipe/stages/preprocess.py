"""Preprocess stage - converts audio to spectrograms using scropipe.synth."""

import json
from pathlib import Path

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

        # Find WAV files (may be in subdirectories from splitter)
        wav_files = list(input_dir.rglob("*.wav"))
        if not wav_files:
            return StageResult(
                success=False,
                message=f"No WAV files found in {input_dir}",
            )

        # Import synth module (requires ML dependencies)
        try:
            from ..synth import audio_utils
            import numpy as np
        except ImportError as e:
            return StageResult(
                success=False,
                message=f"ML dependencies not installed. Run: pip install scropipe[ml]\nError: {e}",
            )

        output_dir = self.ensure_output_dir()

        self.log(f"[bold blue]Preprocessing {len(wav_files)} audio files...[/bold blue]")
        self.log(f"[dim]Max duration: {max_duration}s[/dim]")
        if augment:
            self.log("[dim]Augmentation enabled: 5x samples will be generated[/dim]")

        processed_count = 0
        error_count = 0

        for wav_file in wav_files:
            try:
                # Load and standardize audio with configurable duration
                audio = audio_utils.load_and_standardize(wav_file, max_duration)
                base_name = wav_file.stem

                # Process original
                spec = audio_utils.audio_to_mel_spectrogram(audio)
                np.save(output_dir / f"{base_name}.npy", spec)
                processed_count += 1

                # Process augmentations
                if augment:
                    for suffix, aug_audio in audio_utils.augment_audio(audio, max_duration):
                        aug_spec = audio_utils.audio_to_mel_spectrogram(aug_audio)
                        np.save(output_dir / f"{base_name}{suffix}.npy", aug_spec)
                        processed_count += 1

            except Exception as e:
                self.log(f"[yellow]Warning: Failed to process {wav_file.name}: {e}[/yellow]")
                error_count += 1

        if processed_count == 0:
            return StageResult(
                success=False,
                output_dir=output_dir,
                message="No spectrograms generated",
            )

        # Save metadata for train/generate to read
        metadata = {
            "max_duration": max_duration,
            "sample_rate": audio_utils.SAMPLE_RATE,
            "n_mels": audio_utils.N_MELS,
            "n_fft": audio_utils.N_FFT,
            "hop_length": audio_utils.HOP_LENGTH,
            "spectrogram_shape": list(audio_utils.get_spectrogram_shape(max_duration)),
        }
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        self.log_success(f"Generated {processed_count} spectrograms")
        if error_count > 0:
            self.log(f"[yellow]Errors: {error_count} files[/yellow]")

        return StageResult(
            success=True,
            output_dir=output_dir,
            message=f"Generated {processed_count} spectrograms",
            details={
                "spectrogram_count": processed_count,
                "augmented": augment,
                "max_duration": max_duration,
                "errors": error_count,
            },
        )
