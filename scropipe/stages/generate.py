"""Generate stage - generates audio samples using scropipe.synth."""

from pathlib import Path
from typing import Optional

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

        # Import synth modules (requires ML dependencies)
        try:
            import torch
            import numpy as np
            from ..synth import audio_utils
            from ..synth.train_utils import load_model
            from ..synth.vocoder_train import load_vocoder
        except ImportError as e:
            return StageResult(
                success=False,
                message=f"ML dependencies not installed. Run: pip install scropipe[ml]\nError: {e}",
            )

        output_dir = self.ensure_output_dir()

        try:
            # Set seed if provided
            if seed is not None:
                torch.manual_seed(seed)
                np.random.seed(seed)
                self.log(f"[dim]Random seed: {seed}[/dim]")

            # Load model
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model, checkpoint = load_model(model_path, device)
            self.log(f"[dim]Loaded model from: {model_path}[/dim]")
            self.log(f"[dim]Device: {device}[/dim]")

            # Load vocoder if provided
            vocoder = None
            if vocoder_path and Path(vocoder_path).exists():
                try:
                    vocoder = load_vocoder(vocoder_path, device)
                    self.log(f"[dim]Using HiFi-GAN vocoder: {vocoder_path}[/dim]")
                except Exception as e:
                    self.log(f"[yellow]Warning: Failed to load vocoder: {e}[/yellow]")
                    self.log("[yellow]Falling back to Griffin-Lim[/yellow]")
            else:
                self.log("[dim]Using Griffin-Lim (no vocoder provided)[/dim]")

            # Get normalization params
            norm_params = checkpoint.get('normalization', {'global_min': -80.0, 'range': 80.0})
            global_min = norm_params['global_min']
            norm_range = norm_params['range']

            self.log(f"[dim]Generating {count} samples...[/dim]")

            generated_files = []
            for i in range(count):
                # Generate spectrogram
                with torch.no_grad():
                    generated = model.generate(1, device)

                    if vocoder:
                        # Use HiFi-GAN vocoder
                        # Vocoder expects (batch, n_mels, time)
                        spec_for_vocoder = generated[:, 0, :, :]  # Remove channel dim
                        audio_tensor = vocoder(spec_for_vocoder)
                        audio = audio_tensor[0, 0].cpu().numpy()
                        # Normalize
                        if np.max(np.abs(audio)) > 0:
                            audio = audio / np.max(np.abs(audio)) * 0.9
                    else:
                        # Use Griffin-Lim
                        spec = generated[0, 0].cpu().numpy()
                        spec = spec * norm_range + global_min
                        audio = audio_utils.mel_spectrogram_to_audio(spec)

                # Save audio
                output_path = output_dir / f"generated_{i:04d}.wav"
                audio_utils.save_audio(audio, output_path)
                generated_files.append(str(output_path))

            if not generated_files:
                return StageResult(
                    success=False,
                    output_dir=output_dir,
                    message="No samples generated",
                )

            self.log_success(f"Generated {len(generated_files)} AI samples")

            return StageResult(
                success=True,
                output_dir=output_dir,
                message=f"Generated {len(generated_files)} AI samples",
                details={
                    "sample_count": len(generated_files),
                    "samples": generated_files,
                },
            )

        except Exception as e:
            return StageResult(
                success=False,
                message=f"Unexpected error: {e}",
            )
