"""Train vocoder stage - trains HiFi-GAN using scropipe.synth."""

from pathlib import Path

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

        # Import synth modules (requires ML dependencies)
        try:
            import torch
            from ..synth import audio_utils
            from ..synth.data_loader import AudioSpectrogramDataset
            from ..synth.vocoder import HiFiGANGenerator, MultiPeriodDiscriminator, MultiScaleDiscriminator
            from ..synth.vocoder_train import VocoderTrainer
        except ImportError as e:
            return StageResult(
                success=False,
                message=f"ML dependencies not installed. Run: pip install scropipe[ml]\nError: {e}",
            )

        output_dir = self.ensure_output_dir()
        vocoder_path = output_dir / "vocoder.pth"

        try:
            # Create dataset
            dataset = AudioSpectrogramDataset(audio_dir, spec_dir)
            self.log(f"[dim]Found {len(dataset)} audio/spectrogram pairs[/dim]")

            # Create dataloader
            effective_batch_size = min(batch_size, len(dataset))
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=effective_batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=torch.cuda.is_available(),
            )

            # Create models
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            generator = HiFiGANGenerator(n_mels=audio_utils.N_MELS)
            mpd = MultiPeriodDiscriminator()
            msd = MultiScaleDiscriminator()

            # Count parameters
            g_params = sum(p.numel() for p in generator.parameters() if p.requires_grad)
            d_params = sum(p.numel() for p in mpd.parameters() if p.requires_grad)
            d_params += sum(p.numel() for p in msd.parameters() if p.requires_grad)
            self.log(f"[dim]Generator parameters: {g_params:,}[/dim]")
            self.log(f"[dim]Discriminator parameters: {d_params:,}[/dim]")

            # Create trainer and train
            trainer = VocoderTrainer(
                generator=generator,
                mpd=mpd,
                msd=msd,
                dataloader=dataloader,
                dataset=dataset,
                lr_g=learning_rate,
                lr_d=learning_rate,
                device=device,
            )

            trainer.train(epochs=epochs, save_path=vocoder_path)

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

        except ValueError as e:
            return StageResult(
                success=False,
                message=f"Dataset error: {e}",
            )
        except Exception as e:
            return StageResult(
                success=False,
                message=f"Unexpected error: {e}",
            )
