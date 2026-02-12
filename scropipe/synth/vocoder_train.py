"""Training utilities for the HiFi-GAN vocoder."""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Optional
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich.console import Console

from .vocoder import (
    HiFiGANGenerator,
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
    feature_loss,
    discriminator_loss,
    generator_loss,
)
from .data_loader import AudioSpectrogramDataset
from . import audio_utils

console = Console()


class VocoderTrainer:
    """Trainer for HiFi-GAN vocoder."""

    def __init__(
        self,
        generator: HiFiGANGenerator,
        mpd: MultiPeriodDiscriminator,
        msd: MultiScaleDiscriminator,
        dataloader: DataLoader,
        dataset: AudioSpectrogramDataset,
        lr_g: float = 2e-4,
        lr_d: float = 2e-4,
        device: Optional[torch.device] = None,
    ):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.generator = generator.to(self.device)
        self.mpd = mpd.to(self.device)
        self.msd = msd.to(self.device)
        self.dataloader = dataloader
        self.dataset = dataset

        self.optim_g = optim.AdamW(generator.parameters(), lr=lr_g, betas=(0.8, 0.99))
        self.optim_d = optim.AdamW(
            list(mpd.parameters()) + list(msd.parameters()),
            lr=lr_d,
            betas=(0.8, 0.99)
        )

        self.scheduler_g = optim.lr_scheduler.ExponentialLR(self.optim_g, gamma=0.999)
        self.scheduler_d = optim.lr_scheduler.ExponentialLR(self.optim_d, gamma=0.999)

    def train_epoch(self, progress: Progress, task_id: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.generator.train()
        self.mpd.train()
        self.msd.train()

        total_g_loss = 0.0
        total_d_loss = 0.0
        num_batches = 0

        for audio, spec in self.dataloader:
            audio = audio.to(self.device)
            spec = spec.to(self.device)

            # Generate audio from spectrogram
            audio_gen = self.generator(spec)

            # Match lengths
            min_len = min(audio.shape[-1], audio_gen.shape[-1])
            audio = audio[..., :min_len]
            audio_gen = audio_gen[..., :min_len]

            # Train discriminators
            self.optim_d.zero_grad()

            # MPD
            y_df_r, y_df_g, _, _ = self.mpd(audio, audio_gen.detach())
            loss_disc_f, _, _ = discriminator_loss(y_df_r, y_df_g)

            # MSD
            y_ds_r, y_ds_g, _, _ = self.msd(audio, audio_gen.detach())
            loss_disc_s, _, _ = discriminator_loss(y_ds_r, y_ds_g)

            loss_d = loss_disc_f + loss_disc_s
            loss_d.backward()
            self.optim_d.step()

            # Train generator
            self.optim_g.zero_grad()

            # Mel spectrogram loss (L1)
            audio_gen_for_mel = audio_gen.squeeze(1)
            audio_for_mel = audio.squeeze(1)

            # Compute mel spectrograms
            mel_gen = self._compute_mel_batch(audio_gen_for_mel)
            mel_real = self._compute_mel_batch(audio_for_mel)
            loss_mel = torch.nn.functional.l1_loss(mel_gen, mel_real) * 45

            # Adversarial and feature matching losses
            y_df_r, y_df_g, fmap_f_r, fmap_f_g = self.mpd(audio, audio_gen)
            y_ds_r, y_ds_g, fmap_s_r, fmap_s_g = self.msd(audio, audio_gen)

            loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
            loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
            loss_gen_f = generator_loss(y_df_g)
            loss_gen_s = generator_loss(y_ds_g)

            loss_g = loss_gen_f + loss_gen_s + loss_fm_f + loss_fm_s + loss_mel
            loss_g.backward()
            self.optim_g.step()

            total_g_loss += loss_g.item()
            total_d_loss += loss_d.item()
            num_batches += 1

            progress.advance(task_id)

        self.scheduler_g.step()
        self.scheduler_d.step()

        return {
            'g_loss': total_g_loss / num_batches,
            'd_loss': total_d_loss / num_batches,
        }

    def _compute_mel_batch(self, audio: torch.Tensor) -> torch.Tensor:
        """Compute mel spectrograms for a batch of audio."""
        # Simple mel computation using torchaudio-like approach
        # This is a simplified version - in production you'd use the same
        # mel computation as preprocessing
        import torchaudio.transforms as T

        mel_transform = T.MelSpectrogram(
            sample_rate=audio_utils.SAMPLE_RATE,
            n_fft=audio_utils.N_FFT,
            hop_length=audio_utils.HOP_LENGTH,
            n_mels=audio_utils.N_MELS,
        ).to(self.device)

        mel = mel_transform(audio)
        mel = torch.log(torch.clamp(mel, min=1e-5))
        return mel

    def train(self, epochs: int, save_path: Path):
        """Train the vocoder."""
        console.print(f"[bold blue]Training vocoder on device: {self.device}[/bold blue]")
        console.print(f"[dim]Dataset size: {len(self.dataset)} samples[/dim]")
        console.print()

        best_loss = float('inf')

        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:

            epoch_task = progress.add_task("[cyan]Training vocoder...", total=epochs)

            for epoch in range(epochs):
                batch_task = progress.add_task(
                    f"[green]Epoch {epoch + 1}/{epochs}",
                    total=len(self.dataloader)
                )

                losses = self.train_epoch(progress, batch_task)

                if losses['g_loss'] < best_loss:
                    best_loss = losses['g_loss']
                    self._save_checkpoint(save_path, epoch, losses)

                progress.remove_task(batch_task)
                progress.advance(epoch_task)

                progress.console.print(
                    f"  Epoch {epoch + 1}: G={losses['g_loss']:.4f} D={losses['d_loss']:.4f}"
                )

        console.print()
        console.print(f"[bold green]Vocoder training complete![/bold green]")
        console.print(f"[dim]Model saved to: {save_path}[/dim]")

    def _save_checkpoint(self, path: Path, epoch: int, losses: Dict[str, float]):
        """Save vocoder checkpoint."""
        norm_params = self.dataset.get_normalization_params()

        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'mpd_state_dict': self.mpd.state_dict(),
            'msd_state_dict': self.msd.state_dict(),
            'losses': losses,
            'n_mels': audio_utils.N_MELS,
            'normalization': {
                'global_min': norm_params[0],
                'range': norm_params[1],
            },
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)


def load_vocoder(path: Path, device: Optional[torch.device] = None) -> HiFiGANGenerator:
    """Load trained vocoder from checkpoint."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(path, map_location=device, weights_only=False)

    generator = HiFiGANGenerator(n_mels=checkpoint.get('n_mels', audio_utils.N_MELS))
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.to(device)
    generator.eval()

    return generator
