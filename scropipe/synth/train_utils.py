"""Training utilities for the VAE model."""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Optional, Dict, Any
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich.console import Console

from .model import VAE, vae_loss
from .data_loader import SpectrogramDataset


console = Console()


class Trainer:
    """Trainer class for VAE model."""

    def __init__(
        self,
        model: VAE,
        dataloader: DataLoader,
        dataset: SpectrogramDataset,
        lr: float = 1e-3,
        kl_weight: float = 0.001,
        device: Optional[torch.device] = None,
    ):
        """Initialize trainer.

        Args:
            model: VAE model to train.
            dataloader: DataLoader for training data.
            dataset: Dataset (for normalization params).
            lr: Learning rate.
            kl_weight: Weight for KL divergence in loss.
            device: Device to train on.
        """
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = model.to(self.device)
        self.dataloader = dataloader
        self.dataset = dataset
        self.kl_weight = kl_weight

        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )

        self.history: Dict[str, list] = {
            'total_loss': [],
            'recon_loss': [],
            'kl_loss': [],
        }

    def train_epoch(self, progress: Progress, task_id: int) -> Dict[str, float]:
        """Train for one epoch.

        Args:
            progress: Rich progress bar.
            task_id: Progress bar task ID.

        Returns:
            Dictionary of average losses for the epoch.
        """
        self.model.train()

        total_loss_sum = 0.0
        recon_loss_sum = 0.0
        kl_loss_sum = 0.0
        num_batches = 0

        for batch in self.dataloader:
            batch = batch.to(self.device)

            self.optimizer.zero_grad()

            recon, mu, logvar = self.model(batch)
            total_loss, recon_loss, kl_loss = vae_loss(
                recon, batch, mu, logvar, self.kl_weight
            )

            total_loss.backward()
            self.optimizer.step()

            total_loss_sum += total_loss.item()
            recon_loss_sum += recon_loss.item()
            kl_loss_sum += kl_loss.item()
            num_batches += 1

            progress.advance(task_id)

        avg_losses = {
            'total_loss': total_loss_sum / num_batches,
            'recon_loss': recon_loss_sum / num_batches,
            'kl_loss': kl_loss_sum / num_batches,
        }

        return avg_losses

    def train(self, epochs: int, save_path: Path) -> Dict[str, list]:
        """Train the model for specified number of epochs.

        Args:
            epochs: Number of epochs to train.
            save_path: Path to save model weights.

        Returns:
            Training history dictionary.
        """
        console.print(f"[bold blue]Training on device: {self.device}[/bold blue]")
        console.print(f"[dim]Dataset size: {len(self.dataset)} samples[/dim]")
        console.print(f"[dim]Batches per epoch: {len(self.dataloader)}[/dim]")
        console.print()

        best_loss = float('inf')

        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:

            epoch_task = progress.add_task("[cyan]Training...", total=epochs)

            for epoch in range(epochs):
                batch_task = progress.add_task(
                    f"[green]Epoch {epoch + 1}/{epochs}",
                    total=len(self.dataloader)
                )

                avg_losses = self.train_epoch(progress, batch_task)

                # Update history
                for key, value in avg_losses.items():
                    self.history[key].append(value)

                # Update scheduler
                self.scheduler.step(avg_losses['total_loss'])

                # Save best model
                if avg_losses['total_loss'] < best_loss:
                    best_loss = avg_losses['total_loss']
                    self._save_checkpoint(save_path, epoch, avg_losses)

                progress.remove_task(batch_task)
                progress.advance(epoch_task)

                # Log epoch summary
                progress.console.print(
                    f"  Epoch {epoch + 1}: "
                    f"Loss={avg_losses['total_loss']:.4f} "
                    f"(Recon={avg_losses['recon_loss']:.4f}, "
                    f"KL={avg_losses['kl_loss']:.4f})"
                )

        console.print()
        console.print(f"[bold green]Training complete![/bold green]")
        console.print(f"[dim]Best model saved to: {save_path}[/dim]")

        return self.history

    def _save_checkpoint(self, path: Path, epoch: int, losses: Dict[str, float]):
        """Save model checkpoint.

        Args:
            path: Path to save checkpoint.
            epoch: Current epoch number.
            losses: Current loss values.
        """
        # Get normalization params from dataset
        norm_params = self.dataset.get_normalization_params()

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'losses': losses,
            'input_shape': self.model.input_shape,
            'z_dim': self.model.z_dim,
            'normalization': {
                'global_min': norm_params[0],
                'range': norm_params[1],
            },
        }

        # Include preprocessing metadata if available
        metadata = self.dataset.get_metadata()
        if metadata:
            checkpoint['preprocessing'] = metadata

        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)


def load_model(path: Path, device: Optional[torch.device] = None) -> tuple[VAE, Dict[str, Any]]:
    """Load trained model from checkpoint.

    Args:
        path: Path to checkpoint file.
        device: Device to load model to.

    Returns:
        Tuple of (model, checkpoint_dict).
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(path, map_location=device, weights_only=False)

    model = VAE(
        input_shape=checkpoint['input_shape'],
        z_dim=checkpoint['z_dim'],
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model, checkpoint
