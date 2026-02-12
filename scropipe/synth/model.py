"""Convolutional VAE model for audio spectrogram generation.

The model encodes log-mel spectrograms into a latent space and decodes
them back to spectrograms for audio reconstruction.
"""

import torch
import torch.nn as nn
from typing import Tuple


class Encoder(nn.Module):
    """Convolutional encoder that maps spectrograms to latent space."""

    def __init__(self, input_shape: Tuple[int, int], z_dim: int = 64):
        """Initialize encoder.

        Args:
            input_shape: (height, width) of input spectrograms.
            z_dim: Dimension of latent space.
        """
        super().__init__()
        self.z_dim = z_dim
        self.input_shape = input_shape

        # Convolutional layers
        self.conv_layers = nn.Sequential(
            # Input: 1 x H x W
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )

        # Calculate flattened size after convolutions
        self._compute_flat_size()

        # Latent space projections
        self.fc_mu = nn.Linear(self.flat_size, z_dim)
        self.fc_logvar = nn.Linear(self.flat_size, z_dim)

    def _compute_flat_size(self):
        """Compute the flattened size after conv layers."""
        with torch.no_grad():
            dummy = torch.zeros(1, 1, *self.input_shape)
            out = self.conv_layers(dummy)
            self.conv_output_shape = out.shape[1:]  # (C, H, W)
            self.flat_size = out.numel()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent distribution parameters.

        Args:
            x: Input tensor of shape (batch, 1, H, W).

        Returns:
            Tuple of (mu, logvar) tensors of shape (batch, z_dim).
        """
        h = self.conv_layers(x)
        h = h.view(h.size(0), -1)

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar


class Decoder(nn.Module):
    """Convolutional decoder that reconstructs spectrograms from latent space."""

    def __init__(self, output_shape: Tuple[int, int], z_dim: int = 64,
                 conv_output_shape: Tuple[int, int, int] = None):
        """Initialize decoder.

        Args:
            output_shape: (height, width) of output spectrograms.
            z_dim: Dimension of latent space.
            conv_output_shape: Shape after encoder conv layers (from encoder).
        """
        super().__init__()
        self.z_dim = z_dim
        self.output_shape = output_shape
        self.conv_output_shape = conv_output_shape or (256, 8, 6)

        flat_size = self.conv_output_shape[0] * self.conv_output_shape[1] * self.conv_output_shape[2]

        # Project from latent space
        self.fc = nn.Linear(z_dim, flat_size)

        # Transposed convolutional layers
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
        )

        # Adaptive layer to ensure correct output size
        self.adapt = nn.AdaptiveAvgPool2d(output_shape)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to spectrogram.

        Args:
            z: Latent tensor of shape (batch, z_dim).

        Returns:
            Reconstructed spectrogram of shape (batch, 1, H, W).
        """
        h = self.fc(z)
        h = h.view(h.size(0), *self.conv_output_shape)
        h = self.deconv_layers(h)
        out = self.adapt(h)
        return out


class VAE(nn.Module):
    """Variational Autoencoder for spectrogram generation."""

    def __init__(self, input_shape: Tuple[int, int], z_dim: int = 64):
        """Initialize VAE.

        Args:
            input_shape: (height, width) of spectrograms.
            z_dim: Dimension of latent space.
        """
        super().__init__()
        self.z_dim = z_dim
        self.input_shape = input_shape

        self.encoder = Encoder(input_shape, z_dim)
        self.decoder = Decoder(
            input_shape, z_dim,
            conv_output_shape=self.encoder.conv_output_shape
        )

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for sampling from latent distribution.

        Args:
            mu: Mean of latent distribution.
            logvar: Log variance of latent distribution.

        Returns:
            Sampled latent vector.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through VAE.

        Args:
            x: Input spectrogram tensor of shape (batch, 1, H, W).

        Returns:
            Tuple of (reconstruction, mu, logvar).
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

    def generate(self, num_samples: int, device: torch.device = None) -> torch.Tensor:
        """Generate new spectrograms by sampling from latent space.

        Args:
            num_samples: Number of samples to generate.
            device: Device to generate on.

        Returns:
            Generated spectrograms of shape (num_samples, 1, H, W).
        """
        device = device or next(self.parameters()).device
        z = torch.randn(num_samples, self.z_dim, device=device)
        with torch.no_grad():
            generated = self.decoder(z)
        return generated

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent vector (using mean).

        Args:
            x: Input spectrogram.

        Returns:
            Latent vector (mu).
        """
        mu, _ = self.encoder(x)
        return mu


def vae_loss(recon: torch.Tensor, target: torch.Tensor,
             mu: torch.Tensor, logvar: torch.Tensor,
             kl_weight: float = 0.001) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute VAE loss (reconstruction + KL divergence).

    Args:
        recon: Reconstructed spectrogram.
        target: Original spectrogram.
        mu: Latent mean.
        logvar: Latent log variance.
        kl_weight: Weight for KL divergence term.

    Returns:
        Tuple of (total_loss, recon_loss, kl_loss).
    """
    # Reconstruction loss (MSE)
    recon_loss = nn.functional.mse_loss(recon, target, reduction='mean')

    # KL divergence
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    # Total loss
    total_loss = recon_loss + kl_weight * kl_loss

    return total_loss, recon_loss, kl_loss
