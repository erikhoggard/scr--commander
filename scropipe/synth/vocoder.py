"""HiFi-GAN vocoder for mel spectrogram to audio conversion.

A neural vocoder that produces much higher quality audio than Griffin-Lim.
Can be trained alongside the VAE or used with pretrained weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
import numpy as np


class ResBlock(nn.Module):
    """Residual block with dilated convolutions."""

    def __init__(self, channels: int, kernel_size: int, dilations: List[int]):
        super().__init__()
        self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()

        for dilation in dilations:
            padding = (kernel_size * dilation - dilation) // 2
            self.convs1.append(
                nn.utils.parametrizations.weight_norm(
                    nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding=padding)
                )
            )
            self.convs2.append(
                nn.utils.parametrizations.weight_norm(
                    nn.Conv1d(channels, channels, kernel_size, dilation=1, padding=(kernel_size - 1) // 2)
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv1, conv2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, 0.1)
            xt = conv1(xt)
            xt = F.leaky_relu(xt, 0.1)
            xt = conv2(xt)
            x = xt + x
        return x


class HiFiGANGenerator(nn.Module):
    """HiFi-GAN generator for mel-to-audio conversion.

    Simplified HiFi-GAN v1 architecture optimized for our use case.
    """

    def __init__(
        self,
        n_mels: int = 128,
        upsample_rates: List[int] = [8, 8, 2, 2],
        upsample_kernel_sizes: List[int] = [16, 16, 4, 4],
        upsample_initial_channel: int = 512,
        resblock_kernel_sizes: List[int] = [3, 7, 11],
        resblock_dilations: List[List[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    ):
        super().__init__()

        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)

        # Initial conv
        self.conv_pre = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(n_mels, upsample_initial_channel, 7, padding=3)
        )

        # Upsampling layers
        self.ups = nn.ModuleList()
        ch = upsample_initial_channel
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                nn.utils.parametrizations.weight_norm(
                    nn.ConvTranspose1d(
                        ch, ch // 2, k, stride=u, padding=(k - u) // 2
                    )
                )
            )
            ch = ch // 2

        # Residual blocks
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilations):
                self.resblocks.append(ResBlock(ch, k, d))

        # Output conv
        self.conv_post = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(ch, 1, 7, padding=3)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert mel spectrogram to audio.

        Args:
            x: Mel spectrogram tensor of shape (batch, n_mels, time)

        Returns:
            Audio waveform of shape (batch, 1, samples)
        """
        x = self.conv_pre(x)

        for i, up in enumerate(self.ups):
            x = F.leaky_relu(x, 0.1)
            x = up(x)

            # Apply residual blocks
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        x = F.leaky_relu(x, 0.1)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x


class PeriodDiscriminator(nn.Module):
    """Multi-period discriminator component."""

    def __init__(self, period: int):
        super().__init__()
        self.period = period

        self.convs = nn.ModuleList([
            nn.utils.parametrizations.weight_norm(nn.Conv2d(1, 32, (5, 1), (3, 1), padding=(2, 0))),
            nn.utils.parametrizations.weight_norm(nn.Conv2d(32, 128, (5, 1), (3, 1), padding=(2, 0))),
            nn.utils.parametrizations.weight_norm(nn.Conv2d(128, 512, (5, 1), (3, 1), padding=(2, 0))),
            nn.utils.parametrizations.weight_norm(nn.Conv2d(512, 1024, (5, 1), (3, 1), padding=(2, 0))),
            nn.utils.parametrizations.weight_norm(nn.Conv2d(1024, 1024, (5, 1), 1, padding=(2, 0))),
        ])
        self.conv_post = nn.utils.parametrizations.weight_norm(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        fmap = []

        # Reshape to 2D
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)

        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class ScaleDiscriminator(nn.Module):
    """Multi-scale discriminator component."""

    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.utils.parametrizations.weight_norm(nn.Conv1d(1, 128, 15, 1, padding=7)),
            nn.utils.parametrizations.weight_norm(nn.Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            nn.utils.parametrizations.weight_norm(nn.Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            nn.utils.parametrizations.weight_norm(nn.Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            nn.utils.parametrizations.weight_norm(nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            nn.utils.parametrizations.weight_norm(nn.Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            nn.utils.parametrizations.weight_norm(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = nn.utils.parametrizations.weight_norm(nn.Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        fmap = []
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap


class MultiPeriodDiscriminator(nn.Module):
    """Multi-period discriminator for HiFi-GAN training."""

    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            PeriodDiscriminator(2),
            PeriodDiscriminator(3),
            PeriodDiscriminator(5),
            PeriodDiscriminator(7),
            PeriodDiscriminator(11),
        ])

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []

        for d in self.discriminators:
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class MultiScaleDiscriminator(nn.Module):
    """Multi-scale discriminator for HiFi-GAN training."""

    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            ScaleDiscriminator(),
            ScaleDiscriminator(),
            ScaleDiscriminator(),
        ])
        self.pools = nn.ModuleList([
            nn.Identity(),
            nn.AvgPool1d(4, 2, padding=2),
            nn.AvgPool1d(4, 2, padding=2),
        ])

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []

        for pool, d in zip(self.pools, self.discriminators):
            y_pooled = pool(y)
            y_hat_pooled = pool(y_hat)
            y_d_r, fmap_r = d(y_pooled)
            y_d_g, fmap_g = d(y_hat_pooled)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


def feature_loss(fmap_r: List[List[torch.Tensor]], fmap_g: List[List[torch.Tensor]]) -> torch.Tensor:
    """Feature matching loss."""
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))
    return loss * 2


def discriminator_loss(disc_real_outputs: List[torch.Tensor], disc_generated_outputs: List[torch.Tensor]):
    """Discriminator loss (hinge loss)."""
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg ** 2)
        loss += r_loss + g_loss
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())
    return loss, r_losses, g_losses


def generator_loss(disc_outputs: List[torch.Tensor]) -> torch.Tensor:
    """Generator adversarial loss."""
    loss = 0
    for dg in disc_outputs:
        loss += torch.mean((1 - dg) ** 2)
    return loss
