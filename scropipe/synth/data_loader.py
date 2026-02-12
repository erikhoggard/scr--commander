"""PyTorch Dataset and DataLoader for spectrogram data."""

import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, Tuple, List

from . import audio_utils


class SpectrogramDataset(Dataset):
    """PyTorch Dataset for loading preprocessed spectrograms from .npy files."""

    def __init__(self, data_dir: Path, normalize: bool = True):
        """Initialize dataset.

        Args:
            data_dir: Directory containing .npy spectrogram files.
            normalize: Whether to normalize spectrograms to [0, 1] range.
        """
        self.data_dir = Path(data_dir)
        self.normalize = normalize
        self.metadata = None

        # Load metadata if available
        metadata_path = self.data_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                self.metadata = json.load(f)

        # Find all .npy files
        self.files = sorted(list(self.data_dir.glob("*.npy")))

        if len(self.files) == 0:
            raise ValueError(f"No .npy files found in {data_dir}")

        # Load first file to get shape and compute normalization stats
        sample = np.load(self.files[0])
        self.spec_shape = sample.shape

        # Compute global min/max for normalization
        if normalize:
            self._compute_normalization_stats()

    def _compute_normalization_stats(self):
        """Compute global min/max across all spectrograms for normalization."""
        global_min = float('inf')
        global_max = float('-inf')

        for f in self.files:
            spec = np.load(f)
            global_min = min(global_min, spec.min())
            global_max = max(global_max, spec.max())

        self.global_min = global_min
        self.global_max = global_max
        self.range = global_max - global_min

        # Avoid division by zero
        if self.range == 0:
            self.range = 1.0

    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Load and return a spectrogram.

        Args:
            idx: Index of sample to load.

        Returns:
            Spectrogram tensor of shape (1, H, W).
        """
        spec = np.load(self.files[idx])

        if self.normalize:
            # Normalize to [0, 1]
            spec = (spec - self.global_min) / self.range

        # Add channel dimension and convert to tensor
        spec = torch.from_numpy(spec).float().unsqueeze(0)

        return spec

    def denormalize(self, spec: torch.Tensor) -> torch.Tensor:
        """Denormalize spectrogram back to original scale.

        Args:
            spec: Normalized spectrogram tensor.

        Returns:
            Denormalized spectrogram tensor.
        """
        if not self.normalize:
            return spec

        return spec * self.range + self.global_min

    def get_normalization_params(self) -> Tuple[float, float]:
        """Get normalization parameters for saving/loading.

        Returns:
            Tuple of (global_min, range).
        """
        if self.normalize:
            return (self.global_min, self.range)
        return (0.0, 1.0)

    def get_metadata(self) -> Optional[dict]:
        """Get preprocessing metadata if available.

        Returns:
            Metadata dict or None if not available.
        """
        return self.metadata


class AudioSpectrogramDataset(Dataset):
    """Dataset that loads paired audio and spectrograms for vocoder training."""

    def __init__(self, audio_dir: Path, spec_dir: Path, max_duration: float = 2.0):
        """Initialize dataset.

        Args:
            audio_dir: Directory containing .wav audio files.
            spec_dir: Directory containing corresponding .npy spectrogram files.
            max_duration: Max duration in seconds.
        """
        self.audio_dir = Path(audio_dir)
        self.spec_dir = Path(spec_dir)
        self.max_duration = max_duration

        # Find matching pairs (audio file name should match spec file name)
        audio_files = {f.stem: f for f in self.audio_dir.glob("*.wav")}
        spec_files = {f.stem: f for f in self.spec_dir.glob("*.npy")}

        # Find common stems
        common = set(audio_files.keys()) & set(spec_files.keys())
        self.pairs: List[Tuple[Path, Path]] = [
            (audio_files[stem], spec_files[stem])
            for stem in sorted(common)
        ]

        if len(self.pairs) == 0:
            raise ValueError(f"No matching audio/spectrogram pairs found")

        # Load first spec to get shape
        sample = np.load(self.pairs[0][1])
        self.spec_shape = sample.shape

        # Compute normalization stats
        self._compute_normalization_stats()

    def _compute_normalization_stats(self):
        """Compute global min/max across all spectrograms."""
        global_min = float('inf')
        global_max = float('-inf')

        for _, spec_path in self.pairs:
            spec = np.load(spec_path)
            global_min = min(global_min, spec.min())
            global_max = max(global_max, spec.max())

        self.global_min = global_min
        self.global_max = global_max
        self.range = global_max - global_min
        if self.range == 0:
            self.range = 1.0

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load audio and spectrogram pair.

        Returns:
            Tuple of (audio, spectrogram) tensors.
            Audio shape: (1, samples)
            Spectrogram shape: (n_mels, time)
        """
        audio_path, spec_path = self.pairs[idx]

        # Load audio
        audio = audio_utils.load_and_standardize(audio_path, self.max_duration)
        audio = torch.from_numpy(audio).float().unsqueeze(0)  # (1, samples)

        # Load and normalize spectrogram
        spec = np.load(spec_path)
        spec = (spec - self.global_min) / self.range
        spec = torch.from_numpy(spec).float()  # (n_mels, time)

        return audio, spec

    def get_normalization_params(self) -> Tuple[float, float]:
        return (self.global_min, self.range)


def create_dataloader(
    data_dir: Path,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    normalize: bool = True
) -> Tuple[DataLoader, SpectrogramDataset]:
    """Create DataLoader for training.

    Args:
        data_dir: Directory containing .npy spectrogram files.
        batch_size: Batch size for training.
        shuffle: Whether to shuffle data.
        num_workers: Number of worker processes for loading.
        normalize: Whether to normalize spectrograms.

    Returns:
        Tuple of (DataLoader, Dataset).
    """
    dataset = SpectrogramDataset(data_dir, normalize=normalize)

    # Cap batch size to dataset size to ensure at least one batch
    effective_batch_size = min(batch_size, len(dataset))

    loader = DataLoader(
        dataset,
        batch_size=effective_batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=len(dataset) > effective_batch_size,  # Only drop if we have more than one batch
    )

    return loader, dataset
