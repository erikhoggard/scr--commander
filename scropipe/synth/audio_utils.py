"""Audio processing utilities for the Neural Audio Synthesizer.

Handles audio loading, resampling, augmentation, and spectrogram conversion.
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path

# Audio constants
SAMPLE_RATE = 48000
DEFAULT_DURATION = 2.0


def get_n_samples(duration: float = DEFAULT_DURATION) -> int:
    """Calculate number of samples for a given duration.

    Args:
        duration: Duration in seconds.

    Returns:
        Number of samples.
    """
    return int(SAMPLE_RATE * duration)

# Spectrogram constants
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128


def load_and_standardize(file_path: Path, max_duration: float = DEFAULT_DURATION) -> np.ndarray:
    """Load audio file and standardize to target sample rate and duration.

    Args:
        file_path: Path to the audio file.
        max_duration: Maximum duration in seconds.

    Returns:
        Mono audio array at SAMPLE_RATE, exactly n_samples long.
    """
    n_samples = get_n_samples(max_duration)
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)

    # Pad or trim to exact length
    if len(audio) < n_samples:
        audio = np.pad(audio, (0, n_samples - len(audio)), mode='constant')
    else:
        audio = audio[:n_samples]

    return audio


def pitch_shift(audio: np.ndarray, semitones: float, max_duration: float = DEFAULT_DURATION) -> np.ndarray:
    """Shift pitch by given number of semitones.

    Args:
        audio: Input audio array.
        semitones: Number of semitones to shift (positive or negative).
        max_duration: Maximum duration in seconds.

    Returns:
        Pitch-shifted audio array.
    """
    n_samples = get_n_samples(max_duration)
    shifted = librosa.effects.pitch_shift(
        audio, sr=SAMPLE_RATE, n_steps=semitones
    )
    # Ensure length consistency
    if len(shifted) < n_samples:
        shifted = np.pad(shifted, (0, n_samples - len(shifted)), mode='constant')
    return shifted[:n_samples]


def time_stretch(audio: np.ndarray, rate: float, max_duration: float = DEFAULT_DURATION) -> np.ndarray:
    """Stretch audio by given rate (>1 = faster, <1 = slower).

    Args:
        audio: Input audio array.
        rate: Time stretch rate.
        max_duration: Maximum duration in seconds.

    Returns:
        Time-stretched audio array, padded/trimmed to original length.
    """
    n_samples = get_n_samples(max_duration)
    stretched = librosa.effects.time_stretch(audio, rate=rate)
    # Ensure length consistency
    if len(stretched) < n_samples:
        stretched = np.pad(stretched, (0, n_samples - len(stretched)), mode='constant')
    return stretched[:n_samples]


def add_noise(audio: np.ndarray, noise_level: float = 0.005) -> np.ndarray:
    """Add white noise to audio.

    Args:
        audio: Input audio array.
        noise_level: Standard deviation of noise (default: 0.005).

    Returns:
        Audio with added noise.
    """
    noise = np.random.normal(0, noise_level, len(audio))
    noisy = audio + noise
    # Clip to prevent clipping artifacts
    return np.clip(noisy, -1.0, 1.0)


def augment_audio(audio: np.ndarray, max_duration: float = DEFAULT_DURATION) -> list[tuple[str, np.ndarray]]:
    """Generate augmented versions of the audio.

    Args:
        audio: Input audio array.
        max_duration: Maximum duration in seconds.

    Returns:
        List of (suffix, augmented_audio) tuples.
    """
    augmentations = [
        ("_pitch_up", pitch_shift(audio, 2, max_duration)),
        ("_pitch_down", pitch_shift(audio, -2, max_duration)),
        ("_stretch", time_stretch(audio, 1.1, max_duration)),
        ("_noise", add_noise(audio)),
    ]
    return augmentations


def audio_to_mel_spectrogram(audio: np.ndarray) -> np.ndarray:
    """Convert audio to log-mel spectrogram.

    Args:
        audio: Input audio array.

    Returns:
        Log-mel spectrogram as 2D numpy array.
    """
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
    )
    # Convert to log scale (dB)
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)
    return log_mel


def mel_spectrogram_to_audio(log_mel: np.ndarray) -> np.ndarray:
    """Convert log-mel spectrogram back to audio using Griffin-Lim.

    Args:
        log_mel: Log-mel spectrogram.

    Returns:
        Reconstructed audio waveform.
    """
    # Convert from dB back to power
    mel_spec = librosa.db_to_power(log_mel)

    # Invert mel spectrogram to linear spectrogram
    linear_spec = librosa.feature.inverse.mel_to_stft(
        mel_spec,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
    )

    # Griffin-Lim reconstruction (more iterations = cleaner phase estimation)
    audio = librosa.griffinlim(
        linear_spec,
        n_iter=256,
        hop_length=HOP_LENGTH,
        n_fft=N_FFT,
    )

    # Normalize
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio)) * 0.9

    return audio


def save_audio(audio: np.ndarray, file_path: Path) -> None:
    """Save audio array to WAV file.

    Args:
        audio: Audio array to save.
        file_path: Output file path.
    """
    sf.write(file_path, audio, SAMPLE_RATE)


def get_spectrogram_shape(max_duration: float = DEFAULT_DURATION) -> tuple[int, int]:
    """Get the expected shape of spectrograms for model input.

    Args:
        max_duration: Maximum duration in seconds.

    Returns:
        Tuple of (n_mels, time_frames).
    """
    n_samples = get_n_samples(max_duration)
    # Calculate expected time frames
    time_frames = 1 + (n_samples - N_FFT) // HOP_LENGTH
    return (N_MELS, time_frames)
