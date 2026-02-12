#!/usr/bin/env python3
"""Neural Audio Synthesizer CLI.

A command-line tool for training VAE models on audio samples and generating new variations.
"""

import typer
from pathlib import Path
from typing import Optional
import numpy as np
import torch
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn

from . import audio_utils
from .data_loader import create_dataloader, AudioSpectrogramDataset
from .model import VAE
from .train_utils import Trainer, load_model
from .vocoder import HiFiGANGenerator, MultiPeriodDiscriminator, MultiScaleDiscriminator
from .vocoder_train import VocoderTrainer, load_vocoder

app = typer.Typer(
    name="neural-sampler",
    help="Neural Audio Synthesizer - Generate new audio samples using VAE",
    add_completion=False,
)
console = Console()


@app.command()
def preprocess(
    input_dir: Path = typer.Option(
        ..., "--input-dir", "-i",
        help="Directory containing input .wav files",
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
    output_dir: Path = typer.Option(
        ..., "--output-dir", "-o",
        help="Directory to save preprocessed spectrograms",
    ),
    augment: bool = typer.Option(
        False, "--augment", "-a",
        help="Generate augmented variations of each sample",
    ),
    max_duration: float = typer.Option(
        2.0, "--max-duration",
        help="Maximum sample duration in seconds (samples are padded/trimmed to this length)",
    ),
):
    """Preprocess audio files into spectrograms for training.

    Converts .wav files to log-mel spectrograms, standardizing sample rate
    and duration. Optionally generates augmented variations.
    """
    import json

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all wav files
    wav_files = list(input_dir.glob("*.wav"))

    if not wav_files:
        console.print(f"[red]No .wav files found in {input_dir}[/red]")
        raise typer.Exit(1)

    console.print(f"[bold blue]Preprocessing {len(wav_files)} audio files...[/bold blue]")
    console.print(f"[dim]Max duration: {max_duration}s[/dim]")
    if augment:
        console.print("[dim]Augmentation enabled: 5x samples will be generated[/dim]")
    console.print()

    processed_count = 0
    error_count = 0

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:

        task = progress.add_task("Processing files...", total=len(wav_files))

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
                console.print(f"[yellow]Warning: Failed to process {wav_file.name}: {e}[/yellow]")
                error_count += 1

            progress.advance(task)

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

    console.print()
    console.print(f"[bold green]Preprocessing complete![/bold green]")
    console.print(f"[dim]Processed: {processed_count} spectrograms[/dim]")
    if error_count > 0:
        console.print(f"[yellow]Errors: {error_count} files[/yellow]")
    console.print(f"[dim]Output directory: {output_dir}[/dim]")


@app.command()
def train(
    data_dir: Path = typer.Option(
        ..., "--data-dir", "-d",
        help="Directory containing preprocessed .npy spectrograms",
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
    model_path: Path = typer.Option(
        ..., "--model-path", "-m",
        help="Path to save trained model weights (.pth)",
    ),
    epochs: int = typer.Option(
        100, "--epochs", "-e",
        help="Number of training epochs",
        min=1,
    ),
    batch_size: int = typer.Option(
        32, "--batch-size", "-b",
        help="Training batch size",
        min=1,
    ),
    lr: float = typer.Option(
        1e-3, "--lr", "-l",
        help="Learning rate",
    ),
    z_dim: int = typer.Option(
        64, "--z-dim", "-z",
        help="Latent space dimension",
        min=1,
    ),
    kl_weight: float = typer.Option(
        0.001, "--kl-weight",
        help="Weight for KL divergence in loss",
    ),
):
    """Train the VAE model on preprocessed spectrograms.

    Trains a Convolutional VAE to learn the latent space of audio samples.
    Model weights and normalization parameters are saved for generation.
    """
    console.print("[bold blue]Neural Audio Synthesizer - Training[/bold blue]")
    console.print()

    # Create dataloader
    try:
        dataloader, dataset = create_dataloader(
            data_dir,
            batch_size=batch_size,
            shuffle=True,
            normalize=True,
        )
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    # Get input shape from dataset
    input_shape = dataset.spec_shape
    console.print(f"[dim]Spectrogram shape: {input_shape}[/dim]")
    console.print(f"[dim]Latent dimension: {z_dim}[/dim]")
    console.print()

    # Create model
    model = VAE(input_shape=input_shape, z_dim=z_dim)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    console.print(f"[dim]Model parameters: {num_params:,}[/dim]")
    console.print()

    # Create trainer and train
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(
        model=model,
        dataloader=dataloader,
        dataset=dataset,
        lr=lr,
        kl_weight=kl_weight,
        device=device,
    )

    trainer.train(epochs=epochs, save_path=model_path)


@app.command("train-vocoder")
def train_vocoder(
    audio_dir: Path = typer.Option(
        ..., "--audio-dir", "-a",
        help="Directory containing input .wav files",
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
    spec_dir: Path = typer.Option(
        ..., "--spec-dir", "-s",
        help="Directory containing preprocessed .npy spectrograms",
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
    output_path: Path = typer.Option(
        ..., "--output", "-o",
        help="Path to save vocoder weights (.pth)",
    ),
    epochs: int = typer.Option(
        100, "--epochs", "-e",
        help="Number of training epochs",
        min=1,
    ),
    batch_size: int = typer.Option(
        8, "--batch-size", "-b",
        help="Training batch size (smaller for vocoder due to memory)",
        min=1,
    ),
    lr: float = typer.Option(
        2e-4, "--lr", "-l",
        help="Learning rate",
    ),
):
    """Train HiFi-GAN vocoder for high-quality audio generation.

    The vocoder converts mel spectrograms to audio waveforms.
    Train this after preprocessing your audio samples.
    """
    console.print("[bold blue]HiFi-GAN Vocoder Training[/bold blue]")
    console.print()

    # Create dataset
    try:
        dataset = AudioSpectrogramDataset(audio_dir, spec_dir)
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    console.print(f"[dim]Found {len(dataset)} audio/spectrogram pairs[/dim]")

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
    console.print(f"[dim]Generator parameters: {g_params:,}[/dim]")
    console.print(f"[dim]Discriminator parameters: {d_params:,}[/dim]")
    console.print()

    # Create trainer and train
    trainer = VocoderTrainer(
        generator=generator,
        mpd=mpd,
        msd=msd,
        dataloader=dataloader,
        dataset=dataset,
        lr_g=lr,
        lr_d=lr,
        device=device,
    )

    trainer.train(epochs=epochs, save_path=output_path)


@app.command()
def generate(
    model_path: Path = typer.Option(
        ..., "--model-path", "-m",
        help="Path to trained model weights (.pth)",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    output_dir: Path = typer.Option(
        ..., "--output-dir", "-o",
        help="Directory to save generated audio files",
    ),
    count: int = typer.Option(
        10, "--count", "-c",
        help="Number of samples to generate",
        min=1,
    ),
    seed: Optional[int] = typer.Option(
        None, "--seed", "-s",
        help="Random seed for reproducibility",
    ),
    vocoder_path: Optional[Path] = typer.Option(
        None, "--vocoder", "-v",
        help="Path to trained HiFi-GAN vocoder (uses Griffin-Lim if not provided)",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
):
    """Generate new audio samples from the trained model.

    Samples random points from the latent space and decodes them to audio.
    Uses HiFi-GAN vocoder if provided, otherwise falls back to Griffin-Lim.
    """
    console.print("[bold blue]Neural Audio Synthesizer - Generation[/bold blue]")
    console.print()

    # Set seed if provided
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        console.print(f"[dim]Random seed: {seed}[/dim]")

    # Load model
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, checkpoint = load_model(model_path, device)
        console.print(f"[dim]Loaded model from: {model_path}[/dim]")
        console.print(f"[dim]Device: {device}[/dim]")
    except Exception as e:
        console.print(f"[red]Error loading model: {e}[/red]")
        raise typer.Exit(1)

    # Load vocoder if provided
    vocoder = None
    if vocoder_path:
        try:
            vocoder = load_vocoder(vocoder_path, device)
            console.print(f"[dim]Using HiFi-GAN vocoder: {vocoder_path}[/dim]")
        except Exception as e:
            console.print(f"[yellow]Warning: Failed to load vocoder: {e}[/yellow]")
            console.print(f"[yellow]Falling back to Griffin-Lim[/yellow]")
    else:
        console.print(f"[dim]Using Griffin-Lim (no vocoder provided)[/dim]")

    # Get normalization params
    norm_params = checkpoint.get('normalization', {'global_min': -80.0, 'range': 80.0})
    global_min = norm_params['global_min']
    norm_range = norm_params['range']

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[dim]Generating {count} samples...[/dim]")
    console.print()

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:

        task = progress.add_task("Generating...", total=count)

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

            progress.advance(task)

    console.print()
    console.print(f"[bold green]Generation complete![/bold green]")
    console.print(f"[dim]Generated {count} samples in: {output_dir}[/dim]")


@app.command()
def info(
    model_path: Path = typer.Option(
        ..., "--model-path", "-m",
        help="Path to trained model weights (.pth)",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
):
    """Display information about a trained model."""
    try:
        device = torch.device("cpu")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

        console.print("[bold blue]Model Information[/bold blue]")
        console.print()
        console.print(f"Input shape: {checkpoint['input_shape']}")
        console.print(f"Latent dimension: {checkpoint['z_dim']}")
        console.print(f"Training epoch: {checkpoint['epoch'] + 1}")
        console.print(f"Final losses:")
        for key, value in checkpoint['losses'].items():
            console.print(f"  {key}: {value:.4f}")

        if 'normalization' in checkpoint:
            norm = checkpoint['normalization']
            console.print(f"Normalization:")
            console.print(f"  global_min: {norm['global_min']:.2f}")
            console.print(f"  range: {norm['range']:.2f}")

    except Exception as e:
        console.print(f"[red]Error loading model: {e}[/red]")
        raise typer.Exit(1)


# =============================================================================
# RAVE Commands (for high-quality melodic/harmonic audio)
# =============================================================================

@app.command("rave-preprocess")
def rave_preprocess_cmd(
    input_dir: Path = typer.Option(
        ..., "--input-dir", "-i",
        help="Directory containing input audio files",
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
    output_dir: Path = typer.Option(
        ..., "--output-dir", "-o",
        help="Directory to save preprocessed RAVE dataset",
    ),
    channels: int = typer.Option(
        1, "--channels", "-c",
        help="Number of audio channels (1=mono, 2=stereo)",
    ),
    num_signal: int = typer.Option(
        16384, "--num-signal",
        help="Window size in samples (~0.37s at 44100Hz)",
    ),
):
    """Preprocess audio for RAVE training.

    RAVE works directly on audio waveforms, not spectrograms.
    This prepares your audio files for RAVE training.
    """
    from . import rave_utils

    console.print("[bold blue]RAVE Preprocessing[/bold blue]")
    console.print()

    if not rave_utils.check_rave_available():
        console.print("[red]RAVE CLI not found.[/red]")
        console.print("[dim]Install with: pip install acids-rave[/dim]")
        raise typer.Exit(1)

    success = rave_utils.rave_preprocess(
        input_dir=input_dir,
        output_dir=output_dir,
        channels=channels,
        num_signal=num_signal,
    )

    if success:
        console.print()
        console.print("[bold green]Preprocessing complete![/bold green]")
        console.print(f"[dim]Dataset saved to: {output_dir}[/dim]")
    else:
        console.print("[red]Preprocessing failed[/red]")
        raise typer.Exit(1)


@app.command("rave-train")
def rave_train_cmd(
    data_dir: Path = typer.Option(
        ..., "--data-dir", "-d",
        help="Directory containing preprocessed RAVE dataset",
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
    output_dir: Path = typer.Option(
        ..., "--output-dir", "-o",
        help="Directory to save trained model",
    ),
    name: str = typer.Option(
        "model", "--name", "-n",
        help="Model name",
    ),
    config: str = typer.Option(
        "v2", "--config",
        help="RAVE config: v1, v2, v2_small, discrete, onnx, raspberry",
    ),
    epochs: Optional[int] = typer.Option(
        None, "--epochs", "-e",
        help="Max epochs (default: trains until convergence)",
    ),
    channels: int = typer.Option(
        1, "--channels", "-c",
        help="Number of audio channels",
    ),
):
    """Train a RAVE model on preprocessed audio.

    RAVE training takes several hours but produces high-quality results
    suitable for melodic and harmonic content.

    Configs:
    - v2: Default, good balance of quality and speed
    - v2_small: Smaller model, faster training
    - v1: Original architecture
    - discrete: For discrete latent codes
    """
    from . import rave_utils

    console.print("[bold blue]RAVE Training[/bold blue]")
    console.print()
    console.print(f"[dim]Config: {config}[/dim]")
    console.print(f"[dim]This will take several hours. Press Ctrl+C to stop.[/dim]")
    console.print()

    if not rave_utils.check_rave_available():
        console.print("[red]RAVE CLI not found.[/red]")
        console.print("[dim]Install with: pip install acids-rave[/dim]")
        raise typer.Exit(1)

    success = rave_utils.rave_train(
        db_path=data_dir,
        output_dir=output_dir,
        name=name,
        config=config,
        channels=channels,
        epochs=epochs,
    )

    if success:
        console.print()
        console.print("[bold green]Training complete![/bold green]")
        console.print(f"[dim]Model saved to: {output_dir}[/dim]")
        console.print()
        console.print("[dim]Next step: Export the model with 'scronchler rave-export'[/dim]")
    else:
        console.print("[yellow]Training stopped or failed[/yellow]")
        raise typer.Exit(1)


@app.command("rave-export")
def rave_export_cmd(
    run_dir: Path = typer.Option(
        ..., "--run-dir", "-r",
        help="Path to RAVE training run directory",
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
    streaming: bool = typer.Option(
        False, "--streaming",
        help="Enable streaming mode for realtime use",
    ),
):
    """Export a trained RAVE model for generation.

    Converts the training checkpoint to a TorchScript model
    that can be used for audio generation.
    """
    from . import rave_utils

    console.print("[bold blue]RAVE Export[/bold blue]")
    console.print()

    if not rave_utils.check_rave_available():
        console.print("[red]RAVE CLI not found.[/red]")
        console.print("[dim]Install with: pip install acids-rave[/dim]")
        raise typer.Exit(1)

    model_path = rave_utils.rave_export(
        run_path=run_dir,
        streaming=streaming,
    )

    if model_path:
        console.print()
        console.print("[bold green]Export complete![/bold green]")
        console.print(f"[dim]Model exported to: {model_path}[/dim]")
        console.print()
        console.print("[dim]Generate with: scronchler rave-generate -m {model_path}[/dim]")
    else:
        console.print("[red]Export failed[/red]")
        raise typer.Exit(1)


@app.command("rave-generate")
def rave_generate_cmd(
    model_path: Path = typer.Option(
        ..., "--model-path", "-m",
        help="Path to exported RAVE model (.ts file)",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    input_dir: Path = typer.Option(
        ..., "--input-dir", "-i",
        help="Directory with audio files to use as seeds",
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
    output_dir: Path = typer.Option(
        ..., "--output-dir", "-o",
        help="Directory to save generated audio",
    ),
    count: int = typer.Option(
        10, "--count", "-c",
        help="Number of samples to generate",
        min=1,
    ),
):
    """Generate new audio using a trained RAVE model.

    RAVE generates by encoding input audio and decoding with variations.
    The input files serve as "seeds" for generation.
    """
    from . import rave_utils

    console.print("[bold blue]RAVE Generation[/bold blue]")
    console.print()

    if not rave_utils.check_rave_available():
        console.print("[red]RAVE CLI not found.[/red]")
        console.print("[dim]Install with: pip install acids-rave[/dim]")
        raise typer.Exit(1)

    # Find input audio files
    input_files = list(input_dir.glob("*.wav"))
    if not input_files:
        input_files = list(input_dir.glob("*.mp3"))
    if not input_files:
        input_files = list(input_dir.glob("*.flac"))

    if not input_files:
        console.print(f"[red]No audio files found in {input_dir}[/red]")
        raise typer.Exit(1)

    console.print(f"[dim]Found {len(input_files)} input files[/dim]")
    console.print(f"[dim]Generating {count} samples...[/dim]")

    success = rave_utils.rave_generate(
        model_path=model_path,
        input_paths=input_files,
        output_dir=output_dir,
        count=count,
    )

    if success:
        console.print()
        console.print("[bold green]Generation complete![/bold green]")
        console.print(f"[dim]Samples saved to: {output_dir}[/dim]")
    else:
        console.print("[red]Generation failed[/red]")
        raise typer.Exit(1)


def main():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
