#!/usr/bin/env python3
"""Scropipe CLI - Audio pipeline orchestrator."""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from . import __version__
from .pipeline import Pipeline
from .stages import CollectStage, GenerateStage, PreprocessStage, SplitStage, TrainStage, TrainVocoderStage
from .utils.discovery import check_tools_available, find_all_tools

app = typer.Typer(
    name="scropipe",
    help="Audio pipeline orchestrator for Scrumpler and Scronchler",
    add_completion=False,
)
console = Console()


def version_callback(value: bool):
    if value:
        console.print(f"scropipe version {__version__}")
        raise typer.Exit()


def load_preset(preset_name: str) -> dict:
    """Load preset configuration from TOML file."""
    # Try to import tomllib (Python 3.11+) or tomli
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib

    # Look for preset in standard locations
    preset_paths = [
        Path.cwd() / "presets" / f"{preset_name}.toml",
        Path(__file__).parent.parent / "presets" / f"{preset_name}.toml",
        Path.home() / ".config" / "scropipe" / "presets" / f"{preset_name}.toml",
    ]

    for path in preset_paths:
        if path.exists():
            with open(path, "rb") as f:
                return tomllib.load(f)

    console.print(f"[red]Preset not found: {preset_name}[/red]")
    console.print("[dim]Searched in:[/dim]")
    for path in preset_paths:
        console.print(f"  - {path}")
    raise typer.Exit(1)


@app.callback()
def main_callback(
    version: bool = typer.Option(
        False, "--version", "-V",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit",
    ),
):
    """Scropipe - Audio pipeline orchestrator."""
    pass


@app.command()
def run(
    # Input sources
    input_files: list[Path] = typer.Option(
        [],
        "--input", "-i",
        help="Input audio file(s) to split (can be specified multiple times)",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    include_dirs: list[Path] = typer.Option(
        [],
        "--include", "-I",
        help="Directory of existing samples to include (can be specified multiple times)",
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
    output: Optional[Path] = typer.Option(
        None, "-o", "--output",
        help="Output directory (default: ./scropipe-output)",
    ),
    # Split options
    split: str = typer.Option(
        "transient", "--split", "-s",
        help="Split mode: grid, transient, or texture",
    ),
    # Synthesis options
    synthesize: bool = typer.Option(
        False, "--synthesize", "--synth",
        help="Run synthesis stages (preprocess, train, generate)",
    ),
    augment: bool = typer.Option(
        False, "--augment", "-a",
        help="Augment samples during preprocessing",
    ),
    max_duration: float = typer.Option(
        2.0, "--max-duration",
        help="Maximum sample duration in seconds",
    ),
    epochs: int = typer.Option(
        100, "--epochs", "-e",
        help="Training epochs",
    ),
    count: int = typer.Option(
        10, "--count", "-c",
        help="Number of samples to generate",
    ),
    train_vocoder: bool = typer.Option(
        False, "--train-vocoder",
        help="Train HiFi-GAN vocoder for higher quality output",
    ),
    vocoder_epochs: int = typer.Option(
        50, "--vocoder-epochs",
        help="Vocoder training epochs (only with --train-vocoder)",
    ),
    # Preset
    preset: Optional[str] = typer.Option(
        None, "--preset", "-p",
        help="Use a preset configuration",
    ),
    # Resume
    resume: bool = typer.Option(
        False, "--resume", "-r",
        help="Resume from existing pipeline state",
    ),
    # Split-specific options
    delta: float = typer.Option(
        0.07, "--delta",
        help="Transient detection sensitivity",
    ),
    min_length: float = typer.Option(
        0.05, "--min-length",
        help="Minimum segment length for transient mode",
    ),
    max_length: float = typer.Option(
        10.0, "--max-length",
        help="Maximum segment length for transient mode",
    ),
    chunk_length: Optional[float] = typer.Option(
        None, "--chunk-length",
        help="Chunk length for grid mode",
    ),
    bpm: Optional[float] = typer.Option(
        None, "--bpm",
        help="BPM for musical grid chopping",
    ),
    bars: int = typer.Option(
        4, "--bars",
        help="Number of bars per chunk when using --bpm",
    ),
):
    """Run the complete audio pipeline.

    Split audio files and/or include existing samples, then optionally
    preprocess, train, and generate new samples.

    Examples:

        # Split a single audio file
        scropipe run -i ~/recordings/party.wav --split transient

        # Split multiple files
        scropipe run -i party.wav -i other-recording.wav --split transient

        # Include existing samples without splitting
        scropipe run -I ~/samples/drums/ --synthesize

        # Mix: split files + include existing samples
        scropipe run -i party.wav -i field-recording.wav -I ~/samples/drums/ --synthesize

        # Full pipeline with custom settings
        scropipe run -i drums.wav -I ~/samples/synths/ --synthesize --epochs 200 --count 50
    """
    # Validate inputs
    if not input_files and not include_dirs:
        console.print("[red]Error: Must specify at least one --input or --include[/red]")
        console.print()
        console.print("Examples:")
        console.print("  scropipe run -i audio.wav")
        console.print("  scropipe run -I ~/samples/drums/")
        console.print("  scropipe run -i audio.wav -I ~/samples/drums/")
        raise typer.Exit(1)

    # Load preset if specified
    config = {}
    if preset:
        config = load_preset(preset)
        console.print(f"[dim]Using preset: {preset}[/dim]")

    # CLI options override preset
    final_split = config.get("split", {}).get("mode", split)
    final_synthesize = config.get("synthesize", synthesize)
    final_augment = config.get("preprocess", {}).get("augment", augment)
    final_max_duration = config.get("preprocess", {}).get("max_duration", max_duration)
    final_epochs = config.get("train", {}).get("epochs", epochs)
    final_count = config.get("generate", {}).get("count", count)

    # Check required tools
    required = []
    if input_files:
        required.append("scrumpler")
    if final_synthesize:
        required.append("scronchler")

    if required:
        missing = check_tools_available(*required)
        if missing:
            console.print("[red]Missing required tools:[/red]")
            for tool in missing:
                console.print(f"  - {tool}")
            console.print()
            console.print("[dim]Install them or set environment variables:[/dim]")
            console.print("  SCRUMPLER_PATH=/path/to/scrumpler")
            console.print("  SCRONCHLER_PATH=/path/to/scronchler")
            raise typer.Exit(1)

    # Build split kwargs
    split_kwargs = {
        "delta": delta,
        "min_length": min_length,
        "max_length": max_length,
    }
    if chunk_length:
        split_kwargs["chunk_length"] = chunk_length
    if bpm:
        split_kwargs["bpm"] = bpm
        split_kwargs["bars"] = bars

    # Run pipeline
    pipeline = Pipeline(
        input_files=list(input_files),
        include_dirs=list(include_dirs),
        output_dir=output,
        resume=resume,
    )

    success = pipeline.run(
        split_mode=final_split,
        synthesize=final_synthesize,
        augment=final_augment,
        max_duration=final_max_duration,
        epochs=final_epochs,
        count=final_count,
        train_vocoder=train_vocoder,
        vocoder_epochs=vocoder_epochs,
        **split_kwargs,
    )

    raise typer.Exit(0 if success else 1)


@app.command()
def split(
    input_file: Path = typer.Argument(
        ...,
        help="Input audio file to split",
        exists=True,
    ),
    output: Optional[Path] = typer.Option(
        None, "-o", "--output",
        help="Output directory",
    ),
    mode: str = typer.Option(
        "transient", "-m", "--mode",
        help="Split mode: grid, transient, or texture",
    ),
    delta: float = typer.Option(0.07, "--delta"),
    min_length: float = typer.Option(0.05, "--min-length"),
    max_length: float = typer.Option(10.0, "--max-length"),
    chunk_length: Optional[float] = typer.Option(None, "--chunk-length"),
    bpm: Optional[float] = typer.Option(None, "--bpm"),
    bars: int = typer.Option(4, "--bars"),
):
    """Split audio file into samples using scrumpler.

    Example:
        scropipe split ~/audio/drums.wav --mode transient -o ./samples
    """
    missing = check_tools_available("scrumpler")
    if missing:
        console.print("[red]scrumpler not found[/red]")
        raise typer.Exit(1)

    output_base = output or Path.cwd() / "scropipe-split"
    stage = SplitStage(output_base)

    kwargs = {
        "input_file": input_file,
        "mode": mode,
        "delta": delta,
        "min_length": min_length,
        "max_length": max_length,
    }
    if chunk_length:
        kwargs["chunk_length"] = chunk_length
    if bpm:
        kwargs["bpm"] = bpm
        kwargs["bars"] = bars

    result = stage.run(**kwargs)

    if result.success:
        console.print(f"[green]Success![/green] {result.message}")
        console.print(f"[dim]Output: {result.output_dir}[/dim]")
    else:
        console.print(f"[red]Failed:[/red] {result.message}")
        raise typer.Exit(1)


@app.command()
def collect(
    input_dirs: list[Path] = typer.Argument(
        ...,
        help="Directories containing WAV samples to collect",
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
    output: Optional[Path] = typer.Option(
        None, "-o", "--output",
        help="Output directory",
    ),
    symlink: bool = typer.Option(
        False, "--symlink",
        help="Create symlinks instead of copying files",
    ),
):
    """Collect samples from multiple directories into a single pool.

    Example:
        scropipe collect ./split-output ~/samples/drums -o ./pool
    """
    output_base = output or Path.cwd() / "scropipe-pool"
    stage = CollectStage(output_base)

    result = stage.run(
        split_dirs=[],
        include_dirs=list(input_dirs),
        symlink=symlink,
    )

    if result.success:
        console.print(f"[green]Success![/green] {result.message}")
        console.print(f"[dim]Output: {result.output_dir}[/dim]")
    else:
        console.print(f"[red]Failed:[/red] {result.message}")
        raise typer.Exit(1)


@app.command()
def synthesize(
    input_dir: Path = typer.Argument(
        ...,
        help="Directory containing WAV samples",
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
    output: Optional[Path] = typer.Option(
        None, "-o", "--output",
        help="Output directory",
    ),
    augment: bool = typer.Option(
        False, "-a", "--augment",
        help="Augment samples during preprocessing",
    ),
    max_duration: float = typer.Option(
        2.0, "--max-duration",
        help="Maximum sample duration in seconds",
    ),
    epochs: int = typer.Option(
        100, "-e", "--epochs",
        help="Training epochs",
    ),
    count: int = typer.Option(
        10, "-c", "--count",
        help="Number of samples to generate",
    ),
    train_vocoder: bool = typer.Option(
        False, "--train-vocoder",
        help="Train HiFi-GAN vocoder for higher quality output",
    ),
    vocoder_epochs: int = typer.Option(
        50, "--vocoder-epochs",
        help="Vocoder training epochs",
    ),
):
    """Run synthesis stages on existing samples.

    Preprocesses samples, trains VAE, and generates new variations.

    Example:
        scropipe synthesize ./samples --output ./ai-samples --count 50
    """
    missing = check_tools_available("scronchler")
    if missing:
        console.print("[red]scronchler not found[/red]")
        raise typer.Exit(1)

    output_base = output or Path.cwd() / "scropipe-synth"
    output_base.mkdir(parents=True, exist_ok=True)

    # Stage 1: Preprocess
    console.print("[bold]Preprocessing...[/bold]")
    preprocess = PreprocessStage(output_base)
    result = preprocess.run(
        input_dir=input_dir,
        augment=augment,
        max_duration=max_duration,
    )

    if not result.success:
        console.print(f"[red]Preprocess failed:[/red] {result.message}")
        raise typer.Exit(1)

    # Stage 2: Train
    console.print("[bold]Training...[/bold]")
    train = TrainStage(output_base)
    result = train.run(
        data_dir=preprocess.output_dir,
        epochs=epochs,
    )

    if not result.success:
        console.print(f"[red]Training failed:[/red] {result.message}")
        raise typer.Exit(1)

    model_path = train.output_dir / "model.pth"
    vocoder_path = None

    # Stage 3 (optional): Train vocoder
    if train_vocoder:
        console.print("[bold]Training vocoder...[/bold]")
        vocoder_train = TrainVocoderStage(output_base)
        result = vocoder_train.run(
            audio_dir=input_dir,
            spec_dir=preprocess.output_dir,
            epochs=vocoder_epochs,
        )

        if not result.success:
            console.print(f"[yellow]Vocoder training failed:[/yellow] {result.message}")
            console.print("[yellow]Falling back to Griffin-Lim[/yellow]")
        else:
            vocoder_path = vocoder_train.output_dir / "vocoder.pth"

    # Stage 4: Generate
    console.print("[bold]Generating...[/bold]")
    generate = GenerateStage(output_base)
    result = generate.run(
        model_path=model_path,
        count=count,
        vocoder_path=vocoder_path,
    )

    if not result.success:
        console.print(f"[red]Generation failed:[/red] {result.message}")
        raise typer.Exit(1)

    console.print()
    console.print(f"[green]Success![/green] Generated {count} AI samples")
    console.print(f"[dim]Output: {generate.output_dir}[/dim]")


@app.command()
def tools():
    """Show status of required tools."""
    tools_status = find_all_tools()

    console.print("[bold]Tool Status[/bold]")
    console.print()

    for name, path in tools_status.items():
        if path:
            console.print(f"  [green][/green] {name}: {path}")
        else:
            console.print(f"  [red][/red] {name}: not found")

    console.print()
    console.print("[dim]Set environment variables to override paths:[/dim]")
    console.print("  SCRUMPLER_PATH=/path/to/scrumpler")
    console.print("  SCRONCHLER_PATH=/path/to/scronchler")


def main():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
