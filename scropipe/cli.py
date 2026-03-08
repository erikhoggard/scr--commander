#!/usr/bin/env python3
"""Scropipe CLI - Audio pipeline orchestrator."""

import json
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from . import __version__
from .pipeline import Pipeline
from .stages import (
    CollectStage, GenerateStage, PreprocessStage, SplitStage, TrainStage, TrainVocoderStage,
    RavePreprocessStage, RaveTrainStage, RaveExportStage, RaveGenerateStage,
)
from .utils.discovery import find_all_tools


# Default model storage location
MODELS_DIR = Path.cwd() / "scropipe" / "models"


def get_models_dir() -> Path:
    """Get the models directory, creating it if needed."""
    models_dir = Path.cwd() / "scropipe" / "models"
    return models_dir


def resolve_model(model_ref: str) -> Optional[Path]:
    """Resolve a model reference to a path.

    Args:
        model_ref: Model name or path.

    Returns:
        Path to the model .ts file, or None if not found.
    """
    # If it's a direct path, use it
    model_path = Path(model_ref)
    if model_path.exists():
        if model_path.is_file() and model_path.suffix == ".ts":
            return model_path
        # If it's a directory, look for model.ts inside
        if model_path.is_dir():
            ts_file = model_path / "model.ts"
            if ts_file.exists():
                return ts_file
            # Search for any .ts file
            ts_files = list(model_path.glob("*.ts"))
            if ts_files:
                return ts_files[0]

    # Look in models directory
    models_dir = get_models_dir()
    model_dir = models_dir / model_ref
    if model_dir.exists():
        ts_file = model_dir / "model.ts"
        if ts_file.exists():
            return ts_file
        # Search for any .ts file
        ts_files = list(model_dir.glob("*.ts"))
        if ts_files:
            return ts_files[0]

    return None

app = typer.Typer(
    name="scropipe",
    help="Audio pipeline for splitting, collecting, and synthesizing samples",
    add_completion=False,
)
console = Console()


def version_callback(value: bool):
    if value:
        console.print(f"scropipe version {__version__}")
        raise typer.Exit()


def load_preset(preset_name: str) -> dict:
    """Load preset configuration from TOML file."""
    import tomllib

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
        help="Train HiFi-GAN vocoder for higher quality output (VAE only)",
    ),
    vocoder_epochs: int = typer.Option(
        50, "--vocoder-epochs",
        help="Vocoder training epochs (only with --train-vocoder)",
    ),
    model: str = typer.Option(
        "vae", "--model", "-m",
        help="Model type: 'vae' (fast, good for percussion) or 'rave' (slow, better for melodic)",
    ),
    rave_config: str = typer.Option(
        "v2", "--rave-config",
        help="RAVE config: v2, v2_small, discrete (only with --model rave)",
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
    # Generation options
    no_train: bool = typer.Option(
        False, "--no-train",
        help="Skip training, go straight to export + generate from existing checkpoint",
    ),
    seed_dir: Optional[Path] = typer.Option(
        None, "--seed-dir",
        help="Directory of audio files to use as generation input (default: training pool)",
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
    # Validate inputs (not needed when resuming — inputs are in pipeline.json)
    if not resume and not input_files and not include_dirs:
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

    # --no-train implies --synthesize (you want export+generate)
    if no_train:
        final_synthesize = True

    success = pipeline.run(
        split_mode=final_split,
        synthesize=final_synthesize,
        augment=final_augment,
        max_duration=final_max_duration,
        epochs=final_epochs,
        count=final_count,
        train_vocoder=train_vocoder,
        vocoder_epochs=vocoder_epochs,
        model=model,
        rave_config=rave_config,
        seed_dir=seed_dir,
        no_train=no_train,
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
    """Split audio file into samples.

    Example:
        scropipe split ~/audio/drums.wav --mode transient -o ./samples
    """
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
        help="Train HiFi-GAN vocoder for higher quality output (VAE only)",
    ),
    vocoder_epochs: int = typer.Option(
        50, "--vocoder-epochs",
        help="Vocoder training epochs",
    ),
    model: str = typer.Option(
        "vae", "--model", "-m",
        help="Model type: 'vae' (fast) or 'rave' (slow, better for melodic)",
    ),
    rave_config: str = typer.Option(
        "v2", "--rave-config",
        help="RAVE config: v2, v2_small, discrete (only with --model rave)",
    ),
):
    """Run synthesis stages on existing samples.

    Preprocesses samples, trains model, and generates new variations.

    Use --model rave for melodic/harmonic content like piano.

    Example:
        scropipe synthesize ./samples --output ./ai-samples --count 50
    """
    output_base = output or Path.cwd() / "scropipe-synth"
    output_base.mkdir(parents=True, exist_ok=True)

    if model.lower() == "rave":
        # RAVE pipeline (better for melodic content)
        console.print("[bold blue]Using RAVE model (high-quality melodic synthesis)[/bold blue]")
        console.print()

        # Stage 1: RAVE Preprocess
        console.print("[bold]RAVE Preprocessing...[/bold]")
        rave_preprocess = RavePreprocessStage(output_base)
        result = rave_preprocess.run(input_dir=input_dir)

        if not result.success:
            console.print(f"[red]RAVE preprocess failed:[/red] {result.message}")
            raise typer.Exit(1)

        # Stage 2: RAVE Train
        console.print("[bold]RAVE Training (this takes several hours)...[/bold]")
        rave_train = RaveTrainStage(output_base)
        result = rave_train.run(
            data_dir=rave_preprocess.output_dir,
            config=rave_config,
            epochs=epochs if epochs != 100 else None,  # Use default if not specified
        )

        if not result.success:
            console.print(f"[red]RAVE training failed:[/red] {result.message}")
            raise typer.Exit(1)

        # Stage 3: RAVE Export
        console.print("[bold]Exporting RAVE model...[/bold]")
        rave_export = RaveExportStage(output_base)
        result = rave_export.run(run_dir=rave_train.output_dir)

        if not result.success:
            console.print(f"[red]RAVE export failed:[/red] {result.message}")
            raise typer.Exit(1)

        # Find exported model
        model_path = None
        for ts_file in rave_train.output_dir.glob("**/*.ts"):
            model_path = ts_file
            break

        if not model_path:
            console.print("[red]Could not find exported RAVE model[/red]")
            raise typer.Exit(1)

        # Stage 4: RAVE Generate
        console.print("[bold]Generating with RAVE...[/bold]")
        rave_generate = RaveGenerateStage(output_base)
        result = rave_generate.run(
            model_path=model_path,
            input_dir=input_dir,
            count=count,
        )

        if not result.success:
            console.print(f"[red]RAVE generation failed:[/red] {result.message}")
            raise typer.Exit(1)

        console.print()
        console.print(f"[green]Success![/green] Generated {count} RAVE samples")
        console.print(f"[dim]Output: {rave_generate.output_dir}[/dim]")

    else:
        # VAE pipeline (faster, good for percussion)
        console.print("[bold blue]Using VAE model[/bold blue]")
        console.print()

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


@app.command("train")
def train_cmd(
    output: Path = typer.Option(
        ..., "-o", "--output",
        help="Pipeline output directory (must contain pipeline.json)",
    ),
    epochs: Optional[int] = typer.Option(
        None, "-e", "--epochs",
        help="Max training steps (default: RAVE default of 6M)",
    ),
):
    """Resume RAVE training from the last checkpoint.

    Example:
        scropipe train -o ./scropipe-output
        scropipe train -o ./scropipe-output --epochs 5000
    """
    from .pipeline import Pipeline, StageStatus

    output = Path(output).resolve()
    state_file = output / "pipeline.json"
    if not state_file.exists():
        console.print(f"[red]No pipeline.json found in {output}[/red]")
        console.print("Run 'scropipe run' first to create a pipeline.")
        raise typer.Exit(1)

    pipeline = Pipeline(output_dir=output, resume=True)
    state = pipeline.state

    # Must have preprocessed data
    preprocess_output = pipeline._get_stage_output("rave_preprocess")
    if not preprocess_output:
        console.print("[red]No RAVE preprocessed data found. Run the full pipeline first.[/red]")
        raise typer.Exit(1)

    # Reset train stage so it actually runs
    if "rave_train" in state.stages:
        state.stages["rave_train"].status = StageStatus.PENDING

    # Find checkpoint to resume from
    ckpt = pipeline._find_rave_checkpoint()
    if ckpt:
        console.print(f"[bold]Resuming RAVE training from checkpoint[/bold]")
        console.print(f"[dim]  Checkpoint: {ckpt}[/dim]")
    else:
        console.print("[bold]Starting RAVE training from scratch[/bold]")

    rave_config = state.config.get("rave_config", "v2")

    pipeline._update_stage_state("rave_train", StageStatus.RUNNING)

    result = pipeline.stages["rave_train"].run(
        data_dir=preprocess_output,
        config=rave_config,
        epochs=epochs,
        ckpt=ckpt,
    )

    if not result.success:
        pipeline._update_stage_state("rave_train", StageStatus.FAILED, result)
        console.print(f"[red]Training failed: {result.message}[/red]")
        raise typer.Exit(1)

    pipeline._update_stage_state("rave_train", StageStatus.COMPLETED, result)
    console.print(f"[green]Training complete![/green]")
    console.print(f"[dim]Run 'scropipe generate -o {output}' to generate audio.[/dim]")


@app.command("generate")
def generate_cmd(
    output: Path = typer.Option(
        ..., "-o", "--output",
        help="Pipeline output directory (must contain pipeline.json)",
    ),
    count: int = typer.Option(
        10, "-c", "--count",
        help="Number of samples to generate",
    ),
    seed_dir: Optional[Path] = typer.Option(
        None, "--seed-dir",
        help="Directory of audio to use as input (default: training pool)",
    ),
):
    """Generate audio from a trained RAVE model.

    Exports the model if needed, then generates samples.

    Example:
        scropipe generate -o ./scropipe-output --count 20
        scropipe generate -o ./scropipe-output --count 20 --seed-dir ./other-audio/
    """
    from .pipeline import Pipeline, PipelineState, StageStatus, StageResult

    output = Path(output).resolve()
    state_file = output / "pipeline.json"
    if not state_file.exists():
        console.print(f"[red]No pipeline.json found in {output}[/red]")
        console.print("Run 'scropipe run' first to create a pipeline.")
        raise typer.Exit(1)

    pipeline = Pipeline(output_dir=output, resume=True)
    state = pipeline.state

    # Find the trained model's run dir
    ckpt = pipeline._find_rave_checkpoint()
    if not ckpt:
        console.print("[red]No RAVE checkpoint found. Train first with 'scropipe train'.[/red]")
        raise typer.Exit(1)

    # Find run dir (directory containing config.gin)
    run_dir = ckpt.parent
    while run_dir != run_dir.parent:
        if (run_dir / "config.gin").exists():
            break
        run_dir = run_dir.parent
    else:
        console.print("[red]Could not find RAVE run directory (no config.gin)[/red]")
        raise typer.Exit(1)

    # Mark train as completed with correct run dir
    pipeline._update_stage_state(
        "rave_train",
        StageStatus.COMPLETED,
        StageResult(success=True, output_dir=run_dir, message="Using existing checkpoint"),
    )

    # Export if needed
    existing_ts = list(run_dir.glob("**/*.ts"))
    if existing_ts:
        model_path = existing_ts[0]
        console.print(f"[dim]Model already exported: {model_path.name}[/dim]")
    else:
        console.print("[bold]Exporting RAVE model...[/bold]")
        pipeline._update_stage_state("rave_export", StageStatus.RUNNING)
        result = pipeline.stages["rave_export"].run(run_dir=run_dir)
        if not result.success:
            pipeline._update_stage_state("rave_export", StageStatus.FAILED, result)
            console.print(f"[red]Export failed: {result.message}[/red]")
            raise typer.Exit(1)
        pipeline._update_stage_state("rave_export", StageStatus.COMPLETED, result)
        ts_files = list(run_dir.glob("**/*.ts"))
        if not ts_files:
            console.print("[red]Export succeeded but no .ts file found[/red]")
            raise typer.Exit(1)
        model_path = ts_files[0]

    # Determine seed audio
    if seed_dir:
        gen_input = Path(seed_dir).resolve()
        console.print(f"[dim]Seed audio: {gen_input}[/dim]")
    else:
        gen_input = pipeline._get_stage_output("collect")
        if not gen_input:
            console.print("[red]No pool directory found and no --seed-dir specified[/red]")
            raise typer.Exit(1)

    # Generate
    console.print(f"[bold]Generating {count} samples...[/bold]")
    pipeline._update_stage_state("rave_generate", StageStatus.RUNNING)
    result = pipeline.stages["rave_generate"].run(
        model_path=model_path,
        input_dir=gen_input,
        count=count,
    )

    if not result.success:
        pipeline._update_stage_state("rave_generate", StageStatus.FAILED, result)
        console.print(f"[red]Generation failed: {result.message}[/red]")
        raise typer.Exit(1)

    pipeline._update_stage_state("rave_generate", StageStatus.COMPLETED, result)
    console.print(f"[green]Done![/green] {result.message}")
    console.print(f"[dim]Output: {pipeline.stages['rave_generate'].output_dir}[/dim]")


@app.command()
def tools():
    """Show status of external tools.

    Note: scrumpler and scronchler are now built-in to scropipe.
    Only RAVE is an external tool (for --model rave).
    """
    tools_status = find_all_tools()

    console.print("[bold]External Tool Status[/bold]")
    console.print()

    # Built-in tools
    console.print("  [green]✓[/green] splitter: [dim]built-in[/dim]")
    console.print("  [green]✓[/green] synth (VAE): [dim]built-in[/dim]")

    # External tools
    for name, path in tools_status.items():
        if path:
            console.print(f"  [green]✓[/green] {name}: {path} [dim](optional)[/dim]")
        else:
            console.print(f"  [yellow]○[/yellow] {name}: not found [dim](only needed for --model rave)[/dim]")

    console.print()
    console.print("[bold]Install Options[/bold]")
    console.print("  pip install scropipe       - splitting only (lightweight)")
    console.print("  pip install scropipe[ml]   - full ML synthesis")
    console.print()
    console.print("[dim]Set RAVE_PATH to override rave location (only needed for --model rave)[/dim]")


@app.command()
def train(
    # Input sources (files and/or directories)
    inputs: list[Path] = typer.Argument(
        ...,
        help="Input sample directories and/or audio files to split",
    ),
    # Model naming
    name: str = typer.Option(
        ..., "--name", "-n",
        help="Name for the trained model (required)",
    ),
    # Split options (required if any input is a file)
    split_mode: Optional[str] = typer.Option(
        None, "--split", "-s",
        help="Split mode for audio files: grid, transient, or texture",
    ),
    # Transient split options
    delta: float = typer.Option(
        0.07, "--delta",
        help="Transient detection sensitivity (higher = more splits)",
    ),
    min_length: float = typer.Option(
        0.05, "--min-length",
        help="Minimum segment length in seconds",
    ),
    max_length: float = typer.Option(
        10.0, "--max-length",
        help="Maximum segment length in seconds",
    ),
    # Grid split options
    chunk_length: Optional[float] = typer.Option(
        None, "--chunk-length",
        help="Chunk length in seconds for grid mode",
    ),
    bpm: Optional[float] = typer.Option(
        None, "--bpm",
        help="BPM for musical grid chopping",
    ),
    bars: int = typer.Option(
        4, "--bars",
        help="Number of bars per chunk when using --bpm",
    ),
    # Texture split options
    min_duration: float = typer.Option(
        1.0, "--min-duration",
        help="Minimum texture duration in seconds",
    ),
    max_duration: float = typer.Option(
        30.0, "--max-duration",
        help="Maximum texture duration in seconds",
    ),
    rms_threshold: float = typer.Option(
        0.1, "--rms-threshold",
        help="RMS threshold for texture detection",
    ),
    stability_threshold: float = typer.Option(
        0.15, "--stability-threshold",
        help="Spectral stability threshold for texture detection",
    ),
    # RAVE training options
    epochs: Optional[int] = typer.Option(
        None, "--epochs", "-e",
        help="Max training steps (default: unlimited, stop with Ctrl+C)",
    ),
    config: str = typer.Option(
        "v2", "--config", "-c",
        help="RAVE config: v2, v2_small, discrete",
    ),
    val_every: int = typer.Option(
        500, "--val-every",
        help="Save checkpoint every N steps",
    ),
):
    """Train a RAVE model on samples.

    Creates a reusable model in ./scropipe/models/{name}/.

    Input can be:
    - Directories: treated as existing sample collections
    - WAV files: treated as long-form audio, requires --split

    Examples:

        # Train on existing samples
        scropipe train ./my-samples --name drum-kit

        # Train on long-form audio (must specify split mode)
        scropipe train ./recording.wav --name breaks --split transient

        # Grid chop by BPM
        scropipe train ./stems/*.wav --name piano --split grid --bpm 120 --bars 2

        # Mix samples and long-form audio
        scropipe train ./samples ./recording.wav --name hybrid --split transient
    """
    # Separate inputs into directories (samples) and files (need splitting)
    sample_dirs: list[Path] = []
    audio_files: list[Path] = []

    for inp in inputs:
        if not inp.exists():
            console.print(f"[red]Error: Input not found: {inp}[/red]")
            raise typer.Exit(1)
        if inp.is_dir():
            sample_dirs.append(inp)
        else:
            audio_files.append(inp)

    # Validate: if there are audio files, --split is required
    if audio_files and not split_mode:
        console.print("[red]Error: WAV file inputs require --split flag[/red]")
        console.print()
        console.print("Files that need splitting:")
        for f in audio_files:
            console.print(f"  {f}")
        console.print()
        console.print("Use one of:")
        console.print("  --split transient --delta 0.1")
        console.print("  --split grid --bpm 120 --bars 2")
        console.print("  --split texture --min-duration 2.0")
        raise typer.Exit(1)

    # Validate split mode if provided
    if split_mode and split_mode not in ("grid", "transient", "texture"):
        console.print(f"[red]Error: Invalid split mode: {split_mode}[/red]")
        console.print("Valid modes: grid, transient, texture")
        raise typer.Exit(1)

    # Set up output directories
    models_dir = get_models_dir()
    model_output_dir = models_dir / name

    if model_output_dir.exists():
        console.print(f"[yellow]Warning: Model '{name}' already exists at {model_output_dir}[/yellow]")
        console.print("[yellow]Training will overwrite the existing model.[/yellow]")

    # Create temp directory for intermediate files
    with tempfile.TemporaryDirectory(prefix="scropipe_train_") as temp_dir:
        temp_path = Path(temp_dir)

        # Step 1: Split audio files if any
        split_output_dirs: list[Path] = []
        if audio_files:
            console.print(f"[bold]Splitting {len(audio_files)} audio file(s)...[/bold]")

            # SplitStage outputs to output_base / "splits" / source_name
            split_stage = SplitStage(temp_path / "00-splits")

            for audio_file in audio_files:
                console.print(f"  Splitting {audio_file.name}...")

                # Build split kwargs based on mode
                split_kwargs = {
                    "input_file": audio_file,
                    "mode": split_mode,
                    "delta": delta,
                    "min_length": min_length,
                    "max_length": max_length,
                    "min_duration": min_duration,
                    "max_duration": max_duration,
                    "rms_threshold": rms_threshold,
                    "stability_threshold": stability_threshold,
                }
                if chunk_length:
                    split_kwargs["chunk_length"] = chunk_length
                if bpm:
                    split_kwargs["bpm"] = bpm
                    split_kwargs["bars"] = bars

                result = split_stage.run(**split_kwargs)

                if not result.success:
                    console.print(f"[red]Failed to split {audio_file.name}: {result.message}[/red]")
                    raise typer.Exit(1)

                if result.output_dir:
                    split_output_dirs.append(result.output_dir)
                    sample_count = result.details.get("sample_count", 0) if result.details else 0
                    console.print(f"    [green]✓[/green] {sample_count} samples")

        # Step 2: Collect all samples into pool
        total_sources = len(sample_dirs) + len(split_output_dirs)
        console.print(f"[bold]Collecting samples from {total_sources} source(s)...[/bold]")

        # CollectStage outputs to output_base / "01-pool"
        collect_stage = CollectStage(temp_path)

        result = collect_stage.run(
            split_dirs=split_output_dirs,
            include_dirs=sample_dirs,
            symlink=False,
        )

        if not result.success:
            console.print(f"[red]Failed to collect samples: {result.message}[/red]")
            raise typer.Exit(1)

        pool_dir = result.output_dir
        total_samples = len(list(pool_dir.glob("*.wav")))
        console.print(f"  [green]✓[/green] {total_samples} total samples in pool")

        if total_samples == 0:
            console.print("[red]Error: No samples to train on[/red]")
            raise typer.Exit(1)

        # Step 3: RAVE Preprocess
        console.print("[bold]Preprocessing for RAVE...[/bold]")

        # RavePreprocessStage outputs to output_base / "02-rave-data"
        rave_preprocess = RavePreprocessStage(temp_path)

        result = rave_preprocess.run(input_dir=pool_dir)

        if not result.success:
            console.print(f"[red]RAVE preprocess failed: {result.message}[/red]")
            raise typer.Exit(1)

        rave_data_dir = result.output_dir
        console.print("  [green]✓[/green] RAVE dataset created")

        # Step 4: RAVE Train
        console.print("[bold]Training RAVE model...[/bold]")
        if epochs:
            console.print(f"  [dim]Max steps: {epochs}[/dim]")
        else:
            console.print("  [dim]Training until interrupted (Ctrl+C to stop)[/dim]")
        console.print(f"  [dim]Checkpoints saved every {val_every} steps[/dim]")
        console.print()

        # RaveTrainStage outputs to output_base / "03-rave-model"
        rave_train = RaveTrainStage(temp_path)

        result = rave_train.run(
            data_dir=rave_data_dir,
            name="model",
            config=config,
            epochs=epochs,
            val_every=val_every,
        )

        if not result.success:
            console.print(f"[red]RAVE training failed: {result.message}[/red]")
            raise typer.Exit(1)

        rave_model_dir = result.output_dir
        console.print()
        console.print("  [green]✓[/green] Training complete")

        # Step 5: RAVE Export
        console.print("[bold]Exporting model...[/bold]")
        rave_export = RaveExportStage(temp_path)

        result = rave_export.run(run_dir=rave_model_dir)

        if not result.success:
            console.print(f"[red]RAVE export failed: {result.message}[/red]")
            raise typer.Exit(1)

        # Find the exported .ts file
        ts_files = list(rave_model_dir.glob("**/*.ts"))
        if not ts_files:
            console.print("[red]Error: No exported model found[/red]")
            raise typer.Exit(1)

        exported_model = ts_files[0]
        console.print(f"  [green]✓[/green] Model exported")

        # Step 6: Copy to final location
        console.print(f"[bold]Saving model to {model_output_dir}...[/bold]")
        model_output_dir.mkdir(parents=True, exist_ok=True)

        # Copy the model file
        final_model_path = model_output_dir / "model.ts"
        shutil.copy2(exported_model, final_model_path)

        # Copy checkpoints if they exist
        checkpoint_dirs = list(rave_model_dir.glob("**/checkpoints"))
        if checkpoint_dirs:
            checkpoints_output = model_output_dir / "checkpoints"
            if checkpoints_output.exists():
                shutil.rmtree(checkpoints_output)
            shutil.copytree(checkpoint_dirs[0], checkpoints_output)

        # Save metadata
        metadata = {
            "name": name,
            "created": datetime.now().isoformat(),
            "config": config,
            "epochs": epochs,
            "val_every": val_every,
            "sources": {
                "sample_dirs": [str(d) for d in sample_dirs],
                "audio_files": [str(f) for f in audio_files],
            },
            "total_samples": total_samples,
        }
        if split_mode:
            metadata["split"] = {
                "mode": split_mode,
                "delta": delta,
                "min_length": min_length,
                "max_length": max_length,
            }

        metadata_path = model_output_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        console.print(f"  [green]✓[/green] Model saved")

    console.print()
    console.print(f"[green bold]Success![/green bold] Model '{name}' trained and saved.")
    console.print(f"[dim]Location: {model_output_dir}[/dim]")
    console.print()
    console.print("Generate samples with:")
    console.print(f"  scropipe generate --model {name} --input ./samples --output ./generated")


@app.command()
def generate(
    # Model selection
    model: str = typer.Option(
        ..., "--model", "-m",
        help="Model name or path to .ts file",
    ),
    # Input sources
    inputs: list[Path] = typer.Option(
        ..., "--input", "-i",
        help="Input sample directories or files (can specify multiple)",
    ),
    # Output
    output: Path = typer.Option(
        ..., "--output", "-o",
        help="Output directory for generated samples",
    ),
    # Generation options
    count: int = typer.Option(
        1, "--count", "-n",
        help="Number of variations per input sample",
    ),
):
    """Generate samples using a trained RAVE model.

    Examples:

        # Generate from a named model
        scropipe generate --model drum-kit --input ./seeds --output ./generated

        # Multiple input sources
        scropipe generate --model ambient -i ./folder1 -i ./folder2 -o ./output

        # Generate multiple variations per input
        scropipe generate --model synth --input ./seeds --output ./variations --count 5

        # Use model by path
        scropipe generate --model ./my-model.ts --input ./seeds --output ./generated
    """
    # Resolve model path
    model_path = resolve_model(model)
    if not model_path:
        console.print(f"[red]Error: Model not found: {model}[/red]")
        console.print()
        console.print("Searched in:")
        console.print(f"  - {model} (as path)")
        console.print(f"  - {get_models_dir() / model}/model.ts")
        console.print()
        console.print("List available models with: scropipe models")
        raise typer.Exit(1)

    console.print(f"[dim]Using model: {model_path}[/dim]")

    # Validate inputs
    all_input_files: list[Path] = []
    for inp in inputs:
        if not inp.exists():
            console.print(f"[red]Error: Input not found: {inp}[/red]")
            raise typer.Exit(1)
        if inp.is_dir():
            wav_files = list(inp.glob("*.wav"))
            all_input_files.extend(wav_files)
        else:
            all_input_files.append(inp)

    if not all_input_files:
        console.print("[red]Error: No input WAV files found[/red]")
        raise typer.Exit(1)

    console.print(f"Found {len(all_input_files)} input sample(s)")

    # Create output directory
    output.mkdir(parents=True, exist_ok=True)

    # Load model and generate
    console.print("[bold]Generating samples...[/bold]")

    try:
        import torch
        import torchaudio

        # Load model
        rave_model = torch.jit.load(str(model_path))

        # Detect GPU
        device = torch.device("cpu")
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            rave_model = rave_model.to(device)
            console.print(f"  [dim]Using GPU: {torch.cuda.get_device_name(0)}[/dim]")

        generated_count = 0
        for input_file in all_input_files:
            try:
                # Load audio
                x, sr = torchaudio.load(str(input_file))

                # Resample if needed
                if sr != rave_model.sr:
                    x = torchaudio.functional.resample(x, sr, rave_model.sr)

                x = x.to(device)

                # Generate variations
                for i in range(count):
                    with torch.no_grad():
                        out = rave_model.forward(x[None])

                    # Build output filename
                    if count > 1:
                        out_name = f"{input_file.stem}_var{i+1:02d}.wav"
                    else:
                        out_name = f"{input_file.stem}_gen.wav"

                    out_path = output / out_name
                    torchaudio.save(str(out_path), out[0].cpu(), sample_rate=rave_model.sr)
                    generated_count += 1

            except Exception as e:
                console.print(f"  [yellow]Warning: Failed to process {input_file.name}: {e}[/yellow]")
                continue

        console.print()
        console.print(f"[green bold]Success![/green bold] Generated {generated_count} sample(s)")
        console.print(f"[dim]Output: {output}[/dim]")

    except ImportError as e:
        console.print(f"[red]Error: ML dependencies not installed[/red]")
        console.print(f"[dim]{e}[/dim]")
        console.print()
        console.print("Install with: pip install scropipe[ml]")
        raise typer.Exit(1)


@app.command()
def models(
    # Info subcommand
    info: Optional[str] = typer.Option(
        None, "--info",
        help="Show detailed info for a specific model",
    ),
):
    """List and manage trained RAVE models.

    Models are stored in ./scropipe/models/

    Examples:

        # List all models
        scropipe models

        # Show info for a specific model
        scropipe models --info drum-kit
    """
    models_dir = get_models_dir()

    if info:
        # Show info for specific model
        model_dir = models_dir / info
        metadata_path = model_dir / "metadata.json"

        if not model_dir.exists():
            console.print(f"[red]Error: Model not found: {info}[/red]")
            raise typer.Exit(1)

        console.print(f"[bold]Model: {info}[/bold]")
        console.print(f"[dim]Location: {model_dir}[/dim]")
        console.print()

        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)

            console.print(f"Created: {metadata.get('created', 'unknown')}")
            console.print(f"Config: {metadata.get('config', 'unknown')}")
            console.print(f"Training samples: {metadata.get('total_samples', 'unknown')}")

            if metadata.get("epochs"):
                console.print(f"Max epochs: {metadata['epochs']}")

            if metadata.get("split"):
                split_info = metadata["split"]
                console.print(f"Split mode: {split_info.get('mode', 'unknown')}")

            sources = metadata.get("sources", {})
            if sources.get("sample_dirs"):
                console.print()
                console.print("Sample directories:")
                for d in sources["sample_dirs"]:
                    console.print(f"  - {d}")
            if sources.get("audio_files"):
                console.print()
                console.print("Audio files (split):")
                for f in sources["audio_files"]:
                    console.print(f"  - {f}")
        else:
            console.print("[yellow]No metadata found[/yellow]")

        # Check for model file
        model_file = model_dir / "model.ts"
        if model_file.exists():
            size_mb = model_file.stat().st_size / (1024 * 1024)
            console.print()
            console.print(f"Model file: {size_mb:.1f} MB")

        # Check for checkpoints
        checkpoints_dir = model_dir / "checkpoints"
        if checkpoints_dir.exists():
            ckpt_files = list(checkpoints_dir.glob("*.ckpt"))
            console.print(f"Checkpoints: {len(ckpt_files)}")

    else:
        # List all models
        if not models_dir.exists():
            console.print("[dim]No models found.[/dim]")
            console.print()
            console.print("Train a model with:")
            console.print("  scropipe train ./samples --name my-model")
            return

        model_dirs = [d for d in models_dir.iterdir() if d.is_dir()]

        if not model_dirs:
            console.print("[dim]No models found.[/dim]")
            console.print()
            console.print("Train a model with:")
            console.print("  scropipe train ./samples --name my-model")
            return

        table = Table(title="RAVE Models")
        table.add_column("Name", style="cyan")
        table.add_column("Config")
        table.add_column("Samples")
        table.add_column("Created")

        for model_dir in sorted(model_dirs):
            metadata_path = model_dir / "metadata.json"

            config = "-"
            samples = "-"
            created = "-"

            if metadata_path.exists():
                with open(metadata_path) as f:
                    metadata = json.load(f)
                config = metadata.get("config", "-")
                samples = str(metadata.get("total_samples", "-"))
                created_str = metadata.get("created", "")
                if created_str:
                    try:
                        dt = datetime.fromisoformat(created_str)
                        created = dt.strftime("%Y-%m-%d")
                    except ValueError:
                        created = created_str[:10]

            table.add_row(model_dir.name, config, samples, created)

        console.print(table)
        console.print()
        console.print("[dim]Use --info <name> for details[/dim]")


def main():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
