#!/usr/bin/env python3
"""Scropipe CLI - Audio pipeline orchestrator."""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from . import __version__
from .pipeline import Pipeline
from .stages import (
    CollectStage, GenerateStage, PreprocessStage, SplitStage, TrainStage, TrainVocoderStage,
    RavePreprocessStage, RaveTrainStage, RaveExportStage, RaveGenerateStage,
)
from .utils.discovery import find_all_tools

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


def main():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
