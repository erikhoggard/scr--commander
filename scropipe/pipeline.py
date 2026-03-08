"""Pipeline orchestrator with state management."""

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .stages import (
    CollectStage,
    GenerateStage,
    PreprocessStage,
    SplitStage,
    StageResult,
    TrainStage,
    TrainVocoderStage,
    RavePreprocessStage,
    RaveTrainStage,
    RaveExportStage,
    RaveGenerateStage,
)

console = Console()


class StageStatus(str, Enum):
    """Status of a pipeline stage."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class StageState:
    """State of a single stage."""

    name: str
    status: StageStatus = StageStatus.PENDING
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    output_dir: Optional[str] = None
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineState:
    """Complete pipeline state."""

    input_files: list[str]
    include_dirs: list[str]
    output_base: str
    created_at: str
    updated_at: str
    stages: dict[str, StageState] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    # Track individual split results
    split_outputs: list[str] = field(default_factory=list)

    def save(self, path: Path):
        """Save state to JSON file."""
        data = {
            "input_files": self.input_files,
            "include_dirs": self.include_dirs,
            "output_base": self.output_base,
            "created_at": self.created_at,
            "updated_at": datetime.now().isoformat(),
            "config": self.config,
            "split_outputs": self.split_outputs,
            "stages": {
                name: {
                    "name": state.name,
                    "status": state.status.value,
                    "started_at": state.started_at,
                    "completed_at": state.completed_at,
                    "output_dir": state.output_dir,
                    "message": state.message,
                    "details": state.details,
                }
                for name, state in self.stages.items()
            },
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "PipelineState":
        """Load state from JSON file."""
        with open(path) as f:
            data = json.load(f)

        stages = {}
        for name, state_data in data.get("stages", {}).items():
            stages[name] = StageState(
                name=state_data["name"],
                status=StageStatus(state_data["status"]),
                started_at=state_data.get("started_at"),
                completed_at=state_data.get("completed_at"),
                output_dir=state_data.get("output_dir"),
                message=state_data.get("message", ""),
                details=state_data.get("details", {}),
            )

        # Handle legacy single input_file format
        input_files = data.get("input_files", [])
        if not input_files and "input_file" in data:
            input_files = [data["input_file"]]

        return cls(
            input_files=input_files,
            include_dirs=data.get("include_dirs", []),
            output_base=data["output_base"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            stages=stages,
            config=data.get("config", {}),
            split_outputs=data.get("split_outputs", []),
        )


class Pipeline:
    """Orchestrates the audio processing pipeline."""

    STAGE_ORDER = ["split", "collect", "preprocess", "train", "train_vocoder", "generate"]
    RAVE_STAGE_ORDER = ["split", "collect", "rave_preprocess", "rave_train", "rave_export", "rave_generate"]

    def __init__(
        self,
        input_files: Optional[list[Path]] = None,
        include_dirs: Optional[list[Path]] = None,
        output_dir: Optional[Path] = None,
        resume: bool = False,
    ):
        """Initialize the pipeline.

        Args:
            input_files: List of input audio files to split.
            include_dirs: List of directories with existing samples to include.
            output_dir: Output directory (default: ./scropipe-output).
            resume: Whether to resume from existing state.
        """
        self.input_files = [Path(f).resolve() for f in (input_files or [])]
        self.include_dirs = [Path(d).resolve() for d in (include_dirs or [])]
        self.resume = resume

        if output_dir:
            self.output_base = Path(output_dir).resolve()
        else:
            self.output_base = Path.cwd() / "scropipe-output"

        self.state_file = self.output_base / "pipeline.json"

        if resume and self.state_file.exists():
            self.state = PipelineState.load(self.state_file)
            console.print(f"[dim]Resuming pipeline from {self.state_file}[/dim]")
            # Restore inputs from saved state if not provided on CLI
            if not self.input_files and self.state.input_files:
                self.input_files = [Path(f) for f in self.state.input_files]
            if not self.include_dirs and self.state.include_dirs:
                self.include_dirs = [Path(d) for d in self.state.include_dirs]
        else:
            self.state = PipelineState(
                input_files=[str(f) for f in self.input_files],
                include_dirs=[str(d) for d in self.include_dirs],
                output_base=str(self.output_base),
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat(),
            )

        # Initialize stages
        self._init_stages()

    def _init_stages(self):
        """Initialize stage instances."""
        self.stages = {
            "split": SplitStage(self.output_base),
            "collect": CollectStage(self.output_base),
            "preprocess": PreprocessStage(self.output_base),
            "train": TrainStage(self.output_base),
            "train_vocoder": TrainVocoderStage(self.output_base),
            "generate": GenerateStage(self.output_base),
            "rave_preprocess": RavePreprocessStage(self.output_base),
            "rave_train": RaveTrainStage(self.output_base),
            "rave_export": RaveExportStage(self.output_base),
            "rave_generate": RaveGenerateStage(self.output_base),
        }

    def _update_stage_state(
        self,
        stage_name: str,
        status: StageStatus,
        result: Optional[StageResult] = None,
    ):
        """Update the state of a stage."""
        now = datetime.now().isoformat()

        if stage_name not in self.state.stages:
            self.state.stages[stage_name] = StageState(name=stage_name)

        state = self.state.stages[stage_name]
        state.status = status

        if status == StageStatus.RUNNING:
            state.started_at = now
        elif status in (StageStatus.COMPLETED, StageStatus.FAILED):
            state.completed_at = now

        if result:
            state.message = result.message
            state.details = result.details
            if result.output_dir:
                state.output_dir = str(result.output_dir)

        self.state.save(self.state_file)

    def _should_run_stage(self, stage_name: str) -> bool:
        """Check if a stage should run (not already completed)."""
        if stage_name not in self.state.stages:
            return True
        return self.state.stages[stage_name].status != StageStatus.COMPLETED

    def _get_stage_output(self, stage_name: str) -> Optional[Path]:
        """Get the output directory of a completed stage."""
        if stage_name in self.state.stages:
            state = self.state.stages[stage_name]
            if state.status == StageStatus.COMPLETED and state.output_dir:
                return Path(state.output_dir)
        return None

    def run(
        self,
        split_mode: str = "transient",
        synthesize: bool = False,
        augment: bool = False,
        max_duration: float = 2.0,
        epochs: int = 100,
        count: int = 10,
        train_vocoder: bool = False,
        vocoder_epochs: int = 50,
        model: str = "vae",
        rave_config: str = "v2",
        seed_dir: Optional[Path] = None,
        no_train: bool = False,
        **split_kwargs,
    ) -> bool:
        """Run the complete pipeline.

        Args:
            split_mode: Split mode (grid, transient, texture).
            synthesize: Whether to run synthesis stages.
            augment: Whether to augment samples during preprocessing.
            max_duration: Maximum sample duration for preprocessing.
            epochs: Training epochs.
            count: Number of samples to generate.
            train_vocoder: Whether to train HiFi-GAN vocoder (VAE only).
            vocoder_epochs: Vocoder training epochs.
            model: Model type ('vae' or 'rave').
            rave_config: RAVE config (v2, v2_small, etc.).
            seed_dir: Directory of audio to use as generation input (default: pool).
            **split_kwargs: Additional arguments for split stage.

        Returns:
            True if pipeline completed successfully.
        """
        # Validate inputs
        if not self.input_files and not self.include_dirs:
            console.print("[red]No input files or include directories specified[/red]")
            return False

        self.output_base.mkdir(parents=True, exist_ok=True)

        # On resume, preserve the model type from the saved config
        # (avoids requiring --model rave on every resume command)
        if self.resume and self.state.config.get("model"):
            model = self.state.config["model"]
            rave_config = self.state.config.get("rave_config", rave_config)

        # Store config
        self.state.config = {
            "split_mode": split_mode,
            "synthesize": synthesize,
            "augment": augment,
            "max_duration": max_duration,
            "epochs": epochs,
            "count": count,
            "train_vocoder": train_vocoder,
            "vocoder_epochs": vocoder_epochs,
            "model": model,
            "rave_config": rave_config,
            **split_kwargs,
        }

        # Check model type
        use_rave = model.lower() == "rave"
        if use_rave:
            console.print("[bold blue]Using RAVE model (high-quality melodic synthesis)[/bold blue]")
        else:
            console.print("[bold blue]Using VAE model[/bold blue]")
        self.state.save(self.state_file)

        # Build input description
        input_desc = []
        if self.input_files:
            input_desc.append(f"{len(self.input_files)} file(s) to split")
        if self.include_dirs:
            input_desc.append(f"{len(self.include_dirs)} include dir(s)")

        console.print()
        console.print(Panel.fit(
            f"[bold blue]Scropipe Pipeline[/bold blue]\n"
            f"Inputs: {', '.join(input_desc)}\n"
            f"Output: {self.output_base}",
            title="Starting Pipeline",
        ))
        console.print()

        # Stage 1: Split (if we have input files)
        if self.input_files:
            if self._should_run_stage("split"):
                console.print("[bold]Stage 1: Split[/bold]")
                self._update_stage_state("split", StageStatus.RUNNING)

                split_outputs = []
                all_succeeded = True

                for input_file in self.input_files:
                    result = self.stages["split"].run(
                        input_file=input_file,
                        mode=split_mode,
                        **split_kwargs,
                    )

                    if not result.success:
                        console.print(f"[red]Split failed for {input_file.name}: {result.message}[/red]")
                        all_succeeded = False
                        break

                    if result.output_dir:
                        split_outputs.append(str(result.output_dir))

                if not all_succeeded:
                    self._update_stage_state("split", StageStatus.FAILED, result)
                    return False

                # Store split outputs in state
                self.state.split_outputs = split_outputs
                self._update_stage_state(
                    "split",
                    StageStatus.COMPLETED,
                    StageResult(
                        success=True,
                        output_dir=self.stages["split"].output_dir,
                        message=f"Split {len(self.input_files)} file(s)",
                        details={"split_outputs": split_outputs},
                    ),
                )
                console.print()
            else:
                console.print("[dim]Stage 1: Split (already completed)[/dim]")
        else:
            console.print("[dim]Stage 1: Split (skipped - no input files)[/dim]")
            self._update_stage_state("split", StageStatus.SKIPPED)

        # Stage 2: Collect (pool all samples)
        if self._should_run_stage("collect"):
            console.print("[bold]Stage 2: Collect[/bold]")
            self._update_stage_state("collect", StageStatus.RUNNING)

            # Gather split output directories
            split_dirs = [Path(p) for p in self.state.split_outputs]

            result = self.stages["collect"].run(
                split_dirs=split_dirs,
                include_dirs=self.include_dirs,
            )

            if not result.success:
                self._update_stage_state("collect", StageStatus.FAILED, result)
                console.print(f"[red]Collect failed: {result.message}[/red]")
                return False

            self._update_stage_state("collect", StageStatus.COMPLETED, result)
            console.print()
        else:
            console.print("[dim]Stage 2: Collect (already completed)[/dim]")

        if not synthesize:
            self._print_summary()
            return True

        pool_output = self._get_stage_output("collect")
        if not pool_output:
            console.print("[red]Cannot find pool output[/red]")
            return False

        # Use seed_dir for generation input if specified, otherwise use pool
        gen_input = Path(seed_dir).resolve() if seed_dir else pool_output

        if use_rave:
            # RAVE pipeline (for melodic/harmonic content)
            return self._run_rave_synthesis(pool_output, gen_input, epochs, count, rave_config, no_train)
        else:
            # VAE pipeline (for percussion/textures)
            return self._run_vae_synthesis(
                pool_output, augment, max_duration, epochs, count,
                train_vocoder, vocoder_epochs, no_train
            )

    def _run_vae_synthesis(
        self,
        pool_output: Path,
        augment: bool,
        max_duration: float,
        epochs: int,
        count: int,
        train_vocoder: bool,
        vocoder_epochs: int,
        no_train: bool = False,
    ) -> bool:
        """Run VAE synthesis pipeline."""
        # On resume, always re-run generation (it's fast)
        if self.resume and "generate" in self.state.stages:
            self.state.stages["generate"].status = StageStatus.PENDING

        # Stage 3: Preprocess
        if self._should_run_stage("preprocess"):
            console.print("[bold]Stage 3: Preprocess[/bold]")
            self._update_stage_state("preprocess", StageStatus.RUNNING)

            result = self.stages["preprocess"].run(
                input_dir=pool_output,
                augment=augment,
                max_duration=max_duration,
            )

            if not result.success:
                self._update_stage_state("preprocess", StageStatus.FAILED, result)
                console.print(f"[red]Preprocess failed: {result.message}[/red]")
                return False

            self._update_stage_state("preprocess", StageStatus.COMPLETED, result)
            console.print()
        else:
            console.print("[dim]Stage 3: Preprocess (already completed)[/dim]")

        # Stage 4: Train
        preprocess_output = self._get_stage_output("preprocess")
        if not preprocess_output:
            console.print("[red]Cannot find preprocess output[/red]")
            return False

        if no_train and self._should_run_stage("train"):
            console.print("[yellow]Stage 4: Train (skipped, --no-train)[/yellow]")
            # Check if model exists from a previous run
            train_dir = self.output_base / "03-train"
            model_file = train_dir / "model.pth"
            if not model_file.exists():
                console.print(f"[red]--no-train but no model found at {model_file}[/red]")
                return False
            self._update_stage_state(
                "train",
                StageStatus.COMPLETED,
                StageResult(success=True, output_dir=train_dir, message="Skipped (--no-train)"),
            )
        elif self._should_run_stage("train"):
            console.print("[bold]Stage 4: Train[/bold]")
            self._update_stage_state("train", StageStatus.RUNNING)

            result = self.stages["train"].run(
                data_dir=preprocess_output,
                epochs=epochs,
            )

            if not result.success:
                self._update_stage_state("train", StageStatus.FAILED, result)
                console.print(f"[red]Train failed: {result.message}[/red]")
                return False

            self._update_stage_state("train", StageStatus.COMPLETED, result)
            console.print()
        else:
            console.print("[dim]Stage 4: Train (already completed)[/dim]")

        # Stage 5 (optional): Train Vocoder
        train_output = self._get_stage_output("train")
        if not train_output:
            console.print("[red]Cannot find train output[/red]")
            return False

        model_path = train_output / "model.pth"
        if not model_path.exists():
            console.print(f"[red]Model not found: {model_path}[/red]")
            return False

        vocoder_path = None
        if train_vocoder:
            if self._should_run_stage("train_vocoder"):
                console.print("[bold]Stage 5: Train Vocoder[/bold]")
                self._update_stage_state("train_vocoder", StageStatus.RUNNING)

                result = self.stages["train_vocoder"].run(
                    audio_dir=pool_output,
                    spec_dir=preprocess_output,
                    epochs=vocoder_epochs,
                )

                if not result.success:
                    self._update_stage_state("train_vocoder", StageStatus.FAILED, result)
                    console.print(f"[yellow]Vocoder training failed: {result.message}[/yellow]")
                    console.print("[yellow]Falling back to Griffin-Lim[/yellow]")
                else:
                    self._update_stage_state("train_vocoder", StageStatus.COMPLETED, result)
                    vocoder_output = self._get_stage_output("train_vocoder")
                    if vocoder_output:
                        vocoder_path = vocoder_output / "vocoder.pth"
                console.print()
            else:
                console.print("[dim]Stage 5: Train Vocoder (already completed)[/dim]")
                vocoder_output = self._get_stage_output("train_vocoder")
                if vocoder_output:
                    vocoder_path = vocoder_output / "vocoder.pth"
        else:
            self._update_stage_state("train_vocoder", StageStatus.SKIPPED)

        # Stage 6: Generate
        if self._should_run_stage("generate"):
            console.print("[bold]Stage 6: Generate[/bold]")
            self._update_stage_state("generate", StageStatus.RUNNING)

            result = self.stages["generate"].run(
                model_path=model_path,
                count=count,
                vocoder_path=vocoder_path,
            )

            if not result.success:
                self._update_stage_state("generate", StageStatus.FAILED, result)
                console.print(f"[red]Generate failed: {result.message}[/red]")
                return False

            self._update_stage_state("generate", StageStatus.COMPLETED, result)
            console.print()
        else:
            console.print("[dim]Stage 6: Generate (already completed)[/dim]")

        self._print_summary()
        return True

    def _find_rave_checkpoint(self) -> Optional[Path]:
        """Find the latest RAVE checkpoint for resuming training.

        Looks for 'last.ckpt' in the rave model output dir. Falls back
        to the most recently modified .ckpt file.
        """
        model_dir = self.output_base / "03-rave-model"
        if not model_dir.exists():
            return None

        # Prefer last.ckpt (PyTorch Lightning convention)
        last_ckpts = list(model_dir.glob("**/last.ckpt"))
        if last_ckpts:
            return last_ckpts[0]

        # Fall back to most recent .ckpt
        all_ckpts = list(model_dir.glob("**/*.ckpt"))
        if all_ckpts:
            return max(all_ckpts, key=lambda p: p.stat().st_mtime)

        return None

    def _run_rave_synthesis(
        self,
        pool_output: Path,
        gen_input: Path,
        epochs: int,
        count: int,
        rave_config: str,
        no_train: bool = False,
    ) -> bool:
        """Run RAVE synthesis pipeline for melodic/harmonic content."""
        # On resume, always re-run generation (it's fast)
        if self.resume and "rave_generate" in self.state.stages:
            self.state.stages["rave_generate"].status = StageStatus.PENDING

        # Stage 3: RAVE Preprocess
        if self._should_run_stage("rave_preprocess"):
            console.print("[bold]Stage 3: RAVE Preprocess[/bold]")
            self._update_stage_state("rave_preprocess", StageStatus.RUNNING)

            result = self.stages["rave_preprocess"].run(input_dir=pool_output)

            if not result.success:
                self._update_stage_state("rave_preprocess", StageStatus.FAILED, result)
                console.print(f"[red]RAVE preprocess failed: {result.message}[/red]")
                return False

            self._update_stage_state("rave_preprocess", StageStatus.COMPLETED, result)
            console.print()
        else:
            console.print("[dim]Stage 3: RAVE Preprocess (already completed)[/dim]")

        preprocess_output = self._get_stage_output("rave_preprocess")
        if not preprocess_output:
            console.print("[red]Cannot find RAVE preprocess output[/red]")
            return False

        # Stage 4: RAVE Train
        if no_train:
            # --no-train: skip training, find existing checkpoint and resolve run dir
            ckpt = self._find_rave_checkpoint()
            if not ckpt:
                console.print("[red]--no-train but no checkpoint found to export from[/red]")
                return False
            # Find the run dir (the directory containing config.gin)
            run_dir = ckpt.parent
            while run_dir != run_dir.parent:
                if (run_dir / "config.gin").exists():
                    break
                run_dir = run_dir.parent
            else:
                console.print("[red]Could not find RAVE run directory (no config.gin found)[/red]")
                return False
            console.print(f"[yellow]Stage 4: RAVE Train (skipped, using existing checkpoint)[/yellow]")
            console.print(f"[dim]  Checkpoint: {ckpt}[/dim]")
            self._update_stage_state(
                "rave_train",
                StageStatus.COMPLETED,
                StageResult(success=True, output_dir=run_dir, message="Skipped (--no-train)"),
            )
            # Reset export so it re-runs against the correct run dir
            if "rave_export" in self.state.stages:
                self.state.stages["rave_export"].status = StageStatus.PENDING
        elif self._should_run_stage("rave_train"):
            # Check for a checkpoint to resume from
            resume_ckpt = None
            train_state = self.state.stages.get("rave_train")
            if train_state and train_state.status in (StageStatus.RUNNING, StageStatus.FAILED):
                resume_ckpt = self._find_rave_checkpoint()
                if resume_ckpt:
                    console.print(f"[bold]Stage 4: RAVE Train (resuming from checkpoint)[/bold]")
                    console.print(f"[dim]  Checkpoint: {resume_ckpt}[/dim]")
                else:
                    console.print("[bold]Stage 4: RAVE Train (this takes several hours)[/bold]")
            else:
                console.print("[bold]Stage 4: RAVE Train (this takes several hours)[/bold]")

            self._update_stage_state("rave_train", StageStatus.RUNNING)

            result = self.stages["rave_train"].run(
                data_dir=preprocess_output,
                config=rave_config,
                epochs=epochs if epochs != 100 else None,
                ckpt=resume_ckpt,
            )

            if not result.success:
                self._update_stage_state("rave_train", StageStatus.FAILED, result)
                console.print(f"[red]RAVE training failed: {result.message}[/red]")
                return False

            self._update_stage_state("rave_train", StageStatus.COMPLETED, result)
            console.print()
        else:
            console.print("[dim]Stage 4: RAVE Train (already completed)[/dim]")

        train_output = self._get_stage_output("rave_train")
        if not train_output:
            console.print("[red]Cannot find RAVE train output[/red]")
            return False

        # Stage 5: RAVE Export
        if self._should_run_stage("rave_export"):
            # Check if .ts file already exists (export already done)
            existing_ts = list(train_output.glob("**/*.ts"))
            if existing_ts:
                console.print("[yellow]Stage 5: RAVE Export (model already exported)[/yellow]")
                self._update_stage_state(
                    "rave_export",
                    StageStatus.COMPLETED,
                    StageResult(
                        success=True,
                        output_dir=train_output,
                        message="RAVE model already exported",
                        details={"model_path": str(existing_ts[0])},
                    ),
                )
            else:
                console.print("[bold]Stage 5: RAVE Export[/bold]")
                self._update_stage_state("rave_export", StageStatus.RUNNING)

                result = self.stages["rave_export"].run(run_dir=train_output)

                if not result.success:
                    self._update_stage_state("rave_export", StageStatus.FAILED, result)
                    console.print(f"[red]RAVE export failed: {result.message}[/red]")
                    return False

                self._update_stage_state("rave_export", StageStatus.COMPLETED, result)
                console.print()
        else:
            console.print("[dim]Stage 5: RAVE Export (already completed)[/dim]")

        # Find exported model
        model_path = None
        for ts_file in train_output.glob("**/*.ts"):
            model_path = ts_file
            break

        if not model_path:
            console.print("[red]Could not find exported RAVE model[/red]")
            return False

        # Stage 6: RAVE Generate
        if self._should_run_stage("rave_generate"):
            console.print("[bold]Stage 6: RAVE Generate[/bold]")
            self._update_stage_state("rave_generate", StageStatus.RUNNING)

            if gen_input != pool_output:
                console.print(f"[dim]  Seed audio: {gen_input}[/dim]")
            result = self.stages["rave_generate"].run(
                model_path=model_path,
                input_dir=gen_input,
                count=count,
            )

            if not result.success:
                self._update_stage_state("rave_generate", StageStatus.FAILED, result)
                console.print(f"[red]RAVE generation failed: {result.message}[/red]")
                return False

            self._update_stage_state("rave_generate", StageStatus.COMPLETED, result)
            console.print()
        else:
            console.print("[dim]Stage 6: RAVE Generate (already completed)[/dim]")

        self._print_summary()
        return True

    def _print_summary(self):
        """Print pipeline summary."""
        table = Table(title="Pipeline Summary")
        table.add_column("Stage", style="cyan")
        table.add_column("Status")
        table.add_column("Output")

        status_styles = {
            StageStatus.COMPLETED: "[green]Completed[/green]",
            StageStatus.FAILED: "[red]Failed[/red]",
            StageStatus.RUNNING: "[yellow]Running[/yellow]",
            StageStatus.PENDING: "[dim]Pending[/dim]",
            StageStatus.SKIPPED: "[dim]Skipped[/dim]",
        }

        stage_order = self.STAGE_ORDER
        if self.state.config.get("model", "").lower() == "rave":
            stage_order = self.RAVE_STAGE_ORDER

        for stage_name in stage_order:
            if stage_name in self.state.stages:
                state = self.state.stages[stage_name]
                status_text = status_styles.get(state.status, str(state.status))
                output = state.output_dir or "-"
                if len(output) > 40:
                    output = "..." + output[-37:]
                table.add_row(stage_name.capitalize(), status_text, output)
            else:
                table.add_row(stage_name.capitalize(), "[dim]Not run[/dim]", "-")

        console.print()
        console.print(table)
        console.print()
        console.print(f"[dim]State saved to: {self.state_file}[/dim]")
