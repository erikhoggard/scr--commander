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

        if output_dir:
            self.output_base = Path(output_dir).resolve()
        else:
            self.output_base = Path.cwd() / "scropipe-output"

        self.state_file = self.output_base / "pipeline.json"

        if resume and self.state_file.exists():
            self.state = PipelineState.load(self.state_file)
            console.print(f"[dim]Resuming pipeline from {self.state_file}[/dim]")
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
            train_vocoder: Whether to train HiFi-GAN vocoder.
            vocoder_epochs: Vocoder training epochs.
            **split_kwargs: Additional arguments for split stage.

        Returns:
            True if pipeline completed successfully.
        """
        # Validate inputs
        if not self.input_files and not self.include_dirs:
            console.print("[red]No input files or include directories specified[/red]")
            return False

        self.output_base.mkdir(parents=True, exist_ok=True)

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
            **split_kwargs,
        }
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

        # Stage 3: Preprocess
        pool_output = self._get_stage_output("collect")
        if not pool_output:
            console.print("[red]Cannot find pool output[/red]")
            return False

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

        if self._should_run_stage("train"):
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

        for stage_name in self.STAGE_ORDER:
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
