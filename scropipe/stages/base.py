"""Base class for pipeline stages."""

import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from rich.console import Console

console = Console()


@dataclass
class StageResult:
    """Result of a stage execution."""

    success: bool
    output_dir: Optional[Path] = None
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.success


class Stage(ABC):
    """Abstract base class for pipeline stages."""

    name: str = "base"
    description: str = "Base stage"

    def __init__(self, output_base: Path):
        """Initialize the stage.

        Args:
            output_base: Base output directory for the pipeline.
        """
        self.output_base = Path(output_base)

    @property
    def output_dir(self) -> Path:
        """Get the output directory for this stage."""
        return self.output_base / self.name

    @abstractmethod
    def run(self, **kwargs) -> StageResult:
        """Execute the stage.

        Returns:
            StageResult with success status and details.
        """
        pass

    def run_command(
        self,
        cmd: list[str],
        capture_output: bool = True,
        check: bool = True,
    ) -> subprocess.CompletedProcess:
        """Run a subprocess command.

        Args:
            cmd: Command and arguments.
            capture_output: Whether to capture stdout/stderr.
            check: Whether to raise on non-zero exit.

        Returns:
            CompletedProcess result.
        """
        console.print(f"[dim]Running: {' '.join(str(c) for c in cmd)}[/dim]")
        return subprocess.run(
            cmd,
            capture_output=capture_output,
            text=True,
            check=check,
        )

    def ensure_output_dir(self) -> Path:
        """Create and return the output directory."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return self.output_dir

    def log(self, message: str, style: str = ""):
        """Log a message to console.

        Args:
            message: Message to log.
            style: Rich style string.
        """
        if style:
            console.print(f"[{style}]{message}[/{style}]")
        else:
            console.print(message)

    def log_success(self, message: str):
        """Log a success message."""
        self.log(f"[bold green]OK[/bold green] {message}")

    def log_error(self, message: str):
        """Log an error message."""
        self.log(f"[bold red]ERR[/bold red] {message}")
