"""Main TUI application for scropipe."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.widgets import Button, Header, Input, Label, Static, TabbedContent, TabPane

from .path_suggester import PathSuggester

from .generate_tab import GenerateTab
from .pool_tab import PoolTab
from .split_tab import SplitTab
from .train_tab import TrainTab


class SetupModal(ModalScreen[tuple[Path, Path]]):
    """First-run setup modal for configuring directory paths."""

    DEFAULT_CSS = """
    SetupModal {
        align: center middle;
    }

    SetupModal > Vertical {
        width: 60;
        height: auto;
        max-height: 20;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }

    SetupModal Input {
        margin-bottom: 1;
    }

    SetupModal .error-label {
        color: $error;
        margin-bottom: 1;
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("Welcome to scropipe!", classes="section-title")
            yield Label("Please configure your directories to get started.")
            yield Label("")
            yield Label("Models directory:")
            yield Input(placeholder="e.g. ~/scropipe/models", id="models-dir-input", suggester=PathSuggester(directories_only=True))
            yield Label("Pools directory:")
            yield Input(placeholder="e.g. ~/scropipe/pools", id="pools-dir-input", suggester=PathSuggester(directories_only=True))
            yield Label("", id="setup-error", classes="error-label")
            with Horizontal(classes="action-bar"):
                yield Button("Continue", variant="primary", id="setup-continue")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "setup-continue":
            models_input = self.query_one("#models-dir-input", Input)
            pools_input = self.query_one("#pools-dir-input", Input)

            models_val = models_input.value.strip()
            pools_val = pools_input.value.strip()

            if not models_val or not pools_val:
                error_label = self.query_one("#setup-error", Label)
                error_label.update("Both directories must be specified.")
                return

            models_path = Path(models_val).expanduser().resolve()
            pools_path = Path(pools_val).expanduser().resolve()

            self.dismiss((models_path, pools_path))


TAB_IDS = {
    "1": "tab-split",
    "2": "tab-pool",
    "3": "tab-train",
    "4": "tab-generate",
}


class ScropipeApp(App):
    """Scropipe TUI application."""

    TITLE = "scropipe"
    CSS_PATH = "styles.tcss"

    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit"),
        Binding("1", "switch_tab('tab-split')", "Split", show=False),
        Binding("2", "switch_tab('tab-pool')", "Pool", show=False),
        Binding("3", "switch_tab('tab-train')", "Train", show=False),
        Binding("4", "switch_tab('tab-generate')", "Generate", show=False),
    ]

    active_pool: reactive[str] = reactive("none")
    active_model: reactive[str] = reactive("none")
    gpu_info: reactive[str] = reactive("detecting...")

    def __init__(
        self,
        models_dir: Optional[Path] = None,
        pools_dir: Optional[Path] = None,
    ) -> None:
        super().__init__()
        self.models_dir = models_dir
        self.pools_dir = pools_dir

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent():
            with TabPane("Split", id="tab-split"):
                yield SplitTab()
            with TabPane("Pool", id="tab-pool"):
                yield PoolTab()
            with TabPane("Train", id="tab-train"):
                yield TrainTab()
            with TabPane("Generate", id="tab-generate"):
                yield GenerateTab()
        yield Static("Ready", id="status-bar")

    def action_switch_tab(self, tab_id: str) -> None:
        """Switch to the specified tab."""
        tabbed = self.query_one(TabbedContent)
        tabbed.active = tab_id

    def on_tabbed_content_tab_activated(
        self, event: TabbedContent.TabActivated
    ) -> None:
        """Refresh tab data when switching tabs."""
        tab_id = event.pane.id
        if tab_id == "tab-train":
            try:
                config = self.query_one("#train-config")
                config._populate_pools()
                config._populate_resumable_runs()
            except Exception:
                pass
        elif tab_id == "tab-split":
            try:
                split_tab = self.query_one("SplitTab")
                if hasattr(split_tab, '_refresh_pools'):
                    split_tab._refresh_pools()
            except Exception:
                pass
        elif tab_id == "tab-generate":
            try:
                gen_tab = self.query_one("GenerateTab")
                gen_tab._refresh_models()
            except Exception:
                pass

    def watch_active_pool(self) -> None:
        self._update_status_bar()

    def watch_active_model(self) -> None:
        self._update_status_bar()

    def watch_gpu_info(self) -> None:
        self._update_status_bar()

    def _update_status_bar(self) -> None:
        try:
            bar = self.query_one("#status-bar", Static)
            bar.update(
                f"Pool: {self.active_pool} | Model: {self.active_model} "
                f"| GPU: {self.gpu_info} | Ctrl+Q: Quit"
            )
        except Exception:
            pass

    def _detect_gpu(self) -> None:
        try:
            import torch

            if torch.cuda.is_available():
                self.gpu_info = torch.cuda.get_device_name(0)
            else:
                self.gpu_info = "CPU"
        except ImportError:
            self.gpu_info = "CPU"

    def on_mount(self) -> None:
        """Push setup modal if directories are not configured."""
        self._detect_gpu()
        if self.models_dir is None or self.pools_dir is None:
            self.push_screen(SetupModal(), callback=self._on_setup_complete)

    def _on_setup_complete(self, result: tuple[Path, Path]) -> None:
        """Handle setup modal completion."""
        models_path, pools_path = result
        self.models_dir = models_path
        self.pools_dir = pools_path

        # Create directories
        models_path.mkdir(parents=True, exist_ok=True)
        pools_path.mkdir(parents=True, exist_ok=True)

        # Save config
        from ..config import ScropipeConfig, save_config

        config = ScropipeConfig(models_dir=models_path, pools_dir=pools_path)
        save_config(config)
