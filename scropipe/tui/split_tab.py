"""Split tab widget for the scropipe TUI."""

from __future__ import annotations

from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Input, Label, RadioButton, RadioSet, Static
from textual.worker import Worker


class TransientSettings(Static):
    """Settings panel for transient split mode."""

    def compose(self) -> ComposeResult:
        yield Label("Sensitivity (delta):")
        yield Input(value="0.07", id="transient-delta")
        yield Label("Min length (s):")
        yield Input(value="0.05", id="transient-min-length")
        yield Label("Max length (s):")
        yield Input(value="10.0", id="transient-max-length")


class GridSettings(Static):
    """Settings panel for grid split mode."""

    def compose(self) -> ComposeResult:
        yield Label("Chunk length (s):")
        yield Input(value="2.0", id="grid-chunk-length")
        yield Label("BPM:")
        yield Input(placeholder="optional", id="grid-bpm")
        yield Label("Bars:")
        yield Input(value="4", id="grid-bars")


class TextureSettings(Static):
    """Settings panel for texture split mode."""

    def compose(self) -> ComposeResult:
        yield Label("Min duration (s):")
        yield Input(value="1.0", id="texture-min-duration")
        yield Label("Max duration (s):")
        yield Input(value="30.0", id="texture-max-duration")
        yield Label("RMS threshold:")
        yield Input(value="0.1", id="texture-rms-threshold")
        yield Label("Stability threshold:")
        yield Input(value="0.15", id="texture-stability-threshold")


class SplitTab(Static):
    """Audio splitting interface tab."""

    DEFAULT_CSS = """
    SplitTab {
        height: auto;
        padding: 1 2;
    }

    TransientSettings, GridSettings, TextureSettings {
        height: auto;
        padding: 0 1;
        margin-bottom: 1;
    }

    .settings-hidden {
        display: none;
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical():
            # Source file
            yield Label("Source File", classes="section-title")
            with Horizontal(classes="form-group"):
                yield Input(placeholder="Path to audio file", id="split-source-input")
                yield Button("Browse", id="split-browse-source")

            # Splitting mode
            yield Label("Splitting Mode", classes="section-title")
            with RadioSet(id="split-mode-selector"):
                yield RadioButton("Transient", value=True)
                yield RadioButton("Grid")
                yield RadioButton("Texture")

            # Mode-specific settings panels
            yield TransientSettings(id="transient-settings")
            yield GridSettings(id="grid-settings", classes="settings-hidden")
            yield TextureSettings(id="texture-settings", classes="settings-hidden")

            # Output directory
            yield Label("Output Directory", classes="section-title")
            with Horizontal(classes="form-group"):
                yield Input(placeholder="Path to output directory", id="split-output-input")
                yield Button("Browse", id="split-browse-output")

            # Status
            yield Static("Ready", id="split-status")

            # Action buttons
            with Horizontal(classes="action-bar"):
                yield Button("Split", variant="primary", id="split-btn")
                yield Button("Split & Add to Pool", id="split-and-pool-btn")

    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        """Show/hide settings panels based on selected mode."""
        transient = self.query_one("#transient-settings", TransientSettings)
        grid = self.query_one("#grid-settings", GridSettings)
        texture = self.query_one("#texture-settings", TextureSettings)

        # Hide all
        transient.add_class("settings-hidden")
        grid.add_class("settings-hidden")
        texture.add_class("settings-hidden")

        # Show the selected one
        index = event.radio_set.pressed_index
        if index == 0:
            transient.remove_class("settings-hidden")
        elif index == 1:
            grid.remove_class("settings-hidden")
        elif index == 2:
            texture.remove_class("settings-hidden")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "split-btn":
            self._run_split(add_to_pool=False)
        elif event.button.id == "split-and-pool-btn":
            self._run_split(add_to_pool=True)

    def _run_split(self, add_to_pool: bool = False) -> None:
        """Validate and run the split operation."""
        source_input = self.query_one("#split-source-input", Input)
        output_input = self.query_one("#split-output-input", Input)
        status = self.query_one("#split-status", Static)

        source_path = Path(source_input.value.strip())
        if not source_input.value.strip():
            status.update("Error: Please specify a source file.")
            return
        if not source_path.exists():
            status.update(f"Error: Source file not found: {source_path}")
            return

        output_dir = output_input.value.strip()
        if not output_dir:
            output_dir = str(source_path.parent / "splits")

        # Determine mode
        radio_set = self.query_one("#split-mode-selector", RadioSet)
        mode_index = radio_set.pressed_index
        mode_map = {0: "transient", 1: "grid", 2: "texture"}
        mode = mode_map.get(mode_index, "transient")

        # Gather mode-specific kwargs
        kwargs = self._gather_mode_kwargs(mode)
        kwargs["input_file"] = source_path
        kwargs["mode"] = mode

        status.update(f"Splitting with {mode} mode...")

        self.run_worker(
            self._do_split(Path(output_dir), kwargs, add_to_pool),
            name="split_worker",
            thread=True,
        )

    def _gather_mode_kwargs(self, mode: str) -> dict:
        """Gather mode-specific parameters from input fields."""
        kwargs: dict = {}

        if mode == "transient":
            kwargs["delta"] = float(self.query_one("#transient-delta", Input).value)
            kwargs["min_length"] = float(
                self.query_one("#transient-min-length", Input).value
            )
            kwargs["max_length"] = float(
                self.query_one("#transient-max-length", Input).value
            )
        elif mode == "grid":
            kwargs["chunk_length"] = float(
                self.query_one("#grid-chunk-length", Input).value
            )
            bpm_val = self.query_one("#grid-bpm", Input).value.strip()
            if bpm_val:
                kwargs["bpm"] = float(bpm_val)
            kwargs["bars"] = int(self.query_one("#grid-bars", Input).value)
        elif mode == "texture":
            kwargs["min_duration"] = float(
                self.query_one("#texture-min-duration", Input).value
            )
            kwargs["max_duration"] = float(
                self.query_one("#texture-max-duration", Input).value
            )
            kwargs["rms_threshold"] = float(
                self.query_one("#texture-rms-threshold", Input).value
            )
            kwargs["stability_threshold"] = float(
                self.query_one("#texture-stability-threshold", Input).value
            )

        return kwargs

    async def _do_split(
        self, output_dir: Path, kwargs: dict, add_to_pool: bool
    ) -> None:
        """Run the split stage in a worker thread."""
        from scropipe.stages import SplitStage

        status = self.query_one("#split-status", Static)

        try:
            stage = SplitStage(output_base=output_dir.parent)
            result = stage.run(**kwargs)

            if result.success:
                msg = f"Success: {result.message}"
                if add_to_pool:
                    msg += " (ready to add to pool)"
                self.app.call_from_thread(status.update, msg)
            else:
                self.app.call_from_thread(status.update, f"Error: {result.message}")
        except Exception as e:
            self.app.call_from_thread(status.update, f"Error: {e}")
