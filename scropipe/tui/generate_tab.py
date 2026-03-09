"""Generate tab widget for the scropipe TUI."""

from __future__ import annotations

import threading
from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, DataTable, Input, Label, ProgressBar, Select, Static

from .path_suggester import PathSuggester


class GenerateTab(Static):
    """Generation interface tab for running inference with trained RAVE models."""

    DEFAULT_CSS = """
    GenerateTab {
        height: 1fr;
        padding: 1 2;
    }

    #gen-models-table {
        height: 10;
        margin-bottom: 1;
    }

    .hidden {
        display: none;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._generation_thread: threading.Thread | None = None

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("Model", classes="section-title")
            yield Select([], id="gen-model-select", prompt="Select a model")

            yield Label("Models", classes="section-title")
            yield DataTable(id="gen-models-table", cursor_type="row")

            yield Label("Input Directory", classes="section-title")
            with Horizontal(classes="form-group"):
                yield Input(placeholder="Path to input WAV files", id="gen-input-dir", suggester=PathSuggester())
                yield Button("Browse", id="gen-input-browse-btn")
            yield Static("", id="gen-input-info")

            yield Label("Output Directory", classes="section-title")
            with Horizontal(classes="form-group"):
                yield Input(placeholder="Path for output files", id="gen-output-dir", suggester=PathSuggester(directories_only=True))
                yield Button("Browse", id="gen-output-browse-btn")

            yield Static("", id="gen-status")
            yield ProgressBar(id="gen-progress", total=100)

            with Horizontal(classes="action-bar"):
                yield Button("Generate", variant="primary", id="generate-btn")
                yield Button("Delete Model", variant="error", id="gen-delete-model-btn")

    def on_mount(self) -> None:
        """Set up table columns and populate models on mount."""
        table = self.query_one("#gen-models-table", DataTable)
        table.add_columns("Name", "Arch", "Pool", "Samples", "Size")

        # Hide progress bar initially
        self.query_one("#gen-progress", ProgressBar).display = False

        self._refresh_models()

    def _get_model_manager(self):
        from ..model_manager import ModelManager

        return ModelManager(
            self.app.models_dir or Path.home() / ".local/share/scropipe/models"
        )

    def _refresh_models(self) -> None:
        """Reload models list into both the Select and DataTable."""
        try:
            mm = self._get_model_manager()
            models = mm.list_models()
        except Exception:
            models = []

        # Update Select
        select = self.query_one("#gen-model-select", Select)
        options = [(m.name, m.name) for m in models]
        select.set_options(options)

        # Update DataTable
        table = self.query_one("#gen-models-table", DataTable)
        table.clear()
        for m in models:
            pool = m.pool_name or ""
            size = f"{m.size_mb:.1f} MB"
            table.add_row(m.name, m.config, pool, str(m.total_samples), size)

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """When a row is selected in the DataTable, also select that model in the dropdown."""
        table = self.query_one("#gen-models-table", DataTable)
        row_data = table.get_row(event.row_key)
        if row_data:
            model_name = row_data[0]
            select = self.query_one("#gen-model-select", Select)
            select.value = model_name

    def on_input_changed(self, event: Input.Changed) -> None:
        """Count WAV files when input directory changes."""
        if event.input.id == "gen-input-dir":
            self._update_input_info(event.value)

    def _update_input_info(self, path_str: str) -> None:
        """Count WAV files in the given directory and update the info label."""
        info = self.query_one("#gen-input-info", Static)
        if not path_str.strip():
            info.update("")
            return
        input_path = Path(path_str.strip()).expanduser()
        if input_path.is_dir():
            wav_files = list(input_path.glob("*.wav"))
            info.update(f"Found: {len(wav_files)} files")
        else:
            info.update("Not a valid directory")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "generate-btn":
            self._start_generation()
        elif event.button.id == "gen-delete-model-btn":
            self._delete_selected_model()
        elif event.button.id == "gen-input-browse-btn":
            self._browse_input_dir()
        elif event.button.id == "gen-output-browse-btn":
            self._browse_output_dir()

    def _browse_input_dir(self) -> None:
        """Open a modal to browse for the input directory."""
        from .browse_modal import BrowseModal

        self.app.push_screen(
            BrowseModal(title="Input Directory", select_type="directory"),
            callback=self._on_input_dir_selected,
        )

    def _on_input_dir_selected(self, path: Path | None) -> None:
        """Handle input directory modal result."""
        if path is not None:
            self.query_one("#gen-input-dir", Input).value = str(path)

    def _browse_output_dir(self) -> None:
        """Open a modal to browse for the output directory."""
        from .browse_modal import BrowseModal

        self.app.push_screen(
            BrowseModal(title="Output Directory", select_type="directory"),
            callback=self._on_output_dir_selected,
        )

    def _on_output_dir_selected(self, path: Path | None) -> None:
        """Handle output directory modal result."""
        if path is not None:
            self.query_one("#gen-output-dir", Input).value = str(path)

    def _start_generation(self) -> None:
        """Validate inputs and start the generation process."""
        select = self.query_one("#gen-model-select", Select)
        input_dir = self.query_one("#gen-input-dir", Input).value.strip()
        output_dir = self.query_one("#gen-output-dir", Input).value.strip()
        status = self.query_one("#gen-status", Static)

        # Validate model selection
        if select.value is Select.BLANK:
            status.update("Error: Please select a model.")
            return

        model_name = str(select.value)

        # Validate input directory
        if not input_dir:
            status.update("Error: Please specify an input directory.")
            return
        input_path = Path(input_dir).expanduser()
        if not input_path.is_dir():
            status.update("Error: Input directory does not exist.")
            return

        # Validate output directory
        if not output_dir:
            status.update("Error: Please specify an output directory.")
            return

        output_path = Path(output_dir).expanduser()
        output_path.mkdir(parents=True, exist_ok=True)

        # Show progress bar
        progress = self.query_one("#gen-progress", ProgressBar)
        progress.display = True
        progress.update(progress=0)

        status.update(f"Generating with model {model_name}...")

        # Get model path
        try:
            mm = self._get_model_manager()
            model_path = mm.get_model_path(model_name)
        except KeyError:
            status.update(f"Error: Model '{model_name}' not found.")
            progress.display = False
            return

        # Start worker thread
        self._generation_thread = threading.Thread(
            target=self._generation_worker,
            args=(model_path, input_path, output_path),
            daemon=True,
        )
        self._generation_thread.start()

    def _generation_worker(
        self, model_path: Path, input_dir: Path, output_dir: Path
    ) -> None:
        """Worker thread that runs the generation process.

        Loads the RAVE model with torch.jit.load, processes input WAV files,
        and saves results to the output directory.
        """
        try:
            import torch
            import torchaudio
        except ImportError:
            try:
                self.app.call_from_thread(
                    self._update_generation_status,
                    "Error: torch/torchaudio not installed. Install with: pip install torch torchaudio",
                )
                self.app.call_from_thread(self._hide_progress)
            except Exception:
                pass
            return

        wav_files = list(input_dir.glob("*.wav"))
        if not wav_files:
            try:
                self.app.call_from_thread(
                    self._update_generation_status,
                    "Error: No WAV files found in input directory.",
                )
                self.app.call_from_thread(self._hide_progress)
            except Exception:
                pass
            return

        try:
            model = torch.jit.load(str(model_path))
            model.eval()
        except Exception as e:
            try:
                self.app.call_from_thread(
                    self._update_generation_status,
                    f"Error loading model: {e}",
                )
                self.app.call_from_thread(self._hide_progress)
            except Exception:
                pass
            return

        total = len(wav_files)
        for i, wav_file in enumerate(wav_files):
            try:
                audio, sr = torchaudio.load(str(wav_file))
                with torch.no_grad():
                    # RAVE expects (batch, channels, samples)
                    if audio.dim() == 2:
                        audio = audio.unsqueeze(0)
                    z = model.encode(audio)
                    output = model.decode(z)
                    output = output.squeeze(0)

                out_path = output_dir / wav_file.name
                torchaudio.save(str(out_path), output, sr)
            except Exception as e:
                try:
                    self.app.call_from_thread(
                        self._update_generation_status,
                        f"Warning: Failed to process {wav_file.name}: {e}",
                    )
                except Exception:
                    pass

            # Update progress
            progress_pct = int((i + 1) / total * 100)
            try:
                self.app.call_from_thread(self._update_progress, progress_pct)
            except Exception:
                pass

        try:
            self.app.call_from_thread(
                self._update_generation_status,
                f"Generation complete. Processed {total} files.",
            )
        except Exception:
            pass

    def _update_generation_status(self, message: str) -> None:
        """Update the generation status label."""
        self.query_one("#gen-status", Static).update(message)

    def _update_progress(self, value: int) -> None:
        """Update the progress bar."""
        progress = self.query_one("#gen-progress", ProgressBar)
        progress.update(progress=value)

    def _hide_progress(self) -> None:
        """Hide the progress bar."""
        self.query_one("#gen-progress", ProgressBar).display = False

    def _delete_selected_model(self) -> None:
        """Delete the currently selected model."""
        select = self.query_one("#gen-model-select", Select)
        status = self.query_one("#gen-status", Static)

        if select.value is Select.BLANK:
            status.update("Error: No model selected to delete.")
            return

        model_name = str(select.value)
        try:
            mm = self._get_model_manager()
            mm.delete_model(model_name)
            status.update(f"Deleted model: {model_name}")
            self._refresh_models()
        except KeyError:
            status.update(f"Error: Model '{model_name}' not found.")

    def _update_status(self, message: str) -> None:
        """Update the app status bar."""
        try:
            status_bar = self.app.query_one("#status-bar", Static)
            status_bar.update(message)
        except Exception:
            pass
