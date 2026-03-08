"""Train tab widget for the scropipe TUI."""

from __future__ import annotations

import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Input, Label, RadioButton, RadioSet, Select, Sparkline, Static


class TrainConfigPanel(Static):
    """Configuration panel for setting up a training run."""

    DEFAULT_CSS = """
    TrainConfigPanel {
        height: auto;
        padding: 1 2;
    }

    .hidden {
        display: none;
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("Pool", classes="section-title")
            yield Select([], id="train-pool-select", prompt="Select a pool")

            yield Label("Model Name", classes="section-title")
            yield Input(placeholder="Enter model name", id="train-model-name")

            yield Label("Stop Condition", classes="section-title")
            with RadioSet(id="stop-condition"):
                yield RadioButton("Manual (stop when ready)", value=True)
                yield RadioButton("Max steps:")
                yield RadioButton("Delta target:")

            yield Input(
                value="10000",
                id="train-max-steps",
                placeholder="Max training steps",
                classes="hidden",
            )
            yield Input(
                value="0.001",
                id="train-delta-target",
                placeholder="Target delta value",
                classes="hidden",
            )

            yield Label("Architecture", classes="section-title")
            yield Select(
                [("v2", "v2"), ("v2_small", "v2_small"), ("discrete", "discrete")],
                id="train-arch-select",
                value="v2",
            )

            yield Label("Checkpoint Interval (steps)", classes="section-title")
            yield Input(value="500", id="train-val-every")

            yield Static("GPU: detecting...", id="train-gpu-info")

            with Horizontal(classes="action-bar"):
                yield Button("Start Training", variant="primary", id="start-training-btn")

    def on_mount(self) -> None:
        """Populate pool list and detect GPU on mount."""
        self._populate_pools()
        self._detect_gpu()

    def _populate_pools(self) -> None:
        """Load available pools into the pool selector."""
        from ..pool_manager import PoolManager

        pool_select = self.query_one("#train-pool-select", Select)
        try:
            pm = PoolManager(self.app.pools_dir or Path.home() / ".local/share/scropipe/pools")
            pools = pm.list_pools()
            options = [(f"{p.name} ({p.sample_count} samples)", p.name) for p in pools]
            pool_select.set_options(options)
        except Exception:
            pool_select.set_options([])

    def _detect_gpu(self) -> None:
        """Detect GPU and update the info label."""
        gpu_info = self.query_one("#train-gpu-info", Static)
        try:
            import torch

            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                gpu_info.update(f"GPU: {name}")
            else:
                gpu_info.update("GPU: None (CPU training)")
        except ImportError:
            gpu_info.update("GPU: torch not installed")

    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        """Show/hide conditional inputs based on stop condition selection."""
        max_steps = self.query_one("#train-max-steps", Input)
        delta_target = self.query_one("#train-delta-target", Input)

        index = event.radio_set.pressed_index

        # Hide both by default
        max_steps.add_class("hidden")
        delta_target.add_class("hidden")

        if index == 1:
            max_steps.remove_class("hidden")
        elif index == 2:
            delta_target.remove_class("hidden")


class TrainDashboard(Static):
    """Live training dashboard showing metrics and controls."""

    DEFAULT_CSS = """
    TrainDashboard {
        height: auto;
        padding: 1 2;
    }

    #dash-sparkline {
        height: 5;
        margin: 1 0;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._loss_data: list[float] = []

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("Training...", id="dash-title")
            yield Static("Pool: - | Architecture: -", id="dash-info")
            yield Static("Step: 0 | Loss: - | Delta: -", id="dash-metrics")
            yield Sparkline([], id="dash-sparkline")
            yield Static("Elapsed: 0:00 | ETA: -", id="dash-timing")
            yield Static("No checkpoints yet", id="dash-checkpoint")
            with Horizontal(classes="action-bar"):
                yield Button("Stop Training", variant="warning", id="stop-training-btn")

    def update_metrics(self, step: int, loss: float, delta: float) -> None:
        """Update the dashboard metrics display and sparkline."""
        metrics = self.query_one("#dash-metrics", Static)
        metrics.update(f"Step: {step} | Loss: {loss:.6f} | Delta: {delta:.6f}")

        self._loss_data.append(loss)
        sparkline = self.query_one("#dash-sparkline", Sparkline)
        sparkline.data = self._loss_data

    def set_info(self, pool_name: str, architecture: str) -> None:
        """Set the info line with pool and architecture details."""
        info = self.query_one("#dash-info", Static)
        info.update(f"Pool: {pool_name} | Architecture: {architecture}")

    def set_title(self, model_name: str) -> None:
        """Set the dashboard title."""
        title = self.query_one("#dash-title", Label)
        title.update(f"Training: {model_name}")

    def update_timing(self, elapsed: str, eta: str) -> None:
        """Update the timing display."""
        timing = self.query_one("#dash-timing", Static)
        timing.update(f"Elapsed: {elapsed} | ETA: {eta}")

    def update_checkpoint(self, message: str) -> None:
        """Update the checkpoint info."""
        checkpoint = self.query_one("#dash-checkpoint", Static)
        checkpoint.update(message)

    def reset(self) -> None:
        """Reset dashboard state for a new training run."""
        self._loss_data = []
        self.query_one("#dash-title", Label).update("Training...")
        self.query_one("#dash-info", Static).update("Pool: - | Architecture: -")
        self.query_one("#dash-metrics", Static).update("Step: 0 | Loss: - | Delta: -")
        self.query_one("#dash-sparkline", Sparkline).data = []
        self.query_one("#dash-timing", Static).update("Elapsed: 0:00 | ETA: -")
        self.query_one("#dash-checkpoint", Static).update("No checkpoints yet")


class TrainTab(Static):
    """Training interface tab with config and dashboard views."""

    DEFAULT_CSS = """
    TrainTab {
        height: 1fr;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._training_process: Optional[subprocess.Popen] = None
        self._training_thread: Optional[threading.Thread] = None
        self._stop_requested = threading.Event()

    def compose(self) -> ComposeResult:
        yield TrainConfigPanel(id="train-config")
        yield TrainDashboard(id="train-dashboard")

    def on_mount(self) -> None:
        """Hide dashboard on initial mount."""
        self.query_one("#train-dashboard", TrainDashboard).display = False

    def _switch_to_dashboard(self) -> None:
        """Switch from config view to dashboard view."""
        self.query_one("#train-config", TrainConfigPanel).display = False
        self.query_one("#train-dashboard", TrainDashboard).display = True

    def _switch_to_config(self) -> None:
        """Switch from dashboard view to config view."""
        self.query_one("#train-dashboard", TrainDashboard).display = False
        self.query_one("#train-config", TrainConfigPanel).display = True

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses for training controls."""
        if event.button.id == "start-training-btn":
            self._start_training()
        elif event.button.id == "stop-training-btn":
            self._stop_training()

    def _start_training(self) -> None:
        """Validate inputs and start the training process."""
        config = self.query_one("#train-config", TrainConfigPanel)
        pool_select = config.query_one("#train-pool-select", Select)
        model_name_input = config.query_one("#train-model-name", Input)

        # Validate pool selection
        if pool_select.value is Select.BLANK:
            self._update_status("Error: Please select a pool.")
            return

        # Validate model name
        model_name = model_name_input.value.strip()
        if not model_name:
            self._update_status("Error: Please enter a model name.")
            return

        pool_name = str(pool_select.value)

        # Get architecture
        arch_select = config.query_one("#train-arch-select", Select)
        architecture = str(arch_select.value) if arch_select.value is not Select.BLANK else "v2"

        # Get stop condition
        stop_condition_set = config.query_one("#stop-condition", RadioSet)
        stop_index = stop_condition_set.pressed_index

        stop_condition = "manual"
        stop_value = None
        if stop_index == 1:
            stop_condition = "max_steps"
            stop_value = config.query_one("#train-max-steps", Input).value.strip()
        elif stop_index == 2:
            stop_condition = "delta_target"
            stop_value = config.query_one("#train-delta-target", Input).value.strip()

        # Get checkpoint interval
        val_every = config.query_one("#train-val-every", Input).value.strip()

        # Set up dashboard
        dashboard = self.query_one("#train-dashboard", TrainDashboard)
        dashboard.reset()
        dashboard.set_title(model_name)
        dashboard.set_info(pool_name, architecture)

        # Switch views
        self._switch_to_dashboard()
        self._update_status(f"Training {model_name}...")

        # Start the training worker thread
        self._stop_requested.clear()
        self._training_thread = threading.Thread(
            target=self._training_worker,
            args=(pool_name, model_name, architecture, stop_condition, stop_value, val_every),
            daemon=True,
        )
        self._training_thread.start()

    def _training_worker(
        self,
        pool_name: str,
        model_name: str,
        architecture: str,
        stop_condition: str,
        stop_value: Optional[str],
        val_every: str,
    ) -> None:
        """Worker thread that runs the training process.

        This is a stub implementation. The actual training integration
        would involve:
        1. Getting the pool's samples dir via PoolManager
        2. Running RAVE preprocess + train as subprocess
        3. Parsing stdout for step/loss data
        4. Calling update_metrics() via app.call_from_thread()
        """
        dashboard = self.query_one("#train-dashboard", TrainDashboard)
        start_time = time.time()

        # Stub: simulate training steps for now
        # In production, this would launch a subprocess and parse its output
        step = 0
        while not self._stop_requested.is_set():
            # Check stop conditions
            if stop_condition == "max_steps" and stop_value:
                if step >= int(stop_value):
                    break

            step += 1
            # Simulate a loss value decreasing over time
            loss = 1.0 / (1.0 + step * 0.01)
            delta = abs(loss - 1.0 / (1.0 + (step - 1) * 0.01)) if step > 1 else 0.0

            if stop_condition == "delta_target" and stop_value:
                if delta > 0 and delta < float(stop_value):
                    break

            # Update metrics via the main thread
            try:
                self.app.call_from_thread(dashboard.update_metrics, step, loss, delta)

                # Update timing
                elapsed = time.time() - start_time
                elapsed_str = f"{int(elapsed // 60)}:{int(elapsed % 60):02d}"
                self.app.call_from_thread(dashboard.update_timing, elapsed_str, "-")

                # Update checkpoint info
                if val_every and int(val_every) > 0 and step % int(val_every) == 0:
                    self.app.call_from_thread(
                        dashboard.update_checkpoint,
                        f"Checkpoint saved at step {step}",
                    )
            except Exception:
                break

            self._stop_requested.wait(0.1)

        # Training complete
        try:
            self.app.call_from_thread(
                self._update_status, f"Training complete at step {step}."
            )
        except Exception:
            pass

    def _stop_training(self) -> None:
        """Stop the training process gracefully via SIGINT."""
        self._stop_requested.set()

        if self._training_process is not None:
            try:
                self._training_process.send_signal(signal.SIGINT)
            except (ProcessLookupError, OSError):
                pass
            # Offload blocking wait to a thread to avoid freezing the UI
            proc = self._training_process
            self._training_process = None

            def _wait_and_cleanup():
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    try:
                        proc.terminate()
                    except (ProcessLookupError, OSError):
                        pass

            threading.Thread(target=_wait_and_cleanup, daemon=True).start()

        self._update_status("Training stopped. Checkpoints preserved.")
        self._switch_to_config()

    def _update_status(self, message: str) -> None:
        """Update the app status bar."""
        try:
            status_bar = self.app.query_one("#status-bar", Static)
            status_bar.update(message)
        except Exception:
            pass
