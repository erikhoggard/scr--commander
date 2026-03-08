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
                yield Button("Resume Training", variant="default", id="resume-training-btn")

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
        """Hide dashboard on initial mount and reconcile stale runs."""
        self.query_one("#train-dashboard", TrainDashboard).display = False
        models_dir = getattr(self.app, "models_dir", None)
        if models_dir is not None:
            from ..training_state import reconcile_stale_runs

            reconcile_stale_runs(models_dir)

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
        elif event.button.id == "resume-training-btn":
            self._resume_training()
        elif event.button.id == "stop-training-btn":
            self._stop_training()

    def _resume_training(self) -> None:
        """Show resumable runs and resume the selected one."""
        from ..training_state import find_checkpoint_dir, list_paused_runs

        models_dir = getattr(self.app, "models_dir", None)
        if models_dir is None:
            self._update_status("Error: Models directory not configured.")
            return

        paused = list_paused_runs(models_dir)
        if not paused:
            self._update_status("No paused training runs found.")
            return

        # Resume the most recent paused run
        run = paused[-1]
        output_dir = Path(run.output_dir)

        ckpt_dir = find_checkpoint_dir(output_dir)
        if ckpt_dir is None:
            self._update_status(f"No checkpoints found for {run.model_name}.")
            return

        # Set up dashboard
        dashboard = self.query_one("#train-dashboard", TrainDashboard)
        dashboard.reset()
        dashboard.set_title(f"{run.model_name} (resumed)")
        dashboard.set_info(run.pool_name, run.architecture)

        # Switch views
        self._switch_to_dashboard()
        self._update_status(f"Resuming {run.model_name}...")

        # Start the training worker thread with checkpoint
        self._stop_requested.clear()
        self._training_thread = threading.Thread(
            target=self._training_worker,
            args=(run.pool_name, run.model_name, run.architecture, "manual", None, "500"),
            kwargs={"ckpt_path": ckpt_dir},
            daemon=True,
        )
        self._training_thread.start()

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
        ckpt_path: Optional[Path] = None,
    ) -> None:
        """Worker thread that runs the RAVE training subprocess."""
        from ..pool_manager import PoolManager
        from ..training_state import TrainingRunInfo, save_training_run
        from ..utils.discovery import ToolNotFoundError, find_tool
        from .rave_parser import parse_training_line
        from .rave_runner import build_preprocess_cmd, build_train_cmd

        dashboard = self.query_one("#train-dashboard", TrainDashboard)
        start_time = time.time()
        models_dir = getattr(self.app, "models_dir", None)

        # Find rave
        try:
            rave_cmd = str(find_tool("rave"))
        except ToolNotFoundError:
            self.app.call_from_thread(
                self._update_status,
                "Error: RAVE not found. "
                "Set RAVE_PATH or add rave to PATH.",
            )
            self.app.call_from_thread(self._switch_to_config)
            return

        # Resolve pool samples directory
        pools_dir = getattr(self.app, "pools_dir", None)
        if pools_dir is None:
            self.app.call_from_thread(
                self._update_status,
                "Error: Pools directory not configured.",
            )
            self.app.call_from_thread(self._switch_to_config)
            return

        pm = PoolManager(pools_dir)
        try:
            samples_dir = pm.get_samples_dir(pool_name)
        except KeyError:
            self.app.call_from_thread(
                self._update_status,
                f"Error: Pool '{pool_name}' not found.",
            )
            self.app.call_from_thread(self._switch_to_config)
            return

        # Set up output directory
        if models_dir is None:
            self.app.call_from_thread(
                self._update_status,
                "Error: Models directory not configured.",
            )
            self.app.call_from_thread(self._switch_to_config)
            return

        run_dir = models_dir / model_name
        run_dir.mkdir(parents=True, exist_ok=True)
        output_dir = run_dir / "training_output"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Preprocessing (skip if already done or resuming)
        preprocess_dir = output_dir / "preprocessed"
        if ckpt_path is None and not preprocess_dir.exists():
            self.app.call_from_thread(
                self._update_status, "Preprocessing audio..."
            )
            cmd = build_preprocess_cmd(
                rave_cmd, samples_dir, preprocess_dir
            )
            try:
                result = subprocess.run(
                    cmd, check=False, capture_output=True, text=True
                )
                if result.returncode != 0:
                    self.app.call_from_thread(
                        self._update_status,
                        "Error: Preprocessing failed "
                        f"(exit {result.returncode}).",
                    )
                    self.app.call_from_thread(self._switch_to_config)
                    return
            except Exception as e:
                self.app.call_from_thread(
                    self._update_status, f"Error: {e}"
                )
                self.app.call_from_thread(self._switch_to_config)
                return

        data_dir = preprocess_dir

        # Build train command
        max_steps = None
        if stop_condition == "max_steps" and stop_value:
            max_steps = int(stop_value)

        gpu = None
        try:
            import torch

            if torch.cuda.is_available():
                gpu = 0
        except ImportError:
            pass

        cmd = build_train_cmd(
            rave_cmd=rave_cmd,
            config=architecture,
            data_dir=data_dir,
            name=model_name,
            val_every=int(val_every) if val_every else 500,
            max_steps=max_steps,
            gpu=gpu,
            ckpt=ckpt_path,
        )

        # Save sidecar
        run_info = TrainingRunInfo(
            model_name=model_name,
            pool_name=pool_name,
            architecture=architecture,
            output_dir=str(output_dir),
            status="training",
        )
        save_training_run(run_info, run_dir)

        # Launch training subprocess
        self.app.call_from_thread(
            self._update_status, f"Training {model_name}..."
        )
        try:
            self._training_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(output_dir),
            )
        except Exception as e:
            self.app.call_from_thread(
                self._update_status,
                f"Error: Failed to start training: {e}",
            )
            self.app.call_from_thread(self._switch_to_config)
            run_info.status = "paused"
            save_training_run(run_info, run_dir)
            return

        # Read stdout and parse metrics
        last_step = 0
        prev_loss = 0.0
        try:
            for line in self._training_process.stdout:
                if self._stop_requested.is_set():
                    break

                parsed = parse_training_line(line)
                if parsed is None:
                    continue

                if parsed.get("checkpoint"):
                    self.app.call_from_thread(
                        dashboard.update_checkpoint,
                        f"Checkpoint saved at step {last_step}",
                    )
                    continue

                step = parsed.get("step", last_step)
                loss = parsed.get("loss", 0.0)
                delta = (
                    abs(loss - prev_loss)
                    if prev_loss > 0
                    else 0.0
                )

                last_step = step
                prev_loss = loss

                try:
                    self.app.call_from_thread(
                        dashboard.update_metrics, step, loss, delta
                    )

                    elapsed = time.time() - start_time
                    elapsed_str = (
                        f"{int(elapsed // 3600)}:"
                        f"{int(elapsed % 3600 // 60):02d}"
                        f":{int(elapsed % 60):02d}"
                    )
                    self.app.call_from_thread(
                        dashboard.update_timing, elapsed_str, "-"
                    )
                except Exception:
                    break

                # Check delta stop condition
                if stop_condition == "delta_target" and stop_value:
                    if delta > 0 and delta < float(stop_value):
                        self._stop_requested.set()
                        break

        except Exception:
            pass

        # Wait for process to finish
        if self._training_process is not None:
            try:
                returncode = self._training_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                returncode = None
            self._training_process = None
        else:
            returncode = None

        # Update sidecar status
        if self._stop_requested.is_set():
            run_info.status = "paused"
            save_training_run(run_info, run_dir)
            self.app.call_from_thread(
                self._update_status,
                f"Training paused at step {last_step}. "
                "Checkpoints preserved.",
            )
        elif returncode == 0:
            run_info.status = "completed"
            save_training_run(run_info, run_dir)
            self.app.call_from_thread(
                self._update_status,
                f"Training complete at step {last_step}.",
            )
        else:
            run_info.status = "paused"
            save_training_run(run_info, run_dir)
            self.app.call_from_thread(
                self._update_status,
                f"Training stopped (exit {returncode}). "
                "Checkpoints preserved.",
            )

        self.app.call_from_thread(self._switch_to_config)

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
