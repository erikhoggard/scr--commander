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
                [("v2", "v2"), ("v1", "v1"), ("discrete", "discrete"), ("causal", "causal")],
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
        self._step_data: list[float] = []

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("Training...", id="dash-title")
            yield Static("Pool: - | Architecture: -", id="dash-info")
            yield Static("Step: 0 | Epoch: 0", id="dash-metrics")
            yield Sparkline([], id="dash-sparkline")
            yield Static("Elapsed: 0:00 | ETA: -", id="dash-timing")
            yield Static("No checkpoints yet", id="dash-checkpoint")
            with Horizontal(classes="action-bar"):
                yield Button("Stop Training", variant="warning", id="stop-training-btn")

    def update_metrics(self, step: int, epoch: int, pct: int) -> None:
        """Update the dashboard metrics display and sparkline."""
        metrics = self.query_one("#dash-metrics", Static)
        metrics.update(f"Step: {step} | Epoch: {epoch} | {pct}%")

        self._step_data.append(float(step))
        sparkline = self.query_one("#dash-sparkline", Sparkline)
        sparkline.data = self._step_data

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
        self._step_data = []
        self.query_one("#dash-title", Label).update("Training...")
        self.query_one("#dash-info", Static).update("Pool: - | Architecture: -")
        self.query_one("#dash-metrics", Static).update("Step: 0 | Epoch: 0 | 0%")
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
        from ..utils.rave_compat import wrap_rave_cmd
        from .rave_parser import parse_training_line
        from .rave_runner import build_preprocess_cmd, build_train_cmd, prepare_samples

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
            pm.get_pool(pool_name)  # Verify pool exists (raises KeyError)
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

        # Preprocessing (skip only if a valid metadata.yaml exists).
        # LMDB creates the directory on open, and RAVE truncates
        # metadata.yaml before writing — so a failed preprocess leaves
        # an empty file.  We must check the *content* is valid.
        preprocess_dir = output_dir / "preprocessed"
        metadata_path = preprocess_dir / "metadata.yaml"
        preprocess_done = False
        if metadata_path.exists():
            try:
                import yaml
                with open(metadata_path) as f:
                    meta = yaml.safe_load(f)
                if isinstance(meta, dict) and "lazy" in meta:
                    preprocess_done = True
            except Exception:
                pass
        if ckpt_path is None and not preprocess_done:
            # Remove stale/incomplete preprocessed directory so LMDB
            # doesn't conflict with the fresh preprocess run.
            if preprocess_dir.exists():
                import shutil
                shutil.rmtree(preprocess_dir, ignore_errors=True)

            # Concatenate short files if needed — RAVE v2 requires
            # num_signal=131072 (~3s) and silently drops shorter files.
            self.app.call_from_thread(
                self._update_status, "Preparing audio..."
            )
            try:
                preprocess_input = prepare_samples(
                    samples_dir, output_dir,
                    pool_dir=pools_dir / pool_name,
                )
            except Exception as e:
                self.app.call_from_thread(
                    self._update_status,
                    f"Error preparing samples: {e}",
                )
                self.app.call_from_thread(self._switch_to_config)
                return

            self.app.call_from_thread(
                self._update_status, "Preprocessing audio..."
            )
            cmd = wrap_rave_cmd(build_preprocess_cmd(
                rave_cmd, preprocess_input, preprocess_dir,
            ))
            try:
                result = subprocess.run(
                    cmd, check=False, capture_output=True, text=True
                )
                if result.returncode != 0:
                    detail = (result.stderr or result.stdout or "")
                    # Show last meaningful line of output
                    detail_lines = [
                        l for l in detail.strip().splitlines() if l.strip()
                    ]
                    hint = detail_lines[-1] if detail_lines else ""
                    self.app.call_from_thread(
                        self._update_status,
                        "Error: Preprocessing failed "
                        f"(exit {result.returncode}). {hint}",
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

        cmd = wrap_rave_cmd(build_train_cmd(
            rave_cmd=rave_cmd,
            config=architecture,
            data_dir=data_dir,
            name=model_name,
            val_every=int(val_every) if val_every else 500,
            max_steps=max_steps,
            gpu=gpu,
            ckpt=ckpt_path,
        ))

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

        # Capture local reference to avoid race with _stop_training
        proc = self._training_process

        # Open a log file so the user can inspect full output
        log_path = output_dir / "train.log"
        log_file = log_path.open("a", encoding="utf-8")

        # Read stdout and parse metrics
        last_step = 0
        # Keep recent unparsed lines so we can show errors on failure
        from collections import deque
        recent_output: deque[str] = deque(maxlen=20)
        try:
            for line in proc.stdout:
                log_file.write(line)
                if self._stop_requested.is_set():
                    break

                parsed = parse_training_line(line)
                if parsed is None:
                    stripped = line.strip()
                    if stripped:
                        recent_output.append(stripped)
                    continue

                if parsed.get("checkpoint"):
                    self.app.call_from_thread(
                        dashboard.update_checkpoint,
                        f"Checkpoint saved at step {last_step}",
                    )
                    continue

                if parsed.get("validation"):
                    continue

                step = parsed.get("step", last_step)
                epoch = parsed.get("epoch", 0)
                pct = parsed.get("pct", 0)

                # Only update on meaningful progress (not every duplicate line)
                if step == last_step:
                    continue

                last_step = step

                try:
                    self.app.call_from_thread(
                        dashboard.update_metrics, step, epoch, pct
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

        except Exception:
            pass
        finally:
            log_file.close()

        # Wait for process to finish (use local ref to avoid race)
        try:
            returncode = proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            returncode = None
        self._training_process = None

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
            # Export model so it appears on the Generate tab
            self.app.call_from_thread(
                self._update_status,
                f"Training complete. Exporting model...",
            )

            export_success = self._export_model(
                rave_cmd, output_dir, run_dir, model_name, pool_name, architecture,
            )

            run_info.status = "completed"
            save_training_run(run_info, run_dir)

            if export_success:
                self.app.call_from_thread(
                    self._update_status,
                    f"Model '{model_name}' ready on Generate tab.",
                )
            else:
                self.app.call_from_thread(
                    self._update_status,
                    f"Training complete but export failed. "
                    f"Run 'scropipe export {model_name}' manually.",
                )
        else:
            run_info.status = "paused"
            save_training_run(run_info, run_dir)
            # Include last lines of output so the user can see why it failed
            error_detail = ""
            if recent_output:
                error_detail = " | " + " ".join(
                    list(recent_output)[-3:]
                )
            self.app.call_from_thread(
                self._update_status,
                f"Training failed (exit {returncode}). "
                f"See {log_path}{error_detail}",
            )

        self.app.call_from_thread(self._switch_to_config)

    def _export_model(
        self,
        rave_cmd: str,
        output_dir: Path,
        run_dir: Path,
        model_name: str,
        pool_name: str,
        architecture: str,
    ) -> bool:
        """Export trained model to model.ts + metadata.json.

        Returns True on success, False on failure.
        """
        import json
        import shutil
        from datetime import datetime

        from ..utils.rave_compat import wrap_rave_cmd
        from .rave_runner import build_export_cmd

        # RAVE export needs the run dir containing config.gin, not the
        # top-level training_output dir.
        config_files = list(output_dir.glob("**/config.gin"))
        if not config_files:
            return False
        rave_run_dir = config_files[0].parent

        cmd = wrap_rave_cmd(build_export_cmd(rave_cmd, str(rave_run_dir)))
        try:
            result = subprocess.run(cmd, check=False, capture_output=True, text=True)
            if result.returncode != 0:
                return False
        except Exception:
            return False

        # Find exported .ts file
        ts_files = list(rave_run_dir.glob("*.ts")) or list(output_dir.glob("**/*.ts"))
        if not ts_files:
            return False

        try:
            # Copy to model.ts in run_dir
            final_path = run_dir / "model.ts"
            shutil.copy2(ts_files[0], final_path)

            # Write metadata.json
            metadata = {
                "name": model_name,
                "created": datetime.now().isoformat(),
                "config": architecture,
                "total_samples": 0,
                "pool_name": pool_name,
            }
            metadata_path = run_dir / "metadata.json"
            metadata_path.write_text(json.dumps(metadata, indent=2))
        except Exception:
            return False

        return True

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
