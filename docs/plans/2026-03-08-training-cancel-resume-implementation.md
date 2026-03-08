# Training Cancel/Resume Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enable cancelling and resuming RAVE training in the TUI via graceful SIGINT shutdown and checkpoint-based resume.

**Architecture:** Replace the stub training worker with a real subprocess manager that launches `rave preprocess` + `rave train`, parses stdout for metrics, handles graceful shutdown via SIGINT, and tracks training state in a JSON sidecar file. Add a Resume button to the config panel.

**Tech Stack:** Python, Textual (TUI), subprocess/Popen, PyTorch Lightning (RAVE's framework), JSON sidecar files

---

### Task 1: Add TrainingRunInfo sidecar data model

**Files:**
- Create: `scropipe/training_state.py`
- Test: `tests/test_training_state.py`

This module manages the JSON sidecar files that track training run state. Each training run gets a `training_run.json` in a dedicated directory under `models_dir`.

**Step 1: Write the failing test**

```python
# tests/test_training_state.py
import json
from scropipe.training_state import TrainingRunInfo, save_training_run, load_training_run, list_paused_runs


def test_save_and_load_training_run(tmp_path):
    run = TrainingRunInfo(
        model_name="my-model",
        pool_name="drums",
        architecture="v2",
        output_dir=str(tmp_path / "output"),
        status="training",
    )
    run_dir = tmp_path / "models" / "my-model"
    save_training_run(run, run_dir)
    loaded = load_training_run(run_dir)
    assert loaded.model_name == "my-model"
    assert loaded.pool_name == "drums"
    assert loaded.architecture == "v2"
    assert loaded.status == "training"


def test_list_paused_runs_filters_by_status(tmp_path):
    models_dir = tmp_path / "models"

    # Create a paused run
    paused_dir = models_dir / "paused-model"
    paused_dir.mkdir(parents=True)
    run1 = TrainingRunInfo(
        model_name="paused-model",
        pool_name="drums",
        architecture="v2",
        output_dir=str(tmp_path / "out1"),
        status="paused",
    )
    save_training_run(run1, paused_dir)

    # Create a completed run
    done_dir = models_dir / "done-model"
    done_dir.mkdir(parents=True)
    run2 = TrainingRunInfo(
        model_name="done-model",
        pool_name="drums",
        architecture="v2",
        output_dir=str(tmp_path / "out2"),
        status="completed",
    )
    save_training_run(run2, done_dir)

    paused = list_paused_runs(models_dir)
    assert len(paused) == 1
    assert paused[0].model_name == "paused-model"


def test_list_paused_runs_empty_dir(tmp_path):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    assert list_paused_runs(models_dir) == []


def test_list_paused_runs_nonexistent_dir(tmp_path):
    assert list_paused_runs(tmp_path / "nope") == []
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_training_state.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scropipe.training_state'`

**Step 3: Write minimal implementation**

```python
# scropipe/training_state.py
"""Training run state tracking via JSON sidecar files."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


@dataclass
class TrainingRunInfo:
    """Metadata about a training run, persisted as training_run.json."""

    model_name: str
    pool_name: str
    architecture: str
    output_dir: str
    status: str  # "training", "paused", "completed"
    started: str = ""

    def __post_init__(self):
        if not self.started:
            self.started = datetime.now(timezone.utc).isoformat()


def save_training_run(run: TrainingRunInfo, run_dir: Path) -> None:
    """Write training run info to run_dir/training_run.json."""
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "training_run.json"
    path.write_text(json.dumps(asdict(run), indent=2))


def load_training_run(run_dir: Path) -> Optional[TrainingRunInfo]:
    """Load training run info from run_dir/training_run.json."""
    path = run_dir / "training_run.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return TrainingRunInfo(**data)


def list_paused_runs(models_dir: Path) -> list[TrainingRunInfo]:
    """Find all training runs with status 'paused' or 'training' (stale)."""
    if not models_dir.exists():
        return []
    runs = []
    for d in sorted(models_dir.iterdir()):
        if not d.is_dir():
            continue
        run = load_training_run(d)
        if run is not None and run.status in ("paused", "training"):
            runs.append(run)
    return runs
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_training_state.py -v`
Expected: All 4 tests PASS

**Step 5: Commit**

```bash
git add scropipe/training_state.py tests/test_training_state.py
git commit -m "feat: add training run state tracking module"
```

---

### Task 2: Update dashboard UI — single Stop button

**Files:**
- Modify: `scropipe/tui/train_tab.py:140-151` (TrainDashboard.compose)
- Modify: `scropipe/tui/train_tab.py:225-232` (TrainTab.on_button_pressed)
- Modify: `scropipe/tui/train_tab.py:359-376` (TrainTab._stop_training)
- Test: `tests/test_tui_train.py`

Replace the "Stop & Save" / "Stop & Discard" buttons with a single "Stop Training" button.

**Step 1: Write the failing test**

Add to `tests/test_tui_train.py`:

```python
@pytest.mark.asyncio
async def test_train_tab_dashboard_has_single_stop_button(app):
    async with app.run_test() as pilot:
        tabbed = app.query_one("TabbedContent")
        tabbed.active = "tab-train"
        await pilot.pause()
        assert app.query_one("#stop-training-btn") is not None
        assert len(app.query("#stop-save-btn")) == 0
        assert len(app.query("#stop-discard-btn")) == 0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_tui_train.py::test_train_tab_dashboard_has_single_stop_button -v`
Expected: FAIL — `#stop-training-btn` not found, `#stop-save-btn` still exists

**Step 3: Modify TrainDashboard and TrainTab**

In `train_tab.py`, `TrainDashboard.compose()`, replace lines 148-150:

```python
            with Horizontal(classes="action-bar"):
                yield Button("Stop Training", variant="warning", id="stop-training-btn")
```

In `TrainTab.on_button_pressed()`, replace the stop button handlers (lines 229-232):

```python
        elif event.button.id == "stop-training-btn":
            self._stop_training()
```

Simplify `_stop_training()` (lines 359-376) — remove the `save` parameter:

```python
    def _stop_training(self) -> None:
        """Stop the training process gracefully via SIGINT."""
        self._stop_requested.set()

        if self._training_process is not None:
            try:
                import signal
                self._training_process.send_signal(signal.SIGINT)
            except (ProcessLookupError, OSError):
                pass
            # Wait up to 10s for graceful shutdown, then SIGTERM
            try:
                self._training_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    self._training_process.terminate()
                except (ProcessLookupError, OSError):
                    pass
            self._training_process = None

        self._update_status("Training stopped. Checkpoints preserved.")
        self._switch_to_config()
```

**Step 4: Update the existing test that references old buttons**

In `tests/test_tui_train.py`, update `test_train_tab_dashboard_has_controls` to reference `#stop-training-btn` instead of `#stop-save-btn` and `#stop-discard-btn`:

```python
@pytest.mark.asyncio
async def test_train_tab_dashboard_has_controls(app):
    async with app.run_test() as pilot:
        tabbed = app.query_one("TabbedContent")
        tabbed.active = "tab-train"
        await pilot.pause()
        assert app.query_one("#dash-title") is not None
        assert app.query_one("#dash-info") is not None
        assert app.query_one("#dash-metrics") is not None
        assert app.query_one("#dash-sparkline") is not None
        assert app.query_one("#dash-timing") is not None
        assert app.query_one("#dash-checkpoint") is not None
        assert app.query_one("#stop-training-btn") is not None
```

**Step 5: Run tests to verify**

Run: `pytest tests/test_tui_train.py -v`
Expected: All tests PASS

**Step 6: Commit**

```bash
git add scropipe/tui/train_tab.py tests/test_tui_train.py
git commit -m "feat: replace save/discard buttons with single stop button"
```

---

### Task 3: Add Resume Training button to config panel

**Files:**
- Modify: `scropipe/tui/train_tab.py:30-70` (TrainConfigPanel.compose)
- Modify: `scropipe/tui/train_tab.py:225-232` (TrainTab.on_button_pressed)
- Test: `tests/test_tui_train.py`

Add a "Resume Training" button next to "Start Training".

**Step 1: Write the failing test**

Add to `tests/test_tui_train.py`:

```python
@pytest.mark.asyncio
async def test_train_tab_has_resume_button(app):
    async with app.run_test() as pilot:
        tabbed = app.query_one("TabbedContent")
        tabbed.active = "tab-train"
        await pilot.pause()
        btn = app.query_one("#resume-training-btn")
        assert btn is not None
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_tui_train.py::test_train_tab_has_resume_button -v`
Expected: FAIL — `#resume-training-btn` not found

**Step 3: Add Resume button to TrainConfigPanel.compose()**

In `train_tab.py`, modify the action-bar in `TrainConfigPanel.compose()` (line 69-70):

```python
            with Horizontal(classes="action-bar"):
                yield Button("Start Training", variant="primary", id="start-training-btn")
                yield Button("Resume Training", variant="default", id="resume-training-btn")
```

Add handler in `TrainTab.on_button_pressed()`:

```python
        elif event.button.id == "resume-training-btn":
            self._resume_training()
```

Add stub `_resume_training()` method to `TrainTab`:

```python
    def _resume_training(self) -> None:
        """Show resumable runs and resume selected one."""
        from ..training_state import list_paused_runs

        models_dir = getattr(self.app, "models_dir", None)
        if models_dir is None:
            self._update_status("Error: Models directory not configured.")
            return

        paused = list_paused_runs(models_dir)
        if not paused:
            self._update_status("No paused training runs found.")
            return

        # For now, resume the most recent paused run
        run = paused[-1]
        self._update_status(f"Resuming {run.model_name}...")
        # TODO: Task 6 will implement the full resume flow
```

**Step 4: Run tests to verify**

Run: `pytest tests/test_tui_train.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add scropipe/tui/train_tab.py tests/test_tui_train.py
git commit -m "feat: add resume training button to config panel"
```

---

### Task 4: Build RAVE command constructor

**Files:**
- Create: `scropipe/tui/rave_runner.py`
- Test: `tests/test_rave_runner.py`

Extract the logic for building the `rave preprocess` and `rave train` command lines into a testable module that the TUI worker will use. This avoids duplicating the command construction logic from `stages/rave.py` and makes it testable without actually running RAVE.

**Step 1: Write the failing test**

```python
# tests/test_rave_runner.py
from pathlib import Path
from scropipe.tui.rave_runner import build_preprocess_cmd, build_train_cmd


def test_build_preprocess_cmd():
    cmd = build_preprocess_cmd(
        rave_cmd="/usr/bin/rave",
        input_dir=Path("/data/samples"),
        output_dir=Path("/data/preprocessed"),
    )
    assert cmd == [
        "/usr/bin/rave", "preprocess",
        "--input_path", "/data/samples",
        "--output_path", "/data/preprocessed",
        "--num_signal", "131072",
    ]


def test_build_train_cmd_basic():
    cmd = build_train_cmd(
        rave_cmd="/usr/bin/rave",
        config="v2",
        data_dir=Path("/data/preprocessed"),
        name="my-model",
        val_every=500,
    )
    assert "/usr/bin/rave" in cmd
    assert "train" in cmd
    assert "--config" in cmd
    assert "v2" in cmd
    assert "--val_every" in cmd
    assert "500" in cmd


def test_build_train_cmd_with_checkpoint():
    cmd = build_train_cmd(
        rave_cmd="/usr/bin/rave",
        config="v2",
        data_dir=Path("/data/preprocessed"),
        name="my-model",
        val_every=500,
        ckpt=Path("/checkpoints/best.ckpt"),
    )
    assert "--ckpt" in cmd
    assert "/checkpoints/best.ckpt" in cmd


def test_build_train_cmd_with_max_steps():
    cmd = build_train_cmd(
        rave_cmd="/usr/bin/rave",
        config="v2",
        data_dir=Path("/data/preprocessed"),
        name="my-model",
        val_every=500,
        max_steps=10000,
    )
    assert "--max_steps" in cmd
    assert "10000" in cmd


def test_build_train_cmd_with_gpu():
    cmd = build_train_cmd(
        rave_cmd="/usr/bin/rave",
        config="v2",
        data_dir=Path("/data/preprocessed"),
        name="my-model",
        val_every=500,
        gpu=0,
    )
    assert "--gpu" in cmd
    assert "0" in cmd
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_rave_runner.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write the implementation**

```python
# scropipe/tui/rave_runner.py
"""Build RAVE CLI commands for the TUI training worker."""

from __future__ import annotations

from pathlib import Path
from typing import Optional


def build_preprocess_cmd(
    rave_cmd: str,
    input_dir: Path,
    output_dir: Path,
    num_signal: int = 131072,
) -> list[str]:
    """Build the rave preprocess command."""
    return [
        rave_cmd, "preprocess",
        "--input_path", str(input_dir),
        "--output_path", str(output_dir),
        "--num_signal", str(num_signal),
    ]


def build_train_cmd(
    rave_cmd: str,
    config: str,
    data_dir: Path,
    name: str,
    val_every: int = 500,
    max_steps: Optional[int] = None,
    gpu: Optional[int] = None,
    workers: int = 0,
    n_signal: int = 131072,
    ckpt: Optional[Path] = None,
) -> list[str]:
    """Build the rave train command."""
    cmd = [
        rave_cmd, "train",
        "--config", config,
        "--db_path", str(data_dir),
        "--name", name,
        "--n_signal", str(n_signal),
        "--workers", str(workers),
        "--val_every", str(val_every),
    ]
    if gpu is not None:
        cmd.extend(["--gpu", str(gpu)])
    if max_steps is not None:
        cmd.extend(["--max_steps", str(max_steps)])
    if ckpt is not None:
        cmd.extend(["--ckpt", str(ckpt)])
    return cmd
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_rave_runner.py -v`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add scropipe/tui/rave_runner.py tests/test_rave_runner.py
git commit -m "feat: add RAVE command builder for TUI worker"
```

---

### Task 5: Build RAVE stdout metric parser

**Files:**
- Create: `scropipe/tui/rave_parser.py`
- Test: `tests/test_rave_parser.py`

Parse RAVE/Lightning training output lines to extract step number, loss values, and checkpoint events. PyTorch Lightning 1.9 outputs progress via tqdm-style bars and periodic log lines.

**Step 1: Write the failing test**

```python
# tests/test_rave_parser.py
from scropipe.tui.rave_parser import parse_training_line


def test_parse_epoch_line():
    # Lightning 1.9 style progress line
    line = "Epoch 0:  50%|█████     | 500/1000 [01:23<01:23, 6.00it/s, loss=0.123]"
    result = parse_training_line(line)
    assert result is not None
    assert result["step"] == 500
    assert abs(result["loss"] - 0.123) < 1e-6


def test_parse_epoch_line_v2():
    # Another common Lightning format
    line = "Epoch 1:  10%|█         | 100/1000 [00:30<04:30, 3.33it/s, loss=0.456]"
    result = parse_training_line(line)
    assert result is not None
    assert result["step"] == 100


def test_parse_validation_line():
    line = "Validation: 100%|██████████| 10/10 [00:05<00:00, 2.00it/s]"
    result = parse_training_line(line)
    # Validation lines don't have loss — return None
    assert result is None


def test_parse_checkpoint_line():
    line = "Epoch 0, global step 500: 'valid' reached 0.12300 (best 0.12300), saving model to '/runs/model/version_0/checkpoints/best.ckpt'"
    result = parse_training_line(line)
    assert result is not None
    assert result.get("checkpoint") is True


def test_parse_unrelated_line():
    line = "GPU available: True (cuda), used: True"
    result = parse_training_line(line)
    assert result is None


def test_parse_step_log_line():
    # Some Lightning configs log step-level metrics
    line = "Step 1500: loss=0.0891"
    result = parse_training_line(line)
    assert result is not None
    assert result["step"] == 1500


def test_parse_empty_line():
    assert parse_training_line("") is None
    assert parse_training_line("\n") is None
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_rave_parser.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write the implementation**

```python
# scropipe/tui/rave_parser.py
"""Parse RAVE/PyTorch Lightning training output for metrics."""

from __future__ import annotations

import re
from typing import Optional


# Epoch progress: "Epoch 0:  50%|███| 500/1000 [01:23<01:23, 6.00it/s, loss=0.123]"
_EPOCH_RE = re.compile(
    r"Epoch\s+\d+:\s+\d+%\|[^|]*\|\s*(\d+)/\d+\s*\[.*?loss=([\d.]+)"
)

# Step-level log: "Step 1500: loss=0.0891"
_STEP_RE = re.compile(
    r"Step\s+(\d+):\s*loss=([\d.]+)"
)

# Checkpoint save: "saving model to '...'"
_CKPT_RE = re.compile(
    r"saving model to"
)


def parse_training_line(line: str) -> Optional[dict]:
    """Parse a single line of RAVE training output.

    Returns:
        Dict with parsed info (step, loss, checkpoint), or None if
        the line doesn't contain training metrics.
    """
    line = line.strip()
    if not line:
        return None

    # Check for checkpoint save
    if _CKPT_RE.search(line):
        return {"checkpoint": True}

    # Check epoch progress
    m = _EPOCH_RE.search(line)
    if m:
        return {
            "step": int(m.group(1)),
            "loss": float(m.group(2)),
        }

    # Check step-level log
    m = _STEP_RE.search(line)
    if m:
        return {
            "step": int(m.group(1)),
            "loss": float(m.group(2)),
        }

    return None
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_rave_parser.py -v`
Expected: All 7 tests PASS

**Step 5: Commit**

```bash
git add scropipe/tui/rave_parser.py tests/test_rave_parser.py
git commit -m "feat: add RAVE stdout metric parser"
```

---

### Task 6: Replace stub training worker with real RAVE subprocess

**Files:**
- Modify: `scropipe/tui/train_tab.py:292-358` (_training_worker method)
- Modify: `scropipe/tui/train_tab.py:1-14` (imports)
- Test: `tests/test_rave_runner.py` (add integration-style test)

This is the core task: replace the simulated training worker with real subprocess management.

**Step 1: Write a test for the worker's sidecar lifecycle**

Add to `tests/test_training_state.py`:

```python
def test_status_transitions(tmp_path):
    """Verify the sidecar status transitions work correctly."""
    run_dir = tmp_path / "models" / "test-model"
    run = TrainingRunInfo(
        model_name="test-model",
        pool_name="drums",
        architecture="v2",
        output_dir=str(tmp_path / "output"),
        status="training",
    )

    # Start -> training
    save_training_run(run, run_dir)
    loaded = load_training_run(run_dir)
    assert loaded.status == "training"

    # Cancel -> paused
    run.status = "paused"
    save_training_run(run, run_dir)
    loaded = load_training_run(run_dir)
    assert loaded.status == "paused"

    # Resume -> training
    run.status = "training"
    save_training_run(run, run_dir)
    loaded = load_training_run(run_dir)
    assert loaded.status == "training"

    # Complete -> completed
    run.status = "completed"
    save_training_run(run, run_dir)
    loaded = load_training_run(run_dir)
    assert loaded.status == "completed"

    # completed is not listed as paused
    assert list_paused_runs(tmp_path / "models") == []
```

**Step 2: Run test to verify it passes** (this uses already-implemented code)

Run: `pytest tests/test_training_state.py::test_status_transitions -v`
Expected: PASS

**Step 3: Replace _training_worker in train_tab.py**

Update imports at top of `train_tab.py`:

```python
import os
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Input, Label, RadioButton, RadioSet, Select, Sparkline, Static
```

Replace `_training_worker` method entirely:

```python
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
        from ..utils.discovery import find_tool, ToolNotFoundError
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
                "Error: RAVE not found. Set RAVE_PATH or add rave to PATH.",
            )
            self.app.call_from_thread(self._switch_to_config)
            return

        # Resolve pool samples directory
        pools_dir = getattr(self.app, "pools_dir", None)
        if pools_dir is None:
            self.app.call_from_thread(self._update_status, "Error: Pools directory not configured.")
            self.app.call_from_thread(self._switch_to_config)
            return

        pm = PoolManager(pools_dir)
        try:
            samples_dir = pm.get_samples_dir(pool_name)
        except KeyError:
            self.app.call_from_thread(self._update_status, f"Error: Pool '{pool_name}' not found.")
            self.app.call_from_thread(self._switch_to_config)
            return

        # Set up output directory
        if models_dir is None:
            self.app.call_from_thread(self._update_status, "Error: Models directory not configured.")
            self.app.call_from_thread(self._switch_to_config)
            return

        run_dir = models_dir / model_name
        run_dir.mkdir(parents=True, exist_ok=True)
        output_dir = run_dir / "training_output"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Preprocessing (skip if already done or resuming)
        preprocess_dir = output_dir / "preprocessed"
        if ckpt_path is None and not preprocess_dir.exists():
            self.app.call_from_thread(self._update_status, "Preprocessing audio...")
            cmd = build_preprocess_cmd(rave_cmd, samples_dir, preprocess_dir)
            try:
                result = subprocess.run(cmd, check=False, capture_output=True, text=True)
                if result.returncode != 0:
                    self.app.call_from_thread(
                        self._update_status,
                        f"Error: Preprocessing failed (exit {result.returncode}).",
                    )
                    self.app.call_from_thread(self._switch_to_config)
                    return
            except Exception as e:
                self.app.call_from_thread(self._update_status, f"Error: {e}")
                self.app.call_from_thread(self._switch_to_config)
                return
        elif ckpt_path is None:
            preprocess_dir = preprocess_dir  # already exists

        # Determine data_dir (use existing preprocessed data)
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
        self.app.call_from_thread(self._update_status, f"Training {model_name}...")
        try:
            self._training_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(output_dir),
            )
        except Exception as e:
            self.app.call_from_thread(self._update_status, f"Error: Failed to start training: {e}")
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
                delta = abs(loss - prev_loss) if prev_loss > 0 else 0.0

                last_step = step
                prev_loss = loss

                try:
                    self.app.call_from_thread(dashboard.update_metrics, step, loss, delta)

                    elapsed = time.time() - start_time
                    elapsed_str = f"{int(elapsed // 3600)}:{int(elapsed % 3600 // 60):02d}:{int(elapsed % 60):02d}"
                    self.app.call_from_thread(dashboard.update_timing, elapsed_str, "-")
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
                f"Training paused at step {last_step}. Checkpoints preserved.",
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
                f"Training stopped unexpectedly (exit {returncode}). Checkpoints preserved.",
            )

        self.app.call_from_thread(self._switch_to_config)
```

**Step 4: Run tests to verify nothing broke**

Run: `pytest tests/test_tui_train.py tests/test_training_state.py tests/test_rave_runner.py tests/test_rave_parser.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add scropipe/tui/train_tab.py
git commit -m "feat: replace stub training worker with real RAVE subprocess"
```

---

### Task 7: Wire up resume flow

**Files:**
- Modify: `scropipe/tui/train_tab.py` (_resume_training method)
- Test: `tests/test_training_state.py` (add checkpoint discovery test)

Complete the resume flow: find checkpoints in the output directory and relaunch training with `--ckpt`.

**Step 1: Write the failing test**

Add to `tests/test_training_state.py`:

```python
from scropipe.training_state import find_checkpoint_dir


def test_find_checkpoint_dir(tmp_path):
    """Verify checkpoint directory discovery finds RAVE checkpoint dirs."""
    output_dir = tmp_path / "training_output"
    ckpt_dir = output_dir / "runs" / "model_abc123" / "version_0" / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "best.ckpt").write_text("fake")

    result = find_checkpoint_dir(output_dir)
    assert result is not None
    assert result.name == "checkpoints"
    assert (result / "best.ckpt").exists()


def test_find_checkpoint_dir_no_checkpoints(tmp_path):
    output_dir = tmp_path / "training_output"
    output_dir.mkdir()
    result = find_checkpoint_dir(output_dir)
    assert result is None


def test_find_checkpoint_dir_nonexistent(tmp_path):
    result = find_checkpoint_dir(tmp_path / "nope")
    assert result is None
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_training_state.py::test_find_checkpoint_dir -v`
Expected: FAIL with `ImportError`

**Step 3: Add find_checkpoint_dir to training_state.py**

Add to `scropipe/training_state.py`:

```python
def find_checkpoint_dir(output_dir: Path) -> Optional[Path]:
    """Find the RAVE checkpoint directory inside a training output dir.

    Searches for runs/*/version_*/checkpoints/ containing .ckpt files.
    Returns the checkpoint directory path, or None if not found.
    """
    if not output_dir.exists():
        return None
    ckpt_files = list(output_dir.rglob("*.ckpt"))
    if not ckpt_files:
        return None
    # Return the parent directory of the most recent checkpoint
    newest = max(ckpt_files, key=lambda p: p.stat().st_mtime)
    return newest.parent
```

**Step 4: Complete _resume_training in train_tab.py**

Replace the stub `_resume_training` method:

```python
    def _resume_training(self) -> None:
        """Show resumable runs and resume the selected one."""
        from ..training_state import list_paused_runs, find_checkpoint_dir

        models_dir = getattr(self.app, "models_dir", None)
        if models_dir is None:
            self._update_status("Error: Models directory not configured.")
            return

        paused = list_paused_runs(models_dir)
        if not paused:
            self._update_status("No paused training runs found.")
            return

        # Resume the most recent paused run
        # (Future: could show a selection dialog)
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
```

**Step 5: Run tests to verify**

Run: `pytest tests/test_training_state.py tests/test_tui_train.py -v`
Expected: All tests PASS

**Step 6: Commit**

```bash
git add scropipe/training_state.py scropipe/tui/train_tab.py tests/test_training_state.py
git commit -m "feat: wire up resume training flow with checkpoint discovery"
```

---

### Task 8: Handle stale "training" status on TUI launch

**Files:**
- Modify: `scropipe/tui/train_tab.py:211-213` (TrainTab.on_mount)
- Test: `tests/test_training_state.py`

When the TUI launches, detect any runs with status "training" (meaning TUI exited without graceful stop) and update them to "paused".

**Step 1: Write the failing test**

Add to `tests/test_training_state.py`:

```python
from scropipe.training_state import reconcile_stale_runs


def test_reconcile_stale_runs(tmp_path):
    """Stale 'training' runs should be marked as 'paused'."""
    models_dir = tmp_path / "models"
    run_dir = models_dir / "stale-model"
    run_dir.mkdir(parents=True)
    run = TrainingRunInfo(
        model_name="stale-model",
        pool_name="drums",
        architecture="v2",
        output_dir=str(tmp_path / "out"),
        status="training",
    )
    save_training_run(run, run_dir)

    reconcile_stale_runs(models_dir)

    loaded = load_training_run(run_dir)
    assert loaded.status == "paused"


def test_reconcile_stale_runs_leaves_paused(tmp_path):
    """Already-paused runs stay paused."""
    models_dir = tmp_path / "models"
    run_dir = models_dir / "paused-model"
    run_dir.mkdir(parents=True)
    run = TrainingRunInfo(
        model_name="paused-model",
        pool_name="drums",
        architecture="v2",
        output_dir=str(tmp_path / "out"),
        status="paused",
    )
    save_training_run(run, run_dir)

    reconcile_stale_runs(models_dir)

    loaded = load_training_run(run_dir)
    assert loaded.status == "paused"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_training_state.py::test_reconcile_stale_runs -v`
Expected: FAIL with `ImportError`

**Step 3: Add reconcile_stale_runs to training_state.py**

```python
def reconcile_stale_runs(models_dir: Path) -> None:
    """Mark any runs with status 'training' as 'paused'.

    Called on TUI startup to handle cases where the TUI exited
    without graceful shutdown.
    """
    if not models_dir.exists():
        return
    for d in models_dir.iterdir():
        if not d.is_dir():
            continue
        run = load_training_run(d)
        if run is not None and run.status == "training":
            run.status = "paused"
            save_training_run(run, d)
```

**Step 4: Call reconcile on mount in TrainTab**

Update `TrainTab.on_mount()`:

```python
    def on_mount(self) -> None:
        """Hide dashboard on initial mount and reconcile stale runs."""
        self.query_one("#train-dashboard", TrainDashboard).display = False
        models_dir = getattr(self.app, "models_dir", None)
        if models_dir is not None:
            from ..training_state import reconcile_stale_runs
            reconcile_stale_runs(models_dir)
```

**Step 5: Run tests to verify**

Run: `pytest tests/test_training_state.py tests/test_tui_train.py -v`
Expected: All tests PASS

**Step 6: Commit**

```bash
git add scropipe/training_state.py scropipe/tui/train_tab.py tests/test_training_state.py
git commit -m "feat: reconcile stale training runs on TUI startup"
```

---

### Task 9: Final integration test and cleanup

**Files:**
- Test: `tests/test_tui_train.py`
- Review: all modified files

Run the full test suite, fix any regressions, and verify the complete flow.

**Step 1: Run all tests**

Run: `pytest tests/ -v`
Expected: All tests PASS

**Step 2: Run linter**

Run: `ruff check scropipe/ tests/`
Expected: No errors

**Step 3: Fix any issues found**

Address any test failures or lint errors.

**Step 4: Commit any fixes**

```bash
git add -A
git commit -m "fix: address test and lint issues from training cancel/resume"
```
