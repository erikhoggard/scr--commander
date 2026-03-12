# TUI: Export Model After Training Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make TUI-trained models appear on the Generate tab by running `rave export` and writing `model.ts` + `metadata.json` after training completes.

**Architecture:** Add a `build_export_cmd` helper in `rave_runner.py`, then call it from `_training_worker` in `train_tab.py` after successful training (returncode == 0). Copy the exported `.ts` to `model.ts` and write `metadata.json` — mirroring what `cli.py:1056-1114` already does.

**Tech Stack:** Python subprocess, shutil, json

---

### Task 1: Add `build_export_cmd` to rave_runner.py

**Files:**
- Modify: `scropipe/tui/rave_runner.py` (add function at end)
- Test: `tests/test_rave_runner.py` (if exists, else create)

**Step 1: Write the failing test**

```python
# tests/test_rave_runner.py
from scropipe.tui.rave_runner import build_export_cmd

def test_build_export_cmd_basic():
    cmd = build_export_cmd("rave", "/path/to/run")
    assert cmd == ["rave", "export", "--run", "/path/to/run"]

def test_build_export_cmd_streaming():
    cmd = build_export_cmd("rave", "/path/to/run", streaming=True)
    assert "--streaming" in cmd
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_rave_runner.py::test_build_export_cmd_basic -v`
Expected: FAIL with ImportError

**Step 3: Write minimal implementation**

Add to `scropipe/tui/rave_runner.py`:

```python
def build_export_cmd(
    rave_cmd: str,
    run_dir: str | Path,
    streaming: bool = False,
) -> list[str]:
    """Build the rave export command."""
    cmd = [rave_cmd, "export", "--run", str(run_dir)]
    if streaming:
        cmd.append("--streaming")
    return cmd
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_rave_runner.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add scropipe/tui/rave_runner.py tests/test_rave_runner.py
git commit -m "feat: add build_export_cmd helper to rave_runner"
```

---

### Task 2: Add post-training export + metadata to train_tab.py

**Files:**
- Modify: `scropipe/tui/train_tab.py:626-632` (the `returncode == 0` branch in `_training_worker`)

**Step 1: Write the failing test**

This is subprocess/thread integration code that's hard to unit test in isolation. Instead we test via the model_manager: after the export step runs, `ModelManager.list_models()` should find the model. We'll verify the export logic by creating a mock `.ts` file in the expected location and confirming `metadata.json` is written.

```python
# tests/test_train_export.py
import json
from pathlib import Path
from scropipe.model_manager import ModelManager

def test_exported_model_visible_in_model_manager(tmp_path):
    """After export, model.ts and metadata.json make the model discoverable."""
    model_dir = tmp_path / "models" / "test-model"
    model_dir.mkdir(parents=True)

    # Simulate what the export step should produce
    (model_dir / "model.ts").write_bytes(b"fake torchscript")
    metadata = {
        "name": "test-model",
        "created": "2026-03-10T00:00:00",
        "config": "v2",
        "total_samples": 0,
        "pool_name": "drums",
    }
    (model_dir / "metadata.json").write_text(json.dumps(metadata))

    mm = ModelManager(tmp_path / "models")
    models = mm.list_models()
    assert len(models) == 1
    assert models[0].name == "test-model"
    assert models[0].config == "v2"
    assert models[0].pool_name == "drums"
```

**Step 2: Run test to verify it passes** (this tests the contract, not the implementation)

Run: `python -m pytest tests/test_train_export.py -v`
Expected: PASS (confirms what our export code needs to produce)

**Step 3: Implement the export step in `_training_worker`**

In `scropipe/tui/train_tab.py`, replace the `elif returncode == 0:` block (lines 626-632) with:

```python
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
```

Add the `_export_model` method to `TrainTab`:

```python
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

        cmd = wrap_rave_cmd(build_export_cmd(rave_cmd, str(output_dir)))
        try:
            result = subprocess.run(cmd, check=False, capture_output=True, text=True)
            if result.returncode != 0:
                return False
        except Exception:
            return False

        # Find exported .ts file
        ts_files = list(output_dir.glob("**/*.ts"))
        if not ts_files:
            return False

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

        return True
```

**Step 4: Run all tests**

Run: `python -m pytest tests/ -v`
Expected: PASS

**Step 5: Commit**

```bash
git add scropipe/tui/train_tab.py tests/test_train_export.py
git commit -m "feat: export model after TUI training so it appears on Generate tab"
```
