# Resume Training Run Selector Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Let the user choose which paused training run to resume via a dropdown instead of blindly picking the most recent one.

**Architecture:** Add a Select widget to TrainConfigPanel, populate it with paused runs from `list_paused_runs()`, and read the selection in `_resume_training()`. Refresh the dropdown on mount and tab switch.

**Tech Stack:** Textual (Select widget), existing training_state module

---

### Task 1: Add resume run selector and wire it up

**Files:**
- Modify: `scropipe/tui/train_tab.py:31-72` (compose), `74-77` (on_mount), `240-281` (_resume_training)
- Modify: `scropipe/tui/app.py:140-145` (tab switch refresh)
- Test: `tests/test_tui_train.py`

**Step 1: Write the failing tests**

Add to `tests/test_tui_train.py`:

```python
@pytest.mark.asyncio
async def test_train_tab_has_resume_run_selector(app):
    async with app.run_test() as pilot:
        tabbed = app.query_one("TabbedContent")
        tabbed.active = "tab-train"
        await pilot.pause()
        select = app.query_one("#resume-run-select")
        assert select is not None


@pytest.mark.asyncio
async def test_train_tab_resume_selector_populates_with_paused_runs(app, tmp_path):
    """When paused runs exist, the resume selector should list them."""
    from scropipe.training_state import TrainingRunInfo, save_training_run

    models_dir = tmp_path / "models"
    run_dir = models_dir / "my-paused-model"
    run_dir.mkdir(parents=True)
    run = TrainingRunInfo(
        model_name="my-paused-model",
        pool_name="drums",
        architecture="v2",
        output_dir=str(run_dir / "training_output"),
        status="paused",
    )
    save_training_run(run, run_dir)

    async with app.run_test() as pilot:
        tabbed = app.query_one("TabbedContent")
        tabbed.active = "tab-train"
        await pilot.pause()
        config = app.query_one("#train-config")
        config._populate_resumable_runs()
        await pilot.pause()
        select = app.query_one("#resume-run-select")
        # Should have at least one option
        assert len(select._options) > 0
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_tui_train.py::test_train_tab_has_resume_run_selector -v`
Expected: FAIL (no widget with id resume-run-select)

**Step 3: Implement**

In `scropipe/tui/train_tab.py`, modify `TrainConfigPanel.compose()` — add the selector and label before the Resume button. Change lines 70-72 from:

```python
            with Horizontal(classes="action-bar"):
                yield Button("Start Training", variant="primary", id="start-training-btn")
                yield Button("Resume Training", variant="default", id="resume-training-btn")
```

to:

```python
            yield Label("Resume Run", classes="section-title")
            yield Select([], id="resume-run-select", prompt="Select a paused run")

            with Horizontal(classes="action-bar"):
                yield Button("Start Training", variant="primary", id="start-training-btn")
                yield Button("Resume Training", variant="default", id="resume-training-btn")
```

Add `_populate_resumable_runs` method to `TrainConfigPanel` (after `_populate_pools`):

```python
    def _populate_resumable_runs(self) -> None:
        """Load paused training runs into the resume selector."""
        from ..training_state import list_paused_runs

        select = self.query_one("#resume-run-select", Select)
        models_dir = getattr(self.app, "models_dir", None)
        if models_dir is None:
            select.set_options([])
            return
        try:
            paused = list_paused_runs(models_dir)
            options = [
                (f"{r.model_name} ({r.architecture}, {r.status})", r.model_name)
                for r in paused
            ]
            select.set_options(options)
        except Exception:
            select.set_options([])
```

Call it from `on_mount` — change lines 74-77 from:

```python
    def on_mount(self) -> None:
        """Populate pool list and detect GPU on mount."""
        self._populate_pools()
        self._detect_gpu()
```

to:

```python
    def on_mount(self) -> None:
        """Populate pool list, resumable runs, and detect GPU on mount."""
        self._populate_pools()
        self._populate_resumable_runs()
        self._detect_gpu()
```

In `scropipe/tui/app.py`, add the refresh call — change lines 140-145 from:

```python
        if tab_id == "tab-train":
            try:
                config = self.query_one("#train-config")
                config._populate_pools()
            except Exception:
                pass
```

to:

```python
        if tab_id == "tab-train":
            try:
                config = self.query_one("#train-config")
                config._populate_pools()
                config._populate_resumable_runs()
            except Exception:
                pass
```

Modify `TrainTab._resume_training()` to use the selection. Change lines 240-281 — replace the `paused = list_paused_runs(models_dir)` and `run = paused[-1]` logic:

```python
    def _resume_training(self) -> None:
        """Resume training for the selected paused run."""
        from ..training_state import find_checkpoint_dir, list_paused_runs

        models_dir = getattr(self.app, "models_dir", None)
        if models_dir is None:
            self._update_status("Error: Models directory not configured.")
            return

        # Read selection from the resume-run-select dropdown
        config = self.query_one("#train-config", TrainConfigPanel)
        select = config.query_one("#resume-run-select", Select)
        if select.value is Select.BLANK:
            self._update_status("Please select a run to resume.")
            return

        selected_name = str(select.value)

        # Find the matching run
        paused = list_paused_runs(models_dir)
        run = None
        for r in paused:
            if r.model_name == selected_name:
                run = r
                break

        if run is None:
            self._update_status(f"Run '{selected_name}' not found or no longer paused.")
            return

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

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_tui_train.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add scropipe/tui/train_tab.py scropipe/tui/app.py tests/test_tui_train.py
git commit -m "feat: add dropdown to select which paused run to resume"
```
