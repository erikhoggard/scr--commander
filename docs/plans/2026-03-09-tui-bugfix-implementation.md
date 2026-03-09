# TUI Bugfix & Flow Repair Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix broken TUI flows: Browse buttons, Split & Add to Pool, stale Train tab pool data, and Pool tab file browsing.

**Architecture:** Create a reusable `BrowseModal` using Textual's `DirectoryTree` widget, then wire it into all Browse buttons. Fix data staleness by refreshing pool dropdowns on tab activation. Complete the Split & Add to Pool feature by adding pool selection and calling `PoolManager.add_files()`.

**Tech Stack:** Python, Textual 8.0.2, DirectoryTree widget

---

### Task 1: Create BrowseModal widget

**Files:**
- Create: `scropipe/tui/browse_modal.py`
- Test: `tests/test_browse_modal.py`

**Step 1: Write the failing test**

```python
# tests/test_browse_modal.py
import pytest
from pathlib import Path
from scropipe.tui.browse_modal import BrowseModal


@pytest.mark.asyncio
async def test_browse_modal_shows_directory_tree(tmp_path):
    """BrowseModal should display a DirectoryTree widget."""
    from textual.app import App, ComposeResult

    class TestApp(App):
        def compose(self) -> ComposeResult:
            yield from ()

    app = TestApp()
    async with app.run_test():
        modal = BrowseModal(
            title="Test Browse",
            start_path=tmp_path,
            select_type="directory",
        )
        await app.push_screen(modal)
        tree = app.query_one("DirectoryTree")
        assert tree is not None
        ok_btn = app.query_one("#browse-ok-btn")
        assert ok_btn is not None
        cancel_btn = app.query_one("#browse-cancel-btn")
        assert cancel_btn is not None


@pytest.mark.asyncio
async def test_browse_modal_cancel_returns_none(tmp_path):
    """Pressing Cancel should dismiss with None."""
    from textual.app import App, ComposeResult
    from textual.widgets import Button

    class TestApp(App):
        def compose(self) -> ComposeResult:
            yield from ()

    results = []
    app = TestApp()
    async with app.run_test() as pilot:
        modal = BrowseModal(
            title="Test",
            start_path=tmp_path,
            select_type="directory",
        )
        await app.push_screen(modal, callback=lambda r: results.append(r))
        await pilot.click("#browse-cancel-btn")
        await pilot.pause()
        assert results == [None]
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_browse_modal.py -v`
Expected: FAIL (module not found)

**Step 3: Implement BrowseModal**

```python
# scropipe/tui/browse_modal.py
"""Reusable file/directory browser modal using Textual's DirectoryTree."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, DirectoryTree, Label


class _FilteredDirectoryTree(DirectoryTree):
    """DirectoryTree that can optionally filter to directories only."""

    def __init__(self, path: str | Path, select_type: str = "file", **kwargs):
        super().__init__(path, **kwargs)
        self._select_type = select_type

    def filter_paths(self, paths):
        if self._select_type == "directory":
            return [p for p in paths if p.is_dir()]
        # For file mode, show everything but could filter to .wav etc later
        return paths


class BrowseModal(ModalScreen[Optional[Path]]):
    """Modal dialog with a DirectoryTree for browsing files/directories."""

    DEFAULT_CSS = """
    BrowseModal {
        align: center middle;
    }

    BrowseModal > Vertical {
        width: 70;
        height: 30;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }

    BrowseModal DirectoryTree {
        height: 1fr;
        margin-bottom: 1;
    }

    BrowseModal #browse-selected-path {
        height: 1;
        margin-bottom: 1;
        color: $text-muted;
    }
    """

    def __init__(
        self,
        title: str = "Browse",
        start_path: Path | None = None,
        select_type: Literal["file", "directory"] = "file",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._title_text = title
        self._start_path = start_path or Path.home()
        self._select_type = select_type
        self._selected_path: Path | None = None

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label(self._title_text)
            yield _FilteredDirectoryTree(
                self._start_path,
                select_type=self._select_type,
                id="browse-tree",
            )
            yield Label("No selection", id="browse-selected-path")
            with Horizontal(classes="action-bar"):
                yield Button("OK", variant="primary", id="browse-ok-btn")
                yield Button("Cancel", id="browse-cancel-btn")

    def on_directory_tree_file_selected(
        self, event: DirectoryTree.FileSelected
    ) -> None:
        if self._select_type == "file":
            self._selected_path = event.path
            self.query_one("#browse-selected-path", Label).update(
                str(event.path)
            )

    def on_directory_tree_directory_selected(
        self, event: DirectoryTree.DirectorySelected
    ) -> None:
        if self._select_type == "directory":
            self._selected_path = event.path
            self.query_one("#browse-selected-path", Label).update(
                str(event.path)
            )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "browse-ok-btn":
            self.dismiss(self._selected_path)
        elif event.button.id == "browse-cancel-btn":
            self.dismiss(None)
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_browse_modal.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add scropipe/tui/browse_modal.py tests/test_browse_modal.py
git commit -m "feat: add BrowseModal with DirectoryTree for file/directory browsing"
```

---

### Task 2: Wire Browse buttons on Split tab

**Files:**
- Modify: `scropipe/tui/split_tab.py` (lines 127-132 in `on_button_pressed`)
- Test: `tests/test_tui_split.py`

**Step 1: Write the failing test**

Add to `tests/test_tui_split.py`:

```python
@pytest.mark.asyncio
async def test_split_browse_source_opens_modal(app):
    """Clicking Browse for source should push a BrowseModal."""
    async with app.run_test() as pilot:
        tabbed = app.query_one("TabbedContent")
        tabbed.active = "tab-split"
        await pilot.pause()
        await pilot.click("#split-browse-source")
        await pilot.pause()
        # BrowseModal should be on the screen stack
        from scropipe.tui.browse_modal import BrowseModal
        assert any(isinstance(s, BrowseModal) for s in app.screen_stack)


@pytest.mark.asyncio
async def test_split_browse_output_opens_modal(app):
    """Clicking Browse for output should push a BrowseModal."""
    async with app.run_test() as pilot:
        tabbed = app.query_one("TabbedContent")
        tabbed.active = "tab-split"
        await pilot.pause()
        await pilot.click("#split-browse-output")
        await pilot.pause()
        from scropipe.tui.browse_modal import BrowseModal
        assert any(isinstance(s, BrowseModal) for s in app.screen_stack)
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_tui_split.py::test_split_browse_source_opens_modal tests/test_tui_split.py::test_split_browse_output_opens_modal -v`
Expected: FAIL (BrowseModal not pushed)

**Step 3: Add Browse button handlers to SplitTab**

In `scropipe/tui/split_tab.py`, add import at top:

```python
from .browse_modal import BrowseModal
```

Replace the `on_button_pressed` method (line 127-132):

```python
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "split-btn":
            self._run_split(add_to_pool=False)
        elif event.button.id == "split-and-pool-btn":
            self._run_split(add_to_pool=True)
        elif event.button.id == "split-browse-source":
            self.app.push_screen(
                BrowseModal(title="Select Source File", select_type="file"),
                callback=self._on_browse_source,
            )
        elif event.button.id == "split-browse-output":
            self.app.push_screen(
                BrowseModal(title="Select Output Directory", select_type="directory"),
                callback=self._on_browse_output,
            )
```

Add callback methods:

```python
    def _on_browse_source(self, path: Path | None) -> None:
        """Handle source file browse result."""
        if path is not None:
            self.query_one("#split-source-input", Input).value = str(path)

    def _on_browse_output(self, path: Path | None) -> None:
        """Handle output directory browse result."""
        if path is not None:
            self.query_one("#split-output-input", Input).value = str(path)
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_tui_split.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add scropipe/tui/split_tab.py tests/test_tui_split.py
git commit -m "fix: wire Browse buttons on Split tab to BrowseModal"
```

---

### Task 3: Add pool selector and complete "Split & Add to Pool"

**Files:**
- Modify: `scropipe/tui/split_tab.py`
- Test: `tests/test_tui_split.py`

**Step 1: Write the failing test**

Add to `tests/test_tui_split.py`:

```python
@pytest.mark.asyncio
async def test_split_tab_has_pool_selector(app):
    """Split tab should have a pool Select dropdown."""
    async with app.run_test() as pilot:
        tabbed = app.query_one("TabbedContent")
        tabbed.active = "tab-split"
        await pilot.pause()
        pool_select = app.query_one("#split-pool-select")
        assert pool_select is not None
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_tui_split.py::test_split_tab_has_pool_selector -v`
Expected: FAIL (widget not found)

**Step 3: Add pool selector and wire Split & Add to Pool**

In `scropipe/tui/split_tab.py`, add imports:

```python
from textual.widgets import Button, Input, Label, RadioButton, RadioSet, Select, Static
```

In `compose()`, add pool selector before the action buttons (before line 103 `with Horizontal(classes="action-bar")`):

```python
            # Pool selector (for "Split & Add to Pool")
            yield Label("Target Pool", classes="section-title")
            yield Select([], id="split-pool-select", prompt="Select a pool")
```

Add a mount handler to populate the pool dropdown:

```python
    def on_mount(self) -> None:
        """Populate pool list on mount."""
        self._refresh_pools()

    def _refresh_pools(self) -> None:
        """Load available pools into the pool selector."""
        from ..pool_manager import PoolManager

        pool_select = self.query_one("#split-pool-select", Select)
        pools_dir = getattr(self.app, "pools_dir", None)
        if pools_dir is None:
            return
        try:
            pm = PoolManager(pools_dir)
            pools = pm.list_pools()
            options = [(f"{p.name} ({p.sample_count} samples)", p.name) for p in pools]
            pool_select.set_options(options)
        except Exception:
            pool_select.set_options([])
```

Modify `_run_split` to validate pool selection when `add_to_pool=True`:

```python
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

        # Validate pool selection if adding to pool
        pool_name = None
        if add_to_pool:
            pool_select = self.query_one("#split-pool-select", Select)
            if pool_select.value is Select.BLANK:
                status.update("Error: Please select a target pool.")
                return
            pool_name = str(pool_select.value)

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
            self._do_split(Path(output_dir), kwargs, add_to_pool, pool_name),
            name="split_worker",
            thread=True,
        )
```

Modify `_do_split` to actually add files to the pool:

```python
    async def _do_split(
        self, output_dir: Path, kwargs: dict, add_to_pool: bool, pool_name: str | None = None,
    ) -> None:
        """Run the split stage in a worker thread."""
        from scropipe.stages import SplitStage

        status = self.query_one("#split-status", Static)

        try:
            stage = SplitStage(output_base=output_dir.parent)
            result = stage.run(**kwargs)

            if result.success:
                msg = f"Success: {result.message}"
                if add_to_pool and pool_name and result.output_dir:
                    try:
                        from ..pool_manager import PoolManager

                        pools_dir = getattr(self.app, "pools_dir", None)
                        if pools_dir:
                            pm = PoolManager(pools_dir)
                            wav_files = list(result.output_dir.rglob("*.wav"))
                            if wav_files:
                                added = pm.add_files(pool_name, wav_files)
                                msg += f" | Added {added} samples to pool '{pool_name}'"
                                # Refresh pool selector to show updated counts
                                self.app.call_from_thread(self._refresh_pools)
                            else:
                                msg += " | No WAV files to add to pool"
                    except Exception as e:
                        msg += f" | Failed to add to pool: {e}"
                self.app.call_from_thread(status.update, msg)
            else:
                self.app.call_from_thread(status.update, f"Error: {result.message}")
        except Exception as e:
            self.app.call_from_thread(status.update, f"Error: {e}")
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_tui_split.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add scropipe/tui/split_tab.py tests/test_tui_split.py
git commit -m "feat: complete Split & Add to Pool with pool selector and PoolManager integration"
```

---

### Task 4: Fix Train tab stale pool data

**Files:**
- Modify: `scropipe/tui/train_tab.py` (TrainConfigPanel)
- Modify: `scropipe/tui/app.py` (add tab change watcher)
- Test: `tests/test_tui_train.py`

**Step 1: Write the failing test**

Add to `tests/test_tui_train.py`:

```python
@pytest.mark.asyncio
async def test_train_tab_refreshes_pools_on_tab_switch(tmp_path):
    """Train tab should refresh pool data when switched to."""
    from scropipe.pool_manager import PoolManager

    pools_dir = tmp_path / "pools"
    pools_dir.mkdir()
    models_dir = tmp_path / "models"
    models_dir.mkdir()

    app = ScropipeApp(models_dir=models_dir, pools_dir=pools_dir)
    async with app.run_test() as pilot:
        # Start on split tab
        tabbed = app.query_one("TabbedContent")
        tabbed.active = "tab-split"
        await pilot.pause()

        # Create a pool while on split tab
        pm = PoolManager(pools_dir)
        pm.create_pool("test-pool")

        # Switch to train tab
        tabbed.active = "tab-train"
        await pilot.pause()

        # Pool selector should show the new pool
        from textual.widgets import Select
        pool_select = app.query_one("#train-pool-select", Select)
        option_values = [opt.value for opt in pool_select._options]
        assert "test-pool" in option_values
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_tui_train.py::test_train_tab_refreshes_pools_on_tab_switch -v`
Expected: FAIL (pool not in dropdown because it was populated on mount before pool existed)

**Step 3: Add tab activation refresh**

In `scropipe/tui/app.py`, add a watcher for tab changes. Add this method to `ScropipeApp`:

```python
    def on_tabbed_content_tab_activated(
        self, event: TabbedContent.TabActivated
    ) -> None:
        """Refresh tab data when switching tabs."""
        tab_id = event.pane.id
        if tab_id == "tab-train":
            try:
                config = self.query_one("#train-config", TrainConfigPanel)
                config._populate_pools()
            except Exception:
                pass
        elif tab_id == "tab-split":
            try:
                split_tab = self.query_one("SplitTab")
                split_tab._refresh_pools()
            except Exception:
                pass
        elif tab_id == "tab-generate":
            try:
                gen_tab = self.query_one("GenerateTab")
                gen_tab._refresh_models()
            except Exception:
                pass
```

Add the import for TrainConfigPanel at the top of app.py:

```python
from .train_tab import TrainConfigPanel, TrainTab
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_tui_train.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add scropipe/tui/app.py tests/test_tui_train.py
git commit -m "fix: refresh pool/model data on tab switch to prevent stale dropdowns"
```

---

### Task 5: Wire Browse buttons on Pool tab

**Files:**
- Modify: `scropipe/tui/pool_tab.py` (lines 235-250 in `on_button_pressed`)
- Test: `tests/test_tui_pool.py`

**Step 1: Write the failing test**

Add to `tests/test_tui_pool.py`:

```python
@pytest.mark.asyncio
async def test_pool_add_files_opens_browse_modal(app):
    """Clicking Add Files should push a BrowseModal."""
    from scropipe.pool_manager import PoolManager

    # Create a pool first
    pm = PoolManager(app.pools_dir)
    pm.create_pool("test-pool")

    async with app.run_test() as pilot:
        tabbed = app.query_one("TabbedContent")
        tabbed.active = "tab-pool"
        await pilot.pause()

        # Select the pool
        pool_list = app.query_one("#pool-list")
        if pool_list.children:
            pool_list.index = 0
            await pilot.pause()

        await pilot.click("#add-files-btn")
        await pilot.pause()

        from scropipe.tui.browse_modal import BrowseModal
        assert any(isinstance(s, BrowseModal) for s in app.screen_stack)


@pytest.mark.asyncio
async def test_pool_add_dir_opens_browse_modal(app):
    """Clicking Add Directory should push a BrowseModal."""
    from scropipe.pool_manager import PoolManager

    pm = PoolManager(app.pools_dir)
    pm.create_pool("test-pool")

    async with app.run_test() as pilot:
        tabbed = app.query_one("TabbedContent")
        tabbed.active = "tab-pool"
        await pilot.pause()

        pool_list = app.query_one("#pool-list")
        if pool_list.children:
            pool_list.index = 0
            await pilot.pause()

        await pilot.click("#add-dir-btn")
        await pilot.pause()

        from scropipe.tui.browse_modal import BrowseModal
        assert any(isinstance(s, BrowseModal) for s in app.screen_stack)
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_tui_pool.py::test_pool_add_files_opens_browse_modal tests/test_tui_pool.py::test_pool_add_dir_opens_browse_modal -v`
Expected: FAIL (FileInputModal pushed instead of BrowseModal)

**Step 3: Replace FileInputModal with BrowseModal in Pool tab**

In `scropipe/tui/pool_tab.py`, add import:

```python
from .browse_modal import BrowseModal
```

Replace the `add-files-btn` and `add-dir-btn` handlers in `on_button_pressed` (lines 235-250):

```python
        elif event.button.id == "add-files-btn":
            pool_name = self._get_selected_pool_name()
            if pool_name is None:
                return
            self.app.push_screen(
                BrowseModal(
                    title="Add Files",
                    select_type="file",
                    start_path=self.app.pools_dir,
                ),
                callback=self._on_add_files_dismiss,
            )
        elif event.button.id == "add-dir-btn":
            pool_name = self._get_selected_pool_name()
            if pool_name is None:
                return
            self.app.push_screen(
                BrowseModal(
                    title="Add Directory",
                    select_type="directory",
                    start_path=self.app.pools_dir,
                ),
                callback=self._on_add_dir_dismiss,
            )
```

Update the callback signatures to accept `Path | None` instead of `str | None`:

```python
    def _on_add_files_dismiss(self, path: Path | None) -> None:
        """Handle add files browse result."""
        if path is None:
            return
        pool_name = self._get_selected_pool_name()
        if pool_name is None:
            return
        try:
            pm = self._get_pool_manager()
            pm.add_files(pool_name, [path])
            self._refresh_pool_list()
            self._show_pool_details(pool_name)
        except Exception:
            pass

    def _on_add_dir_dismiss(self, path: Path | None) -> None:
        """Handle add directory browse result."""
        if path is None:
            return
        pool_name = self._get_selected_pool_name()
        if pool_name is None:
            return
        try:
            pm = self._get_pool_manager()
            pm.add_directory(pool_name, path)
            self._refresh_pool_list()
            self._show_pool_details(pool_name)
        except Exception:
            pass
```

Note: `FileInputModal` and `NewPoolModal` can stay in the file — `NewPoolModal` is still used for creating pools. `FileInputModal` can be removed if nothing else imports it, but the Generate tab currently imports it. The Generate tab will be updated in Task 6.

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_tui_pool.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add scropipe/tui/pool_tab.py tests/test_tui_pool.py
git commit -m "fix: replace FileInputModal with BrowseModal on Pool tab"
```

---

### Task 6: Wire Browse buttons on Generate tab

**Files:**
- Modify: `scropipe/tui/generate_tab.py` (lines 135-138, 140-167)
- Test: `tests/test_tui_generate.py`

**Step 1: Write the failing test**

Add to `tests/test_tui_generate.py`:

```python
@pytest.mark.asyncio
async def test_gen_input_browse_opens_browse_modal(app):
    """Clicking Browse for input dir should push a BrowseModal."""
    async with app.run_test() as pilot:
        tabbed = app.query_one("TabbedContent")
        tabbed.active = "tab-generate"
        await pilot.pause()
        await pilot.click("#gen-input-browse-btn")
        await pilot.pause()
        from scropipe.tui.browse_modal import BrowseModal
        assert any(isinstance(s, BrowseModal) for s in app.screen_stack)


@pytest.mark.asyncio
async def test_gen_output_browse_opens_browse_modal(app):
    """Clicking Browse for output dir should push a BrowseModal."""
    async with app.run_test() as pilot:
        tabbed = app.query_one("TabbedContent")
        tabbed.active = "tab-generate"
        await pilot.pause()
        await pilot.click("#gen-output-browse-btn")
        await pilot.pause()
        from scropipe.tui.browse_modal import BrowseModal
        assert any(isinstance(s, BrowseModal) for s in app.screen_stack)
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_tui_generate.py::test_gen_input_browse_opens_browse_modal tests/test_tui_generate.py::test_gen_output_browse_opens_browse_modal -v`
Expected: FAIL (FileInputModal pushed instead of BrowseModal)

**Step 3: Replace FileInputModal with BrowseModal in Generate tab**

In `scropipe/tui/generate_tab.py`, replace the `from .pool_tab import FileInputModal` import with:

```python
from .browse_modal import BrowseModal
```

Replace `_browse_input_dir` and `_browse_output_dir` methods:

```python
    def _browse_input_dir(self) -> None:
        """Open a modal to browse for the input directory."""
        self.app.push_screen(
            BrowseModal(
                title="Input Directory",
                select_type="directory",
            ),
            callback=self._on_input_dir_selected,
        )

    def _on_input_dir_selected(self, path: Path | None) -> None:
        """Handle input directory modal result."""
        if path is not None:
            self.query_one("#gen-input-dir", Input).value = str(path)

    def _browse_output_dir(self) -> None:
        """Open a modal to browse for the output directory."""
        self.app.push_screen(
            BrowseModal(
                title="Output Directory",
                select_type="directory",
            ),
            callback=self._on_output_dir_selected,
        )

    def _on_output_dir_selected(self, path: Path | None) -> None:
        """Handle output directory modal result."""
        if path is not None:
            self.query_one("#gen-output-dir", Input).value = str(path)
```

Add `Path` import if not already present:

```python
from pathlib import Path
```

(Already present at line 7.)

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_tui_generate.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add scropipe/tui/generate_tab.py tests/test_tui_generate.py
git commit -m "fix: replace FileInputModal with BrowseModal on Generate tab"
```

---

### Task 7: Clean up unused FileInputModal

**Files:**
- Modify: `scropipe/tui/pool_tab.py` (remove FileInputModal class)

**Step 1: Check nothing imports FileInputModal**

Run: `grep -r "FileInputModal" scropipe/ tests/`

After Tasks 5 and 6, nothing should import it anymore. If something still does, skip this task.

**Step 2: Remove FileInputModal class**

In `scropipe/tui/pool_tab.py`, delete lines 54-99 (the entire `FileInputModal` class).

**Step 3: Run all tests**

Run: `python -m pytest tests/ -v`
Expected: All PASS

**Step 4: Commit**

```bash
git add scropipe/tui/pool_tab.py
git commit -m "refactor: remove unused FileInputModal"
```

---

### Task 8: Run full test suite and manual smoke test

**Step 1: Run all tests**

Run: `python -m pytest tests/ -v`
Expected: All PASS

**Step 2: Manual smoke test checklist**

Run: `python -m scropipe.tui.app` (or however the TUI is launched)

Verify:
- [ ] Split tab: Browse button for source opens DirectoryTree modal
- [ ] Split tab: Browse button for output opens DirectoryTree modal
- [ ] Split tab: Pool selector dropdown appears and shows pools
- [ ] Split tab: "Split & Add to Pool" validates pool selection
- [ ] Pool tab: "Add Files" opens DirectoryTree modal
- [ ] Pool tab: "Add Directory" opens DirectoryTree modal
- [ ] Train tab: Switching to tab refreshes pool dropdown with current counts
- [ ] Generate tab: Browse buttons open DirectoryTree modals
- [ ] Tab switching via keys 1-4 still works
- [ ] PathSuggester autocomplete still works in text inputs (right-arrow to accept)

**Step 3: Final commit if any fixes needed**

```bash
git add -A
git commit -m "fix: address issues found during smoke testing"
```
