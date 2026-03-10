# Path Input UX Improvements Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix two UX friction points: Tab key completes paths in input fields (like a terminal), and the file browser starts at the drive root with the current directory expanded.

**Architecture:** A custom `PathInput(Input)` subclass intercepts Tab to cycle through filesystem matches. `BrowseModal` roots its `DirectoryTree` at the drive root and pre-expands to the start path. On Windows, a drive selector dropdown allows switching drives.

**Tech Stack:** Textual (Input, DirectoryTree, Select), pathlib, sys.platform

---

### Task 1: PathInput widget — Tab cycling with mocked filesystem tests

**Files:**
- Create: `tests/test_path_input.py`
- Create: `scropipe/tui/path_input.py`

**Step 1: Write the failing tests**

In `tests/test_path_input.py`:

```python
"""Tests for PathInput tab-cycling path completion."""

import pytest
from pathlib import Path
from unittest.mock import patch


def _make_tree(tmp_path):
    """Create a test directory structure."""
    (tmp_path / "Documents").mkdir()
    (tmp_path / "Downloads").mkdir()
    (tmp_path / "Desktop").mkdir()
    (tmp_path / "file.txt").touch()
    return tmp_path


@pytest.mark.asyncio
async def test_tab_cycles_through_matches(tmp_path):
    """Tab should cycle through matching entries."""
    from textual.app import App, ComposeResult
    from scropipe.tui.path_input import PathInput

    tree = _make_tree(tmp_path)

    class TestApp(App):
        def compose(self) -> ComposeResult:
            yield PathInput(id="test-input")

    app = TestApp()
    async with app.run_test() as pilot:
        inp = app.query_one("#test-input", PathInput)
        # Type a partial path matching Desktop, Documents, Downloads
        inp.value = str(tree) + "/D"
        inp.cursor_position = len(inp.value)
        await pilot.press("tab")
        first = inp.value
        await pilot.press("tab")
        second = inp.value
        await pilot.press("tab")
        third = inp.value
        # All three D-directories should appear in some order
        values = {first, second, third}
        expected_names = {"Desktop", "Documents", "Downloads"}
        actual_names = {Path(v).name for v in values}
        assert actual_names == expected_names


@pytest.mark.asyncio
async def test_tab_single_match_completes_with_separator(tmp_path):
    """When only one match, Tab completes and appends separator for dirs."""
    from textual.app import App, ComposeResult
    from scropipe.tui.path_input import PathInput

    tree = _make_tree(tmp_path)

    class TestApp(App):
        def compose(self) -> ComposeResult:
            yield PathInput(id="test-input")

    app = TestApp()
    async with app.run_test() as pilot:
        inp = app.query_one("#test-input", PathInput)
        inp.value = str(tree) + "/Des"
        inp.cursor_position = len(inp.value)
        await pilot.press("tab")
        assert inp.value == str(tree) + "/Desktop/"


@pytest.mark.asyncio
async def test_tab_no_match_does_nothing(tmp_path):
    """Tab with no matches should leave value unchanged."""
    from textual.app import App, ComposeResult
    from scropipe.tui.path_input import PathInput

    tree = _make_tree(tmp_path)

    class TestApp(App):
        def compose(self) -> ComposeResult:
            yield PathInput(id="test-input")

    app = TestApp()
    async with app.run_test() as pilot:
        inp = app.query_one("#test-input", PathInput)
        inp.value = str(tree) + "/Zzz"
        inp.cursor_position = len(inp.value)
        original = inp.value
        await pilot.press("tab")
        assert inp.value == original


@pytest.mark.asyncio
async def test_typing_resets_cycle(tmp_path):
    """Typing after Tab should reset the cycle position."""
    from textual.app import App, ComposeResult
    from scropipe.tui.path_input import PathInput

    tree = _make_tree(tmp_path)

    class TestApp(App):
        def compose(self) -> ComposeResult:
            yield PathInput(id="test-input")

    app = TestApp()
    async with app.run_test() as pilot:
        inp = app.query_one("#test-input", PathInput)
        inp.value = str(tree) + "/D"
        inp.cursor_position = len(inp.value)
        await pilot.press("tab")
        first_tab = inp.value
        # Type something to reset
        inp.value = str(tree) + "/D"
        inp.cursor_position = len(inp.value)
        await pilot.press("tab")
        # Should get the same first result again
        assert inp.value == first_tab


@pytest.mark.asyncio
async def test_tab_completes_file(tmp_path):
    """Tab should also complete files, without trailing separator."""
    from textual.app import App, ComposeResult
    from scropipe.tui.path_input import PathInput

    tree = _make_tree(tmp_path)

    class TestApp(App):
        def compose(self) -> ComposeResult:
            yield PathInput(id="test-input")

    app = TestApp()
    async with app.run_test() as pilot:
        inp = app.query_one("#test-input", PathInput)
        inp.value = str(tree) + "/fi"
        inp.cursor_position = len(inp.value)
        await pilot.press("tab")
        assert inp.value == str(tree) + "/file.txt"


@pytest.mark.asyncio
async def test_directories_only_mode(tmp_path):
    """PathInput(directories_only=True) should skip files."""
    from textual.app import App, ComposeResult
    from scropipe.tui.path_input import PathInput

    tree = _make_tree(tmp_path)

    class TestApp(App):
        def compose(self) -> ComposeResult:
            yield PathInput(directories_only=True, id="test-input")

    app = TestApp()
    async with app.run_test() as pilot:
        inp = app.query_one("#test-input", PathInput)
        inp.value = str(tree) + "/fi"
        inp.cursor_position = len(inp.value)
        original = inp.value
        await pilot.press("tab")
        # No match — file.txt is a file, not a directory
        assert inp.value == original
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_path_input.py -v`
Expected: ImportError — `scropipe.tui.path_input` does not exist

**Step 3: Write the PathInput widget**

In `scropipe/tui/path_input.py`:

```python
"""PathInput — an Input widget with Tab-cycling path completion."""

from __future__ import annotations

from pathlib import Path

from textual.events import Key
from textual.widgets import Input

from .path_suggester import PathSuggester


class PathInput(Input):
    """Input that intercepts Tab to cycle through filesystem path matches.

    Press Tab to complete the current partial path. If multiple entries match,
    repeated Tab presses cycle through them. Typing anything resets the cycle.
    Shift+Tab still moves focus backward as normal.
    """

    def __init__(
        self,
        *,
        directories_only: bool = False,
        **kwargs,
    ) -> None:
        # Wire up PathSuggester for right-arrow ghost suggestions too
        suggester = PathSuggester(directories_only=directories_only)
        super().__init__(suggester=suggester, **kwargs)
        self._directories_only = directories_only
        self._cycle_matches: list[str] = []
        self._cycle_index: int = -1
        self._cycle_prefix: str = ""

    def _on_key(self, event: Key) -> None:
        if event.key == "tab":
            event.prevent_default()
            event.stop()
            self._tab_complete()
            return
        # Any other key resets cycle state
        self._cycle_matches = []
        self._cycle_index = -1
        self._cycle_prefix = ""

    def _tab_complete(self) -> None:
        """Cycle through matching filesystem entries."""
        current = self.value

        # If we're already cycling on the same prefix, advance
        if self._cycle_matches and self._cycle_prefix:
            self._cycle_index = (self._cycle_index + 1) % len(self._cycle_matches)
            self.value = self._cycle_matches[self._cycle_index]
            self.cursor_position = len(self.value)
            return

        # Build match list from current value
        if not current:
            return

        path = Path(current)
        try:
            expanded = path.expanduser()
        except (RuntimeError, ValueError):
            expanded = path

        parent = expanded.parent
        partial = expanded.name.lower()

        if not parent.is_dir():
            return

        try:
            entries = sorted(parent.iterdir(), key=lambda p: p.name.lower())
        except PermissionError:
            return

        if self._directories_only:
            entries = [e for e in entries if e.is_dir()]

        matches = [e for e in entries if e.name.lower().startswith(partial)]

        if not matches:
            return

        # Build completed path strings preserving original prefix
        prefix = current[: len(current) - len(expanded.name)]
        completed = []
        for m in matches:
            val = prefix + m.name
            if m.is_dir():
                val += "/"
            completed.append(val)

        self._cycle_prefix = current
        self._cycle_matches = completed
        self._cycle_index = 0
        self.value = completed[0]
        self.cursor_position = len(self.value)
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_path_input.py -v`
Expected: All 6 tests PASS

**Step 5: Commit**

```bash
git add scropipe/tui/path_input.py tests/test_path_input.py
git commit -m "feat: add PathInput widget with Tab-cycling path completion"
```

---

### Task 2: Swap Input → PathInput in all TUI tabs

**Files:**
- Modify: `scropipe/tui/split_tab.py` (lines 9, 79, 97)
- Modify: `scropipe/tui/generate_tab.py` (lines 12, 48, 54)
- Modify: `scropipe/tui/app.py` (lines 15, 56, 58)

**Step 1: Update split_tab.py**

Replace the `from .path_suggester import PathSuggester` import with:
```python
from .path_input import PathInput
```

Replace the two path Input widgets (lines 79, 97):
```python
# Line 79: was Input(..., suggester=PathSuggester())
yield PathInput(placeholder="Path to audio file", id="split-source-input")

# Line 97: was Input(..., suggester=PathSuggester(directories_only=True))
yield PathInput(placeholder="Path to output directory", id="split-output-input", directories_only=True)
```

Update `_on_browse_source` and `_on_browse_output` to query `PathInput` instead of `Input`:
```python
# Line 188
self.query_one("#split-source-input", PathInput).value = str(path)
# Line 193
self.query_one("#split-output-input", PathInput).value = str(path)
```

Keep `Input` import for the non-path inputs (transient/grid/texture settings).

**Step 2: Update generate_tab.py**

Replace `from .path_suggester import PathSuggester` with:
```python
from .path_input import PathInput
```

Replace path Input widgets (lines 48, 54):
```python
yield PathInput(placeholder="Path to input WAV files", id="gen-input-dir")
yield PathInput(placeholder="Path for output files", id="gen-output-dir", directories_only=True)
```

Update `_on_input_dir_selected` and `_on_output_dir_selected` to query `PathInput`.

**Step 3: Update app.py (SetupModal)**

Replace `from .path_suggester import PathSuggester` with:
```python
from .path_input import PathInput
```

Replace setup modal Input widgets (lines 56, 58):
```python
yield PathInput(placeholder="e.g. ~/scropipe/models", id="models-dir-input", directories_only=True)
yield PathInput(placeholder="e.g. ~/scropipe/pools", id="pools-dir-input", directories_only=True)
```

Update `on_button_pressed` to query `PathInput`:
```python
models_input = self.query_one("#models-dir-input", PathInput)
pools_input = self.query_one("#pools-dir-input", PathInput)
```

**Step 4: Run existing TUI tests to verify nothing broke**

Run: `python -m pytest tests/test_tui_split.py tests/test_tui_generate.py tests/test_tui_setup.py tests/test_tui_app.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add scropipe/tui/split_tab.py scropipe/tui/generate_tab.py scropipe/tui/app.py
git commit -m "refactor: swap Input for PathInput on all path fields"
```

---

### Task 3: BrowseModal — root at drive root with expanded path

**Files:**
- Modify: `scropipe/tui/browse_modal.py`
- Modify: `tests/test_browse_modal.py`

**Step 1: Write the failing tests**

Add to `tests/test_browse_modal.py`:

```python
@pytest.mark.asyncio
async def test_browse_modal_roots_at_drive_root(tmp_path):
    """BrowseModal should root the tree at the drive root, not start_path."""
    from textual.app import App, ComposeResult
    from textual.widgets import DirectoryTree

    class TestApp(App):
        def compose(self) -> ComposeResult:
            yield from ()

    app = TestApp()
    async with app.run_test() as pilot:
        modal = BrowseModal(
            title="Test",
            start_path=tmp_path,
            select_type="directory",
        )
        await app.push_screen(modal)
        await pilot.pause()
        tree = app.screen.query_one(DirectoryTree)
        # Tree root should be the drive root (C:\ on Windows, / on Unix)
        root = Path(tree.path)
        assert root == Path(tmp_path.anchor)


import sys

@pytest.mark.asyncio
@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only drive selector")
async def test_browse_modal_has_drive_selector_on_windows(tmp_path):
    """On Windows, BrowseModal should show a drive selector."""
    from textual.app import App, ComposeResult
    from textual.widgets import Select

    class TestApp(App):
        def compose(self) -> ComposeResult:
            yield from ()

    app = TestApp()
    async with app.run_test() as pilot:
        modal = BrowseModal(
            title="Test",
            start_path=tmp_path,
            select_type="directory",
        )
        await app.push_screen(modal)
        await pilot.pause()
        drive_select = app.screen.query_one("#browse-drive-select", Select)
        assert drive_select is not None


@pytest.mark.asyncio
@pytest.mark.skipif(sys.platform == "win32", reason="Unix-only: no drive selector")
async def test_browse_modal_no_drive_selector_on_unix(tmp_path):
    """On Unix, BrowseModal should NOT show a drive selector."""
    from textual.app import App, ComposeResult
    from textual.widgets import Select

    class TestApp(App):
        def compose(self) -> ComposeResult:
            yield from ()

    app = TestApp()
    async with app.run_test() as pilot:
        modal = BrowseModal(
            title="Test",
            start_path=tmp_path,
            select_type="directory",
        )
        await app.push_screen(modal)
        await pilot.pause()
        results = app.screen.query("#browse-drive-select")
        assert len(results) == 0
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_browse_modal.py -v`
Expected: `test_browse_modal_roots_at_drive_root` FAILS (tree starts at `tmp_path`, not root)

**Step 3: Implement BrowseModal changes**

Rewrite `scropipe/tui/browse_modal.py`:

```python
"""Reusable file/directory browser modal using Textual's DirectoryTree."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal, Optional

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, DirectoryTree, Label, Select


def _get_drive_root(path: Path) -> Path:
    """Get the root/drive of a path (e.g. C:\\ on Windows, / on Unix)."""
    return Path(path.anchor)


def _list_windows_drives() -> list[str]:
    """List available drive letters on Windows."""
    drives = []
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        drive = f"{letter}:\\"
        if Path(drive).exists():
            drives.append(drive)
    return drives


class _FilteredDirectoryTree(DirectoryTree):
    """DirectoryTree that can optionally filter to directories only."""

    def __init__(self, path: str | Path, select_type: str = "file", **kwargs):
        super().__init__(path, **kwargs)
        self._select_type = select_type

    def filter_paths(self, paths):
        if self._select_type == "directory":
            return [p for p in paths if p.is_dir()]
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

    BrowseModal #browse-drive-select {
        width: 12;
        margin-bottom: 1;
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
        root = _get_drive_root(self._start_path.resolve())
        with Vertical():
            yield Label(self._title_text)
            if sys.platform == "win32":
                drives = _list_windows_drives()
                current_drive = str(root)
                options = [(d, d) for d in drives]
                yield Select(
                    options,
                    value=current_drive,
                    id="browse-drive-select",
                    prompt="Drive",
                )
            yield _FilteredDirectoryTree(
                root,
                select_type=self._select_type,
                id="browse-tree",
            )
            yield Label("No selection", id="browse-selected-path")
            with Horizontal(classes="action-bar"):
                yield Button("OK", variant="primary", id="browse-ok-btn")
                yield Button("Cancel", id="browse-cancel-btn")

    def on_mount(self) -> None:
        """Expand the tree to the start path after mounting."""
        self._expand_to_path(self._start_path.resolve())

    async def _expand_to_path(self, target: Path) -> None:
        """Expand tree nodes from root down to the target directory."""
        tree = self.query_one("#browse-tree", DirectoryTree)
        # Build the chain of directories from root to target
        parts = []
        current = target
        root = _get_drive_root(target)
        while current != root:
            parts.append(current)
            current = current.parent
        parts.reverse()
        # Expand each level — the tree needs time to load children
        for part in parts:
            tree.path = root  # ensure tree is rooted correctly
            node = tree.root
            node.expand()

    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle drive selector change on Windows."""
        if event.select.id == "browse-drive-select" and event.value != Select.BLANK:
            new_root = Path(str(event.value))
            old_tree = self.query_one("#browse-tree", _FilteredDirectoryTree)
            new_tree = _FilteredDirectoryTree(
                new_root,
                select_type=self._select_type,
                id="browse-tree",
            )
            old_tree.replace(new_tree)

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
Expected: All PASS

**Step 5: Commit**

```bash
git add scropipe/tui/browse_modal.py tests/test_browse_modal.py
git commit -m "feat: root BrowseModal at drive root with Windows drive selector"
```

---

### Task 4: Integration smoke test

**Files:**
- Modify: `tests/test_tui_integration.py` (or create if needed)

**Step 1: Run the full test suite**

Run: `python -m pytest tests/ -v`
Expected: All PASS

**Step 2: Manual smoke test**

Run: `python -m scropipe.tui`

Verify:
- [ ] Tab completes paths in Split tab source/output inputs
- [ ] Tab cycles through multiple matches on repeated presses
- [ ] Tab does NOT steal focus on non-path inputs (sensitivity, BPM, etc.)
- [ ] Browse modal opens rooted at drive (e.g. `C:\`)
- [ ] Tree is pre-expanded to the start directory
- [ ] Drive selector dropdown appears and switching drives works
- [ ] Browse modal OK/Cancel still work

**Step 3: Commit any fixes if needed**

```bash
git add -u
git commit -m "fix: integration fixes for path input UX"
```
