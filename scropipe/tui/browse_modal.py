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
