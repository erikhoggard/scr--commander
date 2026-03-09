"""Pool tab widget for the scropipe TUI."""

from __future__ import annotations

from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, ListItem, ListView, Static

from .browse_modal import BrowseModal


class NewPoolModal(ModalScreen[str | None]):
    """Modal dialog for creating a new pool."""

    DEFAULT_CSS = """
    NewPoolModal {
        align: center middle;
    }

    NewPoolModal > Vertical {
        width: 50;
        height: auto;
        max-height: 14;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }

    NewPoolModal Input {
        margin-bottom: 1;
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("New Pool")
            yield Label("Pool name:")
            yield Input(placeholder="Enter pool name", id="new-pool-name")
            with Horizontal(classes="action-bar"):
                yield Button("Create", variant="primary", id="create-pool-btn")
                yield Button("Cancel", id="cancel-pool-btn")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "create-pool-btn":
            name = self.query_one("#new-pool-name", Input).value.strip()
            self.dismiss(name if name else None)
        elif event.button.id == "cancel-pool-btn":
            self.dismiss(None)


class PoolTab(Static):
    """Pool management interface tab."""

    DEFAULT_CSS = """
    PoolTab {
        height: 1fr;
        padding: 1 2;
    }

    #pool-sidebar {
        width: 30;
        height: 1fr;
        border-right: tall $accent;
        padding-right: 1;
    }

    #pool-detail {
        height: 1fr;
        padding-left: 2;
    }

    #pool-list {
        height: 1fr;
        margin-bottom: 1;
    }

    #pool-detail-info {
        margin-bottom: 1;
    }

    #pool-sources-list {
        margin-bottom: 1;
    }
    """

    def compose(self) -> ComposeResult:
        with Horizontal():
            with Vertical(id="pool-sidebar"):
                yield Label("Pools", classes="section-title")
                yield ListView(id="pool-list")
                yield Button("New Pool", variant="primary", id="new-pool-btn")
            with Vertical(id="pool-detail"):
                yield Label("Select a pool", id="pool-detail-title")
                yield Static("", id="pool-detail-info")
                yield Static("", id="pool-sources-list")
                with Horizontal(classes="action-bar"):
                    yield Button("Add Files", id="add-files-btn")
                    yield Button("Add Directory", id="add-dir-btn")
                    yield Button("Delete Pool", variant="error", id="delete-pool-btn")
                    yield Button("Train", variant="success", id="pool-train-btn")

    def on_mount(self) -> None:
        """Refresh pool list on mount."""
        self._refresh_pool_list()

    def _get_pool_manager(self):
        from ..pool_manager import PoolManager

        return PoolManager(
            self.app.pools_dir or Path.home() / ".local/share/scropipe/pools"
        )

    def _refresh_pool_list(self) -> None:
        """Reload the pool list from disk."""
        pool_list = self.query_one("#pool-list", ListView)
        pool_list.clear()
        try:
            pm = self._get_pool_manager()
            pools = pm.list_pools()
            for pool in pools:
                item = ListItem(
                    Label(f"{pool.name} ({pool.sample_count} samples)"),
                )
                item._pool_name = pool.name  # type: ignore[attr-defined]
                pool_list.append(item)
        except Exception:
            pass

    def _get_selected_pool_name(self) -> str | None:
        """Get the name of the currently selected pool."""
        pool_list = self.query_one("#pool-list", ListView)
        if pool_list.highlighted_child is not None:
            return getattr(pool_list.highlighted_child, "_pool_name", None)
        return None

    def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        """Show details when a pool is selected."""
        if event.item is None:
            return
        pool_name = getattr(event.item, "_pool_name", None)
        if pool_name is None:
            return
        self._show_pool_details(pool_name)

    def _show_pool_details(self, pool_name: str) -> None:
        """Display details for the given pool."""
        title = self.query_one("#pool-detail-title", Label)
        info = self.query_one("#pool-detail-info", Static)
        sources_list = self.query_one("#pool-sources-list", Static)

        try:
            pm = self._get_pool_manager()
            pool = pm.get_pool(pool_name)
        except KeyError:
            title.update(f"Pool not found: {pool_name}")
            info.update("")
            sources_list.update("")
            return

        title.update(pool.name)
        info.update(f"Samples: {pool.sample_count}")

        # Group sources by type
        if pool.sources:
            grouped: dict[str, list] = {}
            for src in pool.sources:
                grouped.setdefault(src.source_type, []).append(src)

            lines = ["Sources:"]
            for stype, srcs in grouped.items():
                lines.append(f"  [{stype}]")
                for s in srcs:
                    lines.append(f"    {s.path} ({s.count} files)")
            sources_list.update("\n".join(lines))
        else:
            sources_list.update("No sources yet.")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "new-pool-btn":
            self.app.push_screen(
                NewPoolModal(), callback=self._on_new_pool_dismiss
            )
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
        elif event.button.id == "delete-pool-btn":
            self._delete_selected_pool()
        elif event.button.id == "pool-train-btn":
            self._switch_to_train()

    def _on_new_pool_dismiss(self, name: str | None) -> None:
        """Handle new pool modal result."""
        if name is None:
            return
        try:
            pm = self._get_pool_manager()
            pm.create_pool(name)
            self._refresh_pool_list()
        except ValueError:
            pass  # Pool already exists

    def _on_add_files_dismiss(self, path: Path | None) -> None:
        """Handle add files modal result."""
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
        """Handle add directory modal result."""
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

    def _delete_selected_pool(self) -> None:
        """Delete the currently selected pool."""
        pool_name = self._get_selected_pool_name()
        if pool_name is None:
            return
        try:
            pm = self._get_pool_manager()
            pm.delete_pool(pool_name)
            self._refresh_pool_list()
            self.query_one("#pool-detail-title", Label).update("Select a pool")
            self.query_one("#pool-detail-info", Static).update("")
            self.query_one("#pool-sources-list", Static).update("")
        except KeyError:
            pass

    def _switch_to_train(self) -> None:
        """Switch to the train tab."""
        from textual.widgets import TabbedContent

        self.app.query_one(TabbedContent).active = "tab-train"
