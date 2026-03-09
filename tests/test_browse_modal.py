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
    async with app.run_test() as pilot:
        modal = BrowseModal(
            title="Test Browse",
            start_path=tmp_path,
            select_type="directory",
        )
        await app.push_screen(modal)
        await pilot.pause()
        tree = app.screen.query_one("DirectoryTree")
        assert tree is not None
        ok_btn = app.screen.query_one("#browse-ok-btn")
        assert ok_btn is not None
        cancel_btn = app.screen.query_one("#browse-cancel-btn")
        assert cancel_btn is not None


@pytest.mark.asyncio
async def test_browse_modal_cancel_returns_none(tmp_path):
    """Pressing Cancel should dismiss with None."""
    from textual.app import App, ComposeResult

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
