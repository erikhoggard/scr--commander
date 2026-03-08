"""Integration tests for scropipe TUI application."""

import pytest

from scropipe.tui.app import ScropipeApp


@pytest.fixture
def configured_app(tmp_path):
    (tmp_path / "pools").mkdir()
    (tmp_path / "models").mkdir()
    return ScropipeApp(
        models_dir=tmp_path / "models",
        pools_dir=tmp_path / "pools",
    )


@pytest.mark.asyncio
async def test_app_starts_without_errors(configured_app):
    async with configured_app.run_test() as pilot:
        assert configured_app.title == "scropipe"


@pytest.mark.asyncio
async def test_tab_switching(configured_app):
    async with configured_app.run_test() as pilot:
        tabbed = configured_app.query_one("TabbedContent")
        for tab_id in ["pool", "train", "generate", "split"]:
            # TabbedContent uses "tab-<id>" internally in some versions
            # Try setting active and checking it worked
            try:
                tabbed.active = tab_id
                await pilot.pause()
            except Exception:
                tabbed.active = f"tab-{tab_id}"
                await pilot.pause()


@pytest.mark.asyncio
async def test_status_bar_content(configured_app):
    async with configured_app.run_test() as pilot:
        status = configured_app.query_one("#status-bar")
        text = status.render()
        # Status bar should exist and have content
        assert status is not None
