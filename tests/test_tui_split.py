import pytest
from scropipe.tui.app import ScropipeApp


@pytest.fixture
def app(tmp_path):
    return ScropipeApp(
        models_dir=tmp_path / "models",
        pools_dir=tmp_path / "pools",
    )


@pytest.mark.asyncio
async def test_split_tab_has_mode_selector(app):
    async with app.run_test() as pilot:
        tabbed = app.query_one("TabbedContent")
        tabbed.active = "tab-split"
        await pilot.pause()
        radio_set = app.query("RadioSet")
        assert len(radio_set) >= 1


@pytest.mark.asyncio
async def test_split_tab_has_source_input(app):
    async with app.run_test() as pilot:
        tabbed = app.query_one("TabbedContent")
        tabbed.active = "tab-split"
        await pilot.pause()
        source_input = app.query_one("#split-source-input")
        assert source_input is not None


@pytest.mark.asyncio
async def test_split_tab_has_action_buttons(app):
    async with app.run_test() as pilot:
        tabbed = app.query_one("TabbedContent")
        tabbed.active = "tab-split"
        await pilot.pause()
        split_btn = app.query_one("#split-btn")
        assert split_btn is not None


@pytest.mark.asyncio
async def test_split_tab_has_output_input(app):
    async with app.run_test() as pilot:
        tabbed = app.query_one("TabbedContent")
        tabbed.active = "tab-split"
        await pilot.pause()
        output_input = app.query_one("#split-output-input")
        assert output_input is not None


@pytest.mark.asyncio
async def test_split_tab_has_status(app):
    async with app.run_test() as pilot:
        tabbed = app.query_one("TabbedContent")
        tabbed.active = "tab-split"
        await pilot.pause()
        status = app.query_one("#split-status")
        assert status is not None


@pytest.mark.asyncio
async def test_split_tab_has_split_and_pool_button(app):
    async with app.run_test() as pilot:
        tabbed = app.query_one("TabbedContent")
        tabbed.active = "tab-split"
        await pilot.pause()
        btn = app.query_one("#split-and-pool-btn")
        assert btn is not None
