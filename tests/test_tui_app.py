import pytest
from scropipe.tui.app import ScropipeApp


@pytest.fixture
def app(tmp_path):
    return ScropipeApp(
        models_dir=tmp_path / "models",
        pools_dir=tmp_path / "pools",
    )


@pytest.mark.asyncio
async def test_app_has_four_tabs(app):
    async with app.run_test() as pilot:
        tabs = app.query("Tab")
        tab_labels = [t.label.plain for t in tabs]
        assert "Split" in tab_labels
        assert "Pool" in tab_labels
        assert "Train" in tab_labels
        assert "Generate" in tab_labels


@pytest.mark.asyncio
async def test_app_has_status_bar(app):
    async with app.run_test() as pilot:
        footer = app.query_one("#status-bar")
        assert footer is not None
