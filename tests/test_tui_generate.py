import json
import pytest
from pathlib import Path
from scropipe.tui.app import ScropipeApp


@pytest.fixture
def app(tmp_path):
    models_dir = tmp_path / "models"
    pools_dir = tmp_path / "pools"
    models_dir.mkdir()
    pools_dir.mkdir()
    # Create a fake model for testing
    model_dir = models_dir / "drums-v1"
    model_dir.mkdir()
    (model_dir / "model.ts").write_text("fake")
    with open(model_dir / "metadata.json", "w") as f:
        json.dump({"name": "drums-v1", "created": "", "config": "v2", "total_samples": 10}, f)
    return ScropipeApp(models_dir=models_dir, pools_dir=pools_dir)


@pytest.mark.asyncio
async def test_generate_tab_has_model_selector(app):
    async with app.run_test() as pilot:
        tabbed = app.query_one("TabbedContent")
        tabbed.active = "tab-generate"
        await pilot.pause()
        select = app.query_one("#gen-model-select")
        assert select is not None


@pytest.mark.asyncio
async def test_generate_tab_has_input_output_fields(app):
    async with app.run_test() as pilot:
        tabbed = app.query_one("TabbedContent")
        tabbed.active = "tab-generate"
        await pilot.pause()
        assert app.query_one("#gen-input-dir") is not None
        assert app.query_one("#gen-output-dir") is not None


@pytest.mark.asyncio
async def test_generate_tab_has_generate_button(app):
    async with app.run_test() as pilot:
        tabbed = app.query_one("TabbedContent")
        tabbed.active = "tab-generate"
        await pilot.pause()
        btn = app.query_one("#generate-btn")
        assert btn is not None


@pytest.mark.asyncio
async def test_generate_tab_has_delete_model_button(app):
    async with app.run_test() as pilot:
        tabbed = app.query_one("TabbedContent")
        tabbed.active = "tab-generate"
        await pilot.pause()
        btn = app.query_one("#gen-delete-model-btn")
        assert btn is not None


@pytest.mark.asyncio
async def test_generate_tab_has_models_table(app):
    async with app.run_test() as pilot:
        tabbed = app.query_one("TabbedContent")
        tabbed.active = "tab-generate"
        await pilot.pause()
        table = app.query_one("#gen-models-table")
        assert table is not None


@pytest.mark.asyncio
async def test_generate_tab_has_status(app):
    async with app.run_test() as pilot:
        tabbed = app.query_one("TabbedContent")
        tabbed.active = "tab-generate"
        await pilot.pause()
        status = app.query_one("#gen-status")
        assert status is not None


@pytest.mark.asyncio
async def test_generate_tab_has_progress_bar(app):
    async with app.run_test() as pilot:
        tabbed = app.query_one("TabbedContent")
        tabbed.active = "tab-generate"
        await pilot.pause()
        progress = app.query_one("#gen-progress")
        assert progress is not None
        # Progress bar should be hidden initially
        assert progress.display is False


@pytest.mark.asyncio
async def test_generate_tab_has_input_info(app):
    async with app.run_test() as pilot:
        tabbed = app.query_one("TabbedContent")
        tabbed.active = "tab-generate"
        await pilot.pause()
        info = app.query_one("#gen-input-info")
        assert info is not None
