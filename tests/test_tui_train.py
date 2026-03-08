import pytest

from scropipe.tui.app import ScropipeApp


@pytest.fixture
def app(tmp_path):
    (tmp_path / "pools").mkdir()
    (tmp_path / "models").mkdir()
    return ScropipeApp(models_dir=tmp_path / "models", pools_dir=tmp_path / "pools")


@pytest.mark.asyncio
async def test_train_tab_has_pool_selector(app):
    async with app.run_test() as pilot:
        tabbed = app.query_one("TabbedContent")
        tabbed.active = "tab-train"
        await pilot.pause()
        pool_select = app.query_one("#train-pool-select")
        assert pool_select is not None


@pytest.mark.asyncio
async def test_train_tab_has_model_name_input(app):
    async with app.run_test() as pilot:
        tabbed = app.query_one("TabbedContent")
        tabbed.active = "tab-train"
        await pilot.pause()
        name_input = app.query_one("#train-model-name")
        assert name_input is not None


@pytest.mark.asyncio
async def test_train_tab_has_start_button(app):
    async with app.run_test() as pilot:
        tabbed = app.query_one("TabbedContent")
        tabbed.active = "tab-train"
        await pilot.pause()
        btn = app.query_one("#start-training-btn")
        assert btn is not None


@pytest.mark.asyncio
async def test_train_tab_has_stop_condition_radio(app):
    async with app.run_test() as pilot:
        tabbed = app.query_one("TabbedContent")
        tabbed.active = "tab-train"
        await pilot.pause()
        radio_set = app.query_one("#stop-condition")
        assert radio_set is not None


@pytest.mark.asyncio
async def test_train_tab_has_arch_selector(app):
    async with app.run_test() as pilot:
        tabbed = app.query_one("TabbedContent")
        tabbed.active = "tab-train"
        await pilot.pause()
        arch_select = app.query_one("#train-arch-select")
        assert arch_select is not None


@pytest.mark.asyncio
async def test_train_tab_has_checkpoint_interval(app):
    async with app.run_test() as pilot:
        tabbed = app.query_one("TabbedContent")
        tabbed.active = "tab-train"
        await pilot.pause()
        val_every = app.query_one("#train-val-every")
        assert val_every is not None


@pytest.mark.asyncio
async def test_train_tab_has_gpu_info(app):
    async with app.run_test() as pilot:
        tabbed = app.query_one("TabbedContent")
        tabbed.active = "tab-train"
        await pilot.pause()
        gpu_info = app.query_one("#train-gpu-info")
        assert gpu_info is not None


@pytest.mark.asyncio
async def test_train_tab_conditional_inputs_hidden(app):
    async with app.run_test() as pilot:
        tabbed = app.query_one("TabbedContent")
        tabbed.active = "tab-train"
        await pilot.pause()
        max_steps = app.query_one("#train-max-steps")
        delta_target = app.query_one("#train-delta-target")
        assert max_steps.has_class("hidden")
        assert delta_target.has_class("hidden")


@pytest.mark.asyncio
async def test_train_tab_dashboard_hidden_initially(app):
    async with app.run_test() as pilot:
        tabbed = app.query_one("TabbedContent")
        tabbed.active = "tab-train"
        await pilot.pause()
        dashboard = app.query_one("#train-dashboard")
        assert dashboard.display is False


@pytest.mark.asyncio
async def test_train_tab_dashboard_has_controls(app):
    async with app.run_test() as pilot:
        tabbed = app.query_one("TabbedContent")
        tabbed.active = "tab-train"
        await pilot.pause()
        # Dashboard widgets exist even if hidden
        assert app.query_one("#dash-title") is not None
        assert app.query_one("#dash-info") is not None
        assert app.query_one("#dash-metrics") is not None
        assert app.query_one("#dash-sparkline") is not None
        assert app.query_one("#dash-timing") is not None
        assert app.query_one("#dash-checkpoint") is not None
        assert app.query_one("#stop-training-btn") is not None


@pytest.mark.asyncio
async def test_train_tab_has_resume_button(app):
    async with app.run_test() as pilot:
        tabbed = app.query_one("TabbedContent")
        tabbed.active = "tab-train"
        await pilot.pause()
        btn = app.query_one("#resume-training-btn")
        assert btn is not None


@pytest.mark.asyncio
async def test_train_tab_dashboard_has_single_stop_button(app):
    async with app.run_test() as pilot:
        tabbed = app.query_one("TabbedContent")
        tabbed.active = "tab-train"
        await pilot.pause()
        assert app.query_one("#stop-training-btn") is not None
        assert len(app.query("#stop-save-btn")) == 0
        assert len(app.query("#stop-discard-btn")) == 0
