import pytest
from scropipe.tui.app import ScropipeApp


@pytest.fixture
def app(tmp_path):
    (tmp_path / "pools").mkdir()
    (tmp_path / "models").mkdir()
    return ScropipeApp(models_dir=tmp_path / "models", pools_dir=tmp_path / "pools")


@pytest.mark.asyncio
async def test_pool_tab_has_pool_list(app):
    async with app.run_test() as pilot:
        tabbed = app.query_one("TabbedContent")
        tabbed.active = "tab-pool"
        await pilot.pause()
        pool_list = app.query_one("#pool-list")
        assert pool_list is not None


@pytest.mark.asyncio
async def test_pool_tab_has_new_pool_button(app):
    async with app.run_test() as pilot:
        tabbed = app.query_one("TabbedContent")
        tabbed.active = "tab-pool"
        await pilot.pause()
        btn = app.query_one("#new-pool-btn")
        assert btn is not None


@pytest.mark.asyncio
async def test_pool_tab_has_detail_panel(app):
    async with app.run_test() as pilot:
        tabbed = app.query_one("TabbedContent")
        tabbed.active = "tab-pool"
        await pilot.pause()
        title = app.query_one("#pool-detail-title")
        assert title is not None
        info = app.query_one("#pool-detail-info")
        assert info is not None


@pytest.mark.asyncio
async def test_pool_tab_has_action_buttons(app):
    async with app.run_test() as pilot:
        tabbed = app.query_one("TabbedContent")
        tabbed.active = "tab-pool"
        await pilot.pause()
        assert app.query_one("#add-files-btn") is not None
        assert app.query_one("#add-dir-btn") is not None
        assert app.query_one("#delete-pool-btn") is not None
        assert app.query_one("#pool-train-btn") is not None


@pytest.mark.asyncio
async def test_pool_tab_has_sources_list(app):
    async with app.run_test() as pilot:
        tabbed = app.query_one("TabbedContent")
        tabbed.active = "tab-pool"
        await pilot.pause()
        sources = app.query_one("#pool-sources-list")
        assert sources is not None


@pytest.mark.asyncio
async def test_pool_add_files_opens_browse_modal(app):
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
        await pilot.click("#add-files-btn")
        await pilot.pause()
        from scropipe.tui.browse_modal import BrowseModal
        assert any(isinstance(s, BrowseModal) for s in app.screen_stack)


@pytest.mark.asyncio
async def test_pool_add_dir_opens_browse_modal(app):
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
