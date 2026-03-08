import pytest
from scropipe.tui.app import ScropipeApp, SetupModal


@pytest.mark.asyncio
async def test_first_run_shows_setup_modal(tmp_path):
    app = ScropipeApp(models_dir=None, pools_dir=None)
    async with app.run_test() as pilot:
        assert isinstance(app.screen, SetupModal)
