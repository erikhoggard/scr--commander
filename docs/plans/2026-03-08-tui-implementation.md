# Scropipe TUI Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a Textual-based TUI as the primary interface, with four tabs (Split, Pool, Train, Generate), preserving the existing CLI.

**Architecture:** Single Textual app (`ScropipeApp`) with `TabbedContent` for four phases. New `config.py` handles persistent settings. New `PoolManager` and `ModelManager` classes provide CRUD for pools/models. The TUI is a pure presentation layer on top of existing `scropipe/stages/` code.

**Tech Stack:** Textual (TUI framework), tomli/tomllib (TOML config), existing Rich/Typer CLI, existing PyTorch/RAVE stages.

---

## Task 1: Add Textual Dependency

**Files:**
- Modify: `pyproject.toml`

**Step 1: Add textual to dependencies**

In `pyproject.toml`, add `textual` to the dependencies list:

```toml
dependencies = [
    # CLI
    "typer>=0.12.0",
    "rich>=13.7.0",
    # TUI
    "textual>=1.0.0",
    # Splitter (lightweight)
    "numpy>=1.26.0",
    "scipy>=1.12.0",
    "soundfile>=0.12.0",
]
```

Also add `textual-dev` to the dev dependencies for testing:

```toml
dev = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.23.0",
    "ruff>=0.1.0",
    "textual-dev>=1.0.0",
]
```

**Step 2: Install dependencies**

Run: `pip install -e ".[dev]"`

**Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "feat: add textual dependency for TUI"
```

---

## Task 2: Config Module

**Files:**
- Create: `scropipe/config.py`
- Create: `tests/test_config.py`

**Step 1: Write the failing test**

```python
"""Tests for scropipe config module."""

import pytest
from pathlib import Path
from scropipe.config import ScropipeConfig, load_config, save_config


def test_load_config_returns_defaults_when_no_file(tmp_path):
    config = load_config(tmp_path / "nonexistent.toml")
    assert config.models_dir is None
    assert config.pools_dir is None
    assert config.presets_dir is None


def test_save_and_load_config_roundtrip(tmp_path):
    config_path = tmp_path / "config.toml"
    config = ScropipeConfig(
        models_dir=Path("/data/models"),
        pools_dir=Path("/data/pools"),
    )
    save_config(config, config_path)
    loaded = load_config(config_path)
    assert loaded.models_dir == Path("/data/models")
    assert loaded.pools_dir == Path("/data/pools")
    assert loaded.presets_dir is None


def test_config_needs_setup_when_dirs_not_set():
    config = ScropipeConfig()
    assert config.needs_setup is True


def test_config_needs_setup_false_when_dirs_set():
    config = ScropipeConfig(
        models_dir=Path("/data/models"),
        pools_dir=Path("/data/pools"),
    )
    assert config.needs_setup is False


def test_default_config_path():
    from scropipe.config import default_config_path
    path = default_config_path()
    assert path.name == "config.toml"
    assert "scropipe" in str(path)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL (module not found)

**Step 3: Write the implementation**

```python
"""Scropipe configuration management."""

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class ScropipeConfig:
    """Application configuration."""

    models_dir: Optional[Path] = None
    pools_dir: Optional[Path] = None
    presets_dir: Optional[Path] = None

    @property
    def needs_setup(self) -> bool:
        return self.models_dir is None or self.pools_dir is None


def default_config_path() -> Path:
    """Return the default config file path."""
    return Path.home() / ".config" / "scropipe" / "config.toml"


def load_config(path: Optional[Path] = None) -> ScropipeConfig:
    """Load config from TOML file. Returns defaults if file doesn't exist."""
    path = path or default_config_path()
    if not path.exists():
        return ScropipeConfig()

    with open(path, "rb") as f:
        data = tomllib.load(f)

    return ScropipeConfig(
        models_dir=Path(data["models_dir"]) if data.get("models_dir") else None,
        pools_dir=Path(data["pools_dir"]) if data.get("pools_dir") else None,
        presets_dir=Path(data["presets_dir"]) if data.get("presets_dir") else None,
    )


def save_config(config: ScropipeConfig, path: Optional[Path] = None) -> None:
    """Save config to TOML file."""
    path = path or default_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    if config.models_dir:
        lines.append(f'models_dir = "{config.models_dir}"')
    if config.pools_dir:
        lines.append(f'pools_dir = "{config.pools_dir}"')
    if config.presets_dir:
        lines.append(f'presets_dir = "{config.presets_dir}"')

    path.write_text("\n".join(lines) + "\n")
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py -v`
Expected: All 5 PASS

**Step 5: Commit**

```bash
git add scropipe/config.py tests/test_config.py
git commit -m "feat: add config module for persistent settings"
```

---

## Task 3: Pool Manager

**Files:**
- Create: `scropipe/pool_manager.py`
- Create: `tests/test_pool_manager.py`

The pool manager handles CRUD for named sample pools. Each pool is a directory containing `pool.json` (metadata) and `samples/` (WAV files). Pools can aggregate files from multiple sources.

**Step 1: Write the failing test**

```python
"""Tests for pool manager."""

import json
import pytest
from pathlib import Path
from scropipe.pool_manager import PoolManager, PoolInfo, PoolSource


@pytest.fixture
def pool_mgr(tmp_path):
    return PoolManager(tmp_path)


@pytest.fixture
def sample_wav(tmp_path):
    """Create a minimal WAV file for testing."""
    import struct
    wav_path = tmp_path / "test_sample.wav"
    # Minimal valid WAV: 44-byte header + 2 bytes of silence
    sr = 44100
    data = struct.pack("<h", 0) * 100
    data_size = len(data)
    with open(wav_path, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + data_size))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))  # chunk size
        f.write(struct.pack("<H", 1))   # PCM
        f.write(struct.pack("<H", 1))   # mono
        f.write(struct.pack("<I", sr))  # sample rate
        f.write(struct.pack("<I", sr * 2))  # byte rate
        f.write(struct.pack("<H", 2))   # block align
        f.write(struct.pack("<H", 16))  # bits per sample
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        f.write(data)
    return wav_path


def test_list_pools_empty(pool_mgr):
    assert pool_mgr.list_pools() == []


def test_create_pool(pool_mgr):
    pool_mgr.create_pool("drums")
    pools = pool_mgr.list_pools()
    assert len(pools) == 1
    assert pools[0].name == "drums"
    assert pools[0].sample_count == 0


def test_create_duplicate_pool_raises(pool_mgr):
    pool_mgr.create_pool("drums")
    with pytest.raises(ValueError, match="already exists"):
        pool_mgr.create_pool("drums")


def test_add_files_to_pool(pool_mgr, sample_wav):
    pool_mgr.create_pool("drums")
    pool_mgr.add_files("drums", [sample_wav])
    info = pool_mgr.get_pool("drums")
    assert info.sample_count == 1
    assert len(info.sources) == 1
    assert info.sources[0].source_type == "files"


def test_add_directory_to_pool(pool_mgr, sample_wav, tmp_path):
    # Create a directory with WAV files
    src_dir = tmp_path / "src_samples"
    src_dir.mkdir()
    import shutil
    shutil.copy2(sample_wav, src_dir / "kick.wav")
    shutil.copy2(sample_wav, src_dir / "snare.wav")

    pool_mgr.create_pool("drums")
    pool_mgr.add_directory("drums", src_dir)
    info = pool_mgr.get_pool("drums")
    assert info.sample_count == 2
    assert len(info.sources) == 1
    assert info.sources[0].source_type == "directory"
    assert info.sources[0].count == 2


def test_delete_pool(pool_mgr):
    pool_mgr.create_pool("drums")
    pool_mgr.delete_pool("drums")
    assert pool_mgr.list_pools() == []


def test_get_nonexistent_pool_raises(pool_mgr):
    with pytest.raises(KeyError):
        pool_mgr.get_pool("nope")


def test_pool_samples_dir(pool_mgr):
    pool_mgr.create_pool("drums")
    samples_dir = pool_mgr.get_samples_dir("drums")
    assert samples_dir.exists()
    assert samples_dir.name == "samples"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_pool_manager.py -v`
Expected: FAIL (module not found)

**Step 3: Write the implementation**

```python
"""Pool manager - CRUD operations for named sample pools."""

import json
import shutil
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path


@dataclass
class PoolSource:
    """Record of a source added to a pool."""

    source_type: str  # "files", "directory", "split"
    path: str
    count: int
    added_at: str


@dataclass
class PoolInfo:
    """Metadata about a pool."""

    name: str
    created_at: str
    sample_count: int = 0
    sources: list[PoolSource] = field(default_factory=list)


class PoolManager:
    """Manages named sample pools on disk."""

    def __init__(self, pools_dir: Path):
        self.pools_dir = Path(pools_dir)
        self.pools_dir.mkdir(parents=True, exist_ok=True)

    def list_pools(self) -> list[PoolInfo]:
        """List all pools."""
        pools = []
        if not self.pools_dir.exists():
            return pools
        for pool_dir in sorted(self.pools_dir.iterdir()):
            meta_path = pool_dir / "pool.json"
            if pool_dir.is_dir() and meta_path.exists():
                pools.append(self._load_pool_info(pool_dir))
        return pools

    def create_pool(self, name: str) -> PoolInfo:
        """Create a new empty pool."""
        pool_dir = self.pools_dir / name
        if pool_dir.exists():
            raise ValueError(f"Pool '{name}' already exists")
        pool_dir.mkdir(parents=True)
        (pool_dir / "samples").mkdir()

        info = PoolInfo(name=name, created_at=datetime.now().isoformat())
        self._save_pool_info(pool_dir, info)
        return info

    def get_pool(self, name: str) -> PoolInfo:
        """Get pool info by name."""
        pool_dir = self.pools_dir / name
        if not pool_dir.exists():
            raise KeyError(f"Pool '{name}' not found")
        return self._load_pool_info(pool_dir)

    def get_samples_dir(self, name: str) -> Path:
        """Get the samples directory for a pool."""
        pool_dir = self.pools_dir / name
        if not pool_dir.exists():
            raise KeyError(f"Pool '{name}' not found")
        return pool_dir / "samples"

    def add_files(self, name: str, files: list[Path]) -> int:
        """Add individual files to a pool. Returns count added."""
        pool_dir = self.pools_dir / name
        samples_dir = pool_dir / "samples"
        info = self._load_pool_info(pool_dir)
        count = 0

        for f in files:
            f = Path(f)
            if f.exists() and f.suffix.lower() == ".wav":
                dest = samples_dir / f.name
                # Handle duplicates
                counter = 1
                while dest.exists():
                    dest = samples_dir / f"{f.stem}_{counter}.wav"
                    counter += 1
                shutil.copy2(f, dest)
                count += 1

        if count > 0:
            source = PoolSource(
                source_type="files",
                path=str(files[0].parent) if files else "",
                count=count,
                added_at=datetime.now().isoformat(),
            )
            info.sources.append(source)
            info.sample_count += count
            self._save_pool_info(pool_dir, info)

        return count

    def add_directory(self, name: str, directory: Path) -> int:
        """Add all WAV files from a directory to a pool. Returns count added."""
        pool_dir = self.pools_dir / name
        samples_dir = pool_dir / "samples"
        info = self._load_pool_info(pool_dir)

        wav_files = list(Path(directory).rglob("*.wav"))
        count = 0

        for f in wav_files:
            dest = samples_dir / f.name
            counter = 1
            while dest.exists():
                dest = samples_dir / f"{f.stem}_{counter}.wav"
                counter += 1
            shutil.copy2(f, dest)
            count += 1

        if count > 0:
            source = PoolSource(
                source_type="directory",
                path=str(directory),
                count=count,
                added_at=datetime.now().isoformat(),
            )
            info.sources.append(source)
            info.sample_count += count
            self._save_pool_info(pool_dir, info)

        return count

    def delete_pool(self, name: str) -> None:
        """Delete a pool and all its files."""
        pool_dir = self.pools_dir / name
        if pool_dir.exists():
            shutil.rmtree(pool_dir)

    def _load_pool_info(self, pool_dir: Path) -> PoolInfo:
        meta_path = pool_dir / "pool.json"
        with open(meta_path) as f:
            data = json.load(f)
        sources = [PoolSource(**s) for s in data.get("sources", [])]
        return PoolInfo(
            name=data["name"],
            created_at=data["created_at"],
            sample_count=data.get("sample_count", 0),
            sources=sources,
        )

    def _save_pool_info(self, pool_dir: Path, info: PoolInfo) -> None:
        meta_path = pool_dir / "pool.json"
        data = {
            "name": info.name,
            "created_at": info.created_at,
            "sample_count": info.sample_count,
            "sources": [asdict(s) for s in info.sources],
        }
        with open(meta_path, "w") as f:
            json.dump(data, f, indent=2)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_pool_manager.py -v`
Expected: All 9 PASS

**Step 5: Commit**

```bash
git add scropipe/pool_manager.py tests/test_pool_manager.py
git commit -m "feat: add pool manager for named sample pool CRUD"
```

---

## Task 4: Model Manager

**Files:**
- Create: `scropipe/model_manager.py`
- Create: `tests/test_model_manager.py`

The model manager lists and manages trained models from the configured models directory. Models are directories containing `metadata.json` and `model.ts`.

**Step 1: Write the failing test**

```python
"""Tests for model manager."""

import json
import pytest
from pathlib import Path
from scropipe.model_manager import ModelManager, ModelInfo


@pytest.fixture
def model_mgr(tmp_path):
    return ModelManager(tmp_path)


def _create_fake_model(models_dir: Path, name: str, **kwargs) -> Path:
    """Create a fake model directory with metadata."""
    model_dir = models_dir / name
    model_dir.mkdir(parents=True)
    # Fake model file
    (model_dir / "model.ts").write_text("fake")
    # Metadata
    metadata = {
        "name": name,
        "created": "2026-03-08T12:00:00",
        "config": kwargs.get("config", "v2"),
        "total_samples": kwargs.get("total_samples", 47),
        "sources": {"sample_dirs": [], "audio_files": []},
    }
    with open(model_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)
    return model_dir


def test_list_models_empty(model_mgr):
    assert model_mgr.list_models() == []


def test_list_models(model_mgr, tmp_path):
    _create_fake_model(tmp_path, "drums-v1")
    _create_fake_model(tmp_path, "ambient-v2", config="v2_small")
    models = model_mgr.list_models()
    assert len(models) == 2
    names = [m.name for m in models]
    assert "drums-v1" in names
    assert "ambient-v2" in names


def test_get_model(model_mgr, tmp_path):
    _create_fake_model(tmp_path, "drums-v1", total_samples=47)
    info = model_mgr.get_model("drums-v1")
    assert info.name == "drums-v1"
    assert info.config == "v2"
    assert info.total_samples == 47


def test_get_model_path(model_mgr, tmp_path):
    _create_fake_model(tmp_path, "drums-v1")
    path = model_mgr.get_model_path("drums-v1")
    assert path.name == "model.ts"
    assert path.exists()


def test_delete_model(model_mgr, tmp_path):
    _create_fake_model(tmp_path, "drums-v1")
    model_mgr.delete_model("drums-v1")
    assert model_mgr.list_models() == []


def test_get_nonexistent_model_raises(model_mgr):
    with pytest.raises(KeyError):
        model_mgr.get_model("nope")
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_model_manager.py -v`
Expected: FAIL (module not found)

**Step 3: Write the implementation**

```python
"""Model manager - list and manage trained RAVE models."""

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class ModelInfo:
    """Metadata about a trained model."""

    name: str
    created: str
    config: str
    total_samples: int
    model_path: Path
    size_mb: float = 0.0
    pool_name: Optional[str] = None


class ModelManager:
    """Manages trained models on disk."""

    def __init__(self, models_dir: Path):
        self.models_dir = Path(models_dir)

    def list_models(self) -> list[ModelInfo]:
        """List all trained models."""
        models = []
        if not self.models_dir.exists():
            return models
        for model_dir in sorted(self.models_dir.iterdir()):
            if model_dir.is_dir() and (model_dir / "model.ts").exists():
                try:
                    models.append(self._load_model_info(model_dir))
                except Exception:
                    continue
        return models

    def get_model(self, name: str) -> ModelInfo:
        """Get model info by name."""
        model_dir = self.models_dir / name
        if not model_dir.exists():
            raise KeyError(f"Model '{name}' not found")
        return self._load_model_info(model_dir)

    def get_model_path(self, name: str) -> Path:
        """Get the path to a model's .ts file."""
        model_dir = self.models_dir / name
        ts_file = model_dir / "model.ts"
        if not ts_file.exists():
            raise KeyError(f"Model '{name}' not found or has no model.ts")
        return ts_file

    def delete_model(self, name: str) -> None:
        """Delete a model and all its files."""
        model_dir = self.models_dir / name
        if model_dir.exists():
            shutil.rmtree(model_dir)

    def _load_model_info(self, model_dir: Path) -> ModelInfo:
        metadata_path = model_dir / "metadata.json"
        ts_file = model_dir / "model.ts"

        name = model_dir.name
        created = ""
        config = "unknown"
        total_samples = 0
        pool_name = None

        if metadata_path.exists():
            with open(metadata_path) as f:
                data = json.load(f)
            created = data.get("created", "")
            config = data.get("config", "unknown")
            total_samples = data.get("total_samples", 0)
            pool_name = data.get("pool_name")

        size_mb = ts_file.stat().st_size / (1024 * 1024) if ts_file.exists() else 0

        return ModelInfo(
            name=name,
            created=created,
            config=config,
            total_samples=total_samples,
            model_path=ts_file,
            size_mb=size_mb,
            pool_name=pool_name,
        )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_model_manager.py -v`
Expected: All 6 PASS

**Step 5: Commit**

```bash
git add scropipe/model_manager.py tests/test_model_manager.py
git commit -m "feat: add model manager for trained model CRUD"
```

---

## Task 5: TUI App Shell with Tabs and Status Bar

**Files:**
- Create: `scropipe/tui/__init__.py`
- Create: `scropipe/tui/app.py`
- Create: `scropipe/tui/styles.tcss`
- Create: `tests/test_tui_app.py`

**Step 1: Write the failing test**

```python
"""Tests for TUI app shell."""

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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_tui_app.py -v`
Expected: FAIL (module not found)

**Step 3: Create `scropipe/tui/__init__.py`**

```python
"""Scropipe TUI - Textual-based terminal user interface."""
```

**Step 4: Create `scropipe/tui/styles.tcss`**

```css
Screen {
    layout: vertical;
}

#status-bar {
    dock: bottom;
    height: 1;
    background: $surface;
    color: $text-muted;
    padding: 0 1;
}

TabbedContent {
    height: 1fr;
}

.tab-content {
    padding: 1 2;
}

.section-title {
    margin-top: 1;
    text-style: bold;
}

.form-group {
    margin-bottom: 1;
}

.form-label {
    margin-bottom: 0;
}

.action-bar {
    margin-top: 1;
    layout: horizontal;
}

.action-bar Button {
    margin-right: 1;
}
```

**Step 5: Create `scropipe/tui/app.py`**

```python
"""Main Scropipe TUI application."""

from pathlib import Path
from typing import Optional

from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.widgets import Footer, Header, Label, Static, TabbedContent, TabPane


class ScropipeApp(App):
    """Scropipe TUI application."""

    TITLE = "scropipe"
    CSS_PATH = "styles.tcss"

    BINDINGS = [
        ("ctrl+q", "quit", "Quit"),
        ("1", "switch_tab('split')", "Split"),
        ("2", "switch_tab('pool')", "Pool"),
        ("3", "switch_tab('train')", "Train"),
        ("4", "switch_tab('generate')", "Generate"),
    ]

    def __init__(
        self,
        models_dir: Optional[Path] = None,
        pools_dir: Optional[Path] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.models_dir = models_dir
        self.pools_dir = pools_dir

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent():
            with TabPane("Split", id="split"):
                yield Label("Split tab - coming soon")
            with TabPane("Pool", id="pool"):
                yield Label("Pool tab - coming soon")
            with TabPane("Train", id="train"):
                yield Label("Train tab - coming soon")
            with TabPane("Generate", id="generate"):
                yield Label("Generate tab - coming soon")
        yield Static("Pool: none | Model: none | GPU: detecting...", id="status-bar")

    def action_switch_tab(self, tab_id: str) -> None:
        """Switch to a tab by ID."""
        tabbed = self.query_one(TabbedContent)
        tabbed.active = tab_id
```

**Step 6: Run test to verify it passes**

Run: `pytest tests/test_tui_app.py -v`
Expected: All 2 PASS

**Step 7: Commit**

```bash
git add scropipe/tui/ tests/test_tui_app.py
git commit -m "feat: add TUI app shell with tabs and status bar"
```

---

## Task 6: CLI Entry Point for TUI

**Files:**
- Modify: `scropipe/cli.py`

Wire `scropipe` (no args) to launch the TUI. The existing `main()` function uses Typer; we need it to launch the TUI when no subcommand is given.

**Step 1: Modify `cli.py` to add a TUI launch command**

Add a `tui` command and update the callback to launch TUI when no subcommand is given.

In `scropipe/cli.py`, modify the `main()` function at the bottom:

```python
def main():
    """Entry point for the CLI."""
    import sys
    # If no subcommand given, launch TUI
    if len(sys.argv) == 1:
        from .tui.app import ScropipeApp
        from .config import load_config
        config = load_config()
        app = ScropipeApp(
            models_dir=config.models_dir,
            pools_dir=config.pools_dir,
        )
        app.run()
    else:
        app()
```

**Step 2: Verify CLI still works**

Run: `scropipe --version`
Expected: `scropipe version 0.2.0`

Run: `scropipe --help`
Expected: Shows help with all subcommands

**Step 3: Verify TUI launches**

Run: `python -m scropipe.cli` (with no args)
Expected: TUI launches with four tabs. Press Ctrl+Q to exit.

**Step 4: Commit**

```bash
git add scropipe/cli.py
git commit -m "feat: launch TUI when scropipe is run with no args"
```

---

## Task 7: First-Run Setup Modal

**Files:**
- Modify: `scropipe/tui/app.py`
- Create: `tests/test_tui_setup.py`

**Step 1: Write the failing test**

```python
"""Tests for first-run setup modal."""

import pytest
from pathlib import Path
from scropipe.tui.app import ScropipeApp


@pytest.mark.asyncio
async def test_first_run_shows_setup_modal(tmp_path):
    app = ScropipeApp(models_dir=None, pools_dir=None)
    async with app.run_test() as pilot:
        # Should show setup modal when dirs are not configured
        modal = app.query("SetupModal")
        assert len(modal) == 1
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_tui_setup.py -v`
Expected: FAIL

**Step 3: Add SetupModal to `scropipe/tui/app.py`**

Add to `app.py`:

```python
from textual.screen import ModalScreen
from textual.widgets import Button, Input


class SetupModal(ModalScreen):
    """First-run setup modal for configuring storage directories."""

    DEFAULT_CSS = """
    SetupModal {
        align: center middle;
    }
    #setup-dialog {
        width: 60;
        height: auto;
        max-height: 20;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }
    #setup-dialog Label {
        margin-bottom: 1;
    }
    #setup-dialog Input {
        margin-bottom: 1;
    }
    #setup-dialog Button {
        margin-top: 1;
    }
    """

    def compose(self) -> ComposeResult:
        from textual.containers import Vertical
        with Vertical(id="setup-dialog"):
            yield Label("Welcome to scropipe")
            yield Label("Where should scropipe store its data?")
            yield Label("Models directory:", classes="form-label")
            yield Input(
                placeholder="e.g. /data/scropipe/models",
                id="models-dir-input",
            )
            yield Label("Pools directory:", classes="form-label")
            yield Input(
                placeholder="e.g. /data/scropipe/pools",
                id="pools-dir-input",
            )
            yield Label("(You can change these later in Settings)", classes="form-label")
            yield Button("Continue", variant="primary", id="setup-continue")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "setup-continue":
            models_input = self.query_one("#models-dir-input", Input)
            pools_input = self.query_one("#pools-dir-input", Input)
            models_dir = models_input.value.strip()
            pools_dir = pools_input.value.strip()
            if models_dir and pools_dir:
                self.dismiss((Path(models_dir), Path(pools_dir)))
```

Update `ScropipeApp.on_mount()`:

```python
    def on_mount(self) -> None:
        if self.models_dir is None or self.pools_dir is None:
            self.push_screen(SetupModal(), self._on_setup_complete)

    def _on_setup_complete(self, result: tuple[Path, Path]) -> None:
        self.models_dir, self.pools_dir = result
        # Create directories
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.pools_dir.mkdir(parents=True, exist_ok=True)
        # Save config
        from ..config import ScropipeConfig, save_config
        config = ScropipeConfig(models_dir=self.models_dir, pools_dir=self.pools_dir)
        save_config(config)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_tui_setup.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add scropipe/tui/app.py tests/test_tui_setup.py
git commit -m "feat: add first-run setup modal for storage directories"
```

---

## Task 8: Split Tab

**Files:**
- Create: `scropipe/tui/split_tab.py`
- Modify: `scropipe/tui/app.py`
- Create: `tests/test_tui_split.py`

**Step 1: Write the failing test**

```python
"""Tests for the Split tab."""

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
        tabbed.active = "split"
        await pilot.pause()
        radio_set = app.query("RadioSet")
        assert len(radio_set) >= 1


@pytest.mark.asyncio
async def test_split_tab_has_source_input(app):
    async with app.run_test() as pilot:
        tabbed = app.query_one("TabbedContent")
        tabbed.active = "split"
        await pilot.pause()
        source_input = app.query_one("#split-source-input")
        assert source_input is not None


@pytest.mark.asyncio
async def test_split_tab_has_action_buttons(app):
    async with app.run_test() as pilot:
        tabbed = app.query_one("TabbedContent")
        tabbed.active = "split"
        await pilot.pause()
        split_btn = app.query_one("#split-btn")
        assert split_btn is not None
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_tui_split.py -v`
Expected: FAIL

**Step 3: Create `scropipe/tui/split_tab.py`**

```python
"""Split tab - audio file splitting interface."""

from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import (
    Button,
    Input,
    Label,
    RadioButton,
    RadioSet,
    Static,
)
from textual.widget import Widget
from textual.worker import Worker


class TransientSettings(Static):
    """Settings for transient detection mode."""

    def compose(self) -> ComposeResult:
        yield Label("Sensitivity (delta):", classes="form-label")
        yield Input(value="0.07", id="split-delta", type="number")
        yield Label("Min length (s):", classes="form-label")
        yield Input(value="0.05", id="split-min-length", type="number")
        yield Label("Max length (s):", classes="form-label")
        yield Input(value="10.0", id="split-max-length", type="number")


class GridSettings(Static):
    """Settings for grid (equal chunk) mode."""

    def compose(self) -> ComposeResult:
        yield Label("Chunk length (s):", classes="form-label")
        yield Input(value="2.0", id="split-chunk-length", type="number")
        yield Label("— or use BPM —", classes="form-label")
        yield Label("BPM:", classes="form-label")
        yield Input(placeholder="e.g. 120", id="split-bpm", type="number")
        yield Label("Bars:", classes="form-label")
        yield Input(value="4", id="split-bars", type="number")


class TextureSettings(Static):
    """Settings for texture gating mode."""

    def compose(self) -> ComposeResult:
        yield Label("Min duration (s):", classes="form-label")
        yield Input(value="1.0", id="split-min-duration", type="number")
        yield Label("Max duration (s):", classes="form-label")
        yield Input(value="30.0", id="split-max-duration", type="number")
        yield Label("RMS threshold:", classes="form-label")
        yield Input(value="0.1", id="split-rms-threshold", type="number")
        yield Label("Stability threshold:", classes="form-label")
        yield Input(value="0.15", id="split-stability-threshold", type="number")


class SplitTab(Static):
    """Split tab content."""

    def compose(self) -> ComposeResult:
        with Vertical(classes="tab-content"):
            yield Label("Source File", classes="section-title")
            with Horizontal(classes="form-group"):
                yield Input(
                    placeholder="Path to audio file...",
                    id="split-source-input",
                )
                yield Button("Browse", id="split-browse-btn")

            yield Label("Splitting Mode", classes="section-title")
            with RadioSet(id="split-mode-selector"):
                yield RadioButton("Transient", value=True, id="mode-transient")
                yield RadioButton("Grid", id="mode-grid")
                yield RadioButton("Texture", id="mode-texture")

            yield TransientSettings(id="transient-settings")
            yield GridSettings(id="grid-settings")
            yield TextureSettings(id="texture-settings")

            yield Label("Output Directory", classes="section-title")
            with Horizontal(classes="form-group"):
                yield Input(
                    placeholder="Output directory (auto-generated if empty)",
                    id="split-output-input",
                )
                yield Button("Browse", id="split-output-browse-btn")

            yield Static("", id="split-status")

            with Horizontal(classes="action-bar"):
                yield Button("Split", variant="primary", id="split-btn")
                yield Button("Split & Add to Pool", variant="success", id="split-and-pool-btn")

    def on_mount(self) -> None:
        # Show only transient settings by default
        self.query_one("#grid-settings").display = False
        self.query_one("#texture-settings").display = False

    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        """Switch visible settings based on selected mode."""
        self.query_one("#transient-settings").display = event.index == 0
        self.query_one("#grid-settings").display = event.index == 1
        self.query_one("#texture-settings").display = event.index == 2

    def _get_selected_mode(self) -> str:
        radio_set = self.query_one("#split-mode-selector", RadioSet)
        if radio_set.pressed_index == 1:
            return "grid"
        elif radio_set.pressed_index == 2:
            return "texture"
        return "transient"

    def _get_split_kwargs(self) -> dict:
        """Collect split parameters from the form."""
        mode = self._get_selected_mode()
        source = self.query_one("#split-source-input", Input).value.strip()
        output = self.query_one("#split-output-input", Input).value.strip()

        kwargs = {"input_file": Path(source), "mode": mode}

        if mode == "transient":
            kwargs["delta"] = float(self.query_one("#split-delta", Input).value or "0.07")
            kwargs["min_length"] = float(self.query_one("#split-min-length", Input).value or "0.05")
            kwargs["max_length"] = float(self.query_one("#split-max-length", Input).value or "10.0")
        elif mode == "grid":
            chunk = self.query_one("#split-chunk-length", Input).value
            bpm = self.query_one("#split-bpm", Input).value
            if bpm:
                kwargs["bpm"] = float(bpm)
                kwargs["bars"] = int(self.query_one("#split-bars", Input).value or "4")
            elif chunk:
                kwargs["chunk_length"] = float(chunk)
        elif mode == "texture":
            kwargs["min_duration"] = float(
                self.query_one("#split-min-duration", Input).value or "1.0"
            )
            kwargs["max_duration"] = float(
                self.query_one("#split-max-duration", Input).value or "30.0"
            )
            kwargs["rms_threshold"] = float(
                self.query_one("#split-rms-threshold", Input).value or "0.1"
            )
            kwargs["stability_threshold"] = float(
                self.query_one("#split-stability-threshold", Input).value or "0.15"
            )

        return kwargs

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "split-btn":
            self._run_split(add_to_pool=False)
        elif event.button.id == "split-and-pool-btn":
            self._run_split(add_to_pool=True)

    def _run_split(self, add_to_pool: bool) -> None:
        """Run the split operation in a worker thread."""
        source = self.query_one("#split-source-input", Input).value.strip()
        if not source or not Path(source).exists():
            self.query_one("#split-status").update("[red]Source file not found[/red]")
            return

        output_path = self.query_one("#split-output-input", Input).value.strip()
        if not output_path:
            output_path = str(Path(source).parent / f"{Path(source).stem}_split")

        status = self.query_one("#split-status")
        status.update("[yellow]Splitting...[/yellow]")

        self.run_worker(
            self._do_split(output_path, add_to_pool),
            thread=True,
        )

    async def _do_split(self, output_path: str, add_to_pool: bool) -> None:
        """Perform the split in a background thread."""
        from ..stages import SplitStage

        kwargs = self._get_split_kwargs()
        stage = SplitStage(Path(output_path))
        result = stage.run(**kwargs)

        status = self.query_one("#split-status")
        if result.success:
            count = result.details.get("sample_count", 0)
            status.update(f"[green]Split complete: {count} samples[/green]")
            if add_to_pool and result.output_dir:
                self.app.post_message(
                    self.SplitComplete(result.output_dir, add_to_pool=True)
                )
        else:
            status.update(f"[red]Split failed: {result.message}[/red]")

    class SplitComplete:
        """Message posted when a split operation completes."""
        def __init__(self, output_dir: Path, add_to_pool: bool = False):
            self.output_dir = output_dir
            self.add_to_pool = add_to_pool
```

**Step 4: Update `scropipe/tui/app.py` to use SplitTab**

Replace the Split TabPane placeholder in `compose()`:

```python
from .split_tab import SplitTab

# In compose():
with TabPane("Split", id="split"):
    yield SplitTab()
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/test_tui_split.py -v`
Expected: All 3 PASS

**Step 6: Commit**

```bash
git add scropipe/tui/split_tab.py scropipe/tui/app.py tests/test_tui_split.py
git commit -m "feat: add Split tab with mode selector and parameter forms"
```

---

## Task 9: Pool Tab

**Files:**
- Create: `scropipe/tui/pool_tab.py`
- Modify: `scropipe/tui/app.py`
- Create: `tests/test_tui_pool.py`

**Step 1: Write the failing test**

```python
"""Tests for the Pool tab."""

import pytest
from scropipe.tui.app import ScropipeApp


@pytest.fixture
def app(tmp_path):
    (tmp_path / "pools").mkdir()
    (tmp_path / "models").mkdir()
    return ScropipeApp(
        models_dir=tmp_path / "models",
        pools_dir=tmp_path / "pools",
    )


@pytest.mark.asyncio
async def test_pool_tab_has_pool_list(app):
    async with app.run_test() as pilot:
        tabbed = app.query_one("TabbedContent")
        tabbed.active = "pool"
        await pilot.pause()
        pool_list = app.query_one("#pool-list")
        assert pool_list is not None


@pytest.mark.asyncio
async def test_pool_tab_has_new_pool_button(app):
    async with app.run_test() as pilot:
        tabbed = app.query_one("TabbedContent")
        tabbed.active = "pool"
        await pilot.pause()
        btn = app.query_one("#new-pool-btn")
        assert btn is not None
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_tui_pool.py -v`
Expected: FAIL

**Step 3: Create `scropipe/tui/pool_tab.py`**

```python
"""Pool tab - sample pool management interface."""

from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import (
    Button,
    Input,
    Label,
    ListItem,
    ListView,
    Static,
)

from ..pool_manager import PoolManager


class PoolDetailPanel(Static):
    """Right panel showing pool details."""

    def compose(self) -> ComposeResult:
        yield Label("Select a pool", id="pool-detail-title", classes="section-title")
        yield Static("", id="pool-detail-info")
        yield Static("", id="pool-sources-list")
        with Horizontal(classes="action-bar"):
            yield Button("+ Add Files", id="add-files-btn")
            yield Button("+ Add Directory", id="add-dir-btn")
        with Horizontal(classes="action-bar"):
            yield Button("Delete Pool", variant="error", id="delete-pool-btn")
            yield Button("Train", variant="success", id="pool-train-btn")


class PoolTab(Static):
    """Pool tab content."""

    def compose(self) -> ComposeResult:
        with Horizontal(classes="tab-content"):
            with Vertical(id="pool-sidebar"):
                yield Label("Pools", classes="section-title")
                yield ListView(id="pool-list")
                yield Button("+ New Pool", variant="primary", id="new-pool-btn")
            yield PoolDetailPanel(id="pool-detail")

    def on_mount(self) -> None:
        self._refresh_pool_list()

    def _get_pool_manager(self) -> PoolManager:
        pools_dir = self.app.pools_dir
        if pools_dir is None:
            pools_dir = Path.home() / ".local" / "share" / "scropipe" / "pools"
        return PoolManager(pools_dir)

    def _refresh_pool_list(self) -> None:
        pool_list = self.query_one("#pool-list", ListView)
        pool_list.clear()
        try:
            mgr = self._get_pool_manager()
            for pool in mgr.list_pools():
                pool_list.append(
                    ListItem(Label(f"{pool.name} ({pool.sample_count})"), name=pool.name)
                )
        except Exception:
            pass

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Show pool details when a pool is selected."""
        pool_name = event.item.name
        if not pool_name:
            return
        try:
            mgr = self._get_pool_manager()
            pool = mgr.get_pool(pool_name)
        except KeyError:
            return

        title = self.query_one("#pool-detail-title", Label)
        title.update(f"Pool: {pool.name}")

        info = self.query_one("#pool-detail-info", Static)
        info.update(f"{pool.sample_count} samples")

        sources_list = self.query_one("#pool-sources-list", Static)
        if pool.sources:
            lines = ["Sources:"]
            for src in pool.sources:
                lines.append(f"  {src.path} ({src.count} samples)")
            sources_list.update("\n".join(lines))
        else:
            sources_list.update("No sources yet")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "new-pool-btn":
            self.app.push_screen(NewPoolModal(), self._on_pool_created)
        elif event.button.id == "delete-pool-btn":
            self._delete_selected_pool()
        elif event.button.id == "add-files-btn":
            self._add_files_to_selected_pool()
        elif event.button.id == "add-dir-btn":
            self._add_directory_to_selected_pool()
        elif event.button.id == "pool-train-btn":
            self._go_to_train()

    def _on_pool_created(self, name: str | None) -> None:
        if name:
            try:
                mgr = self._get_pool_manager()
                mgr.create_pool(name)
                self._refresh_pool_list()
            except ValueError as e:
                self.notify(str(e), severity="error")

    def _get_selected_pool_name(self) -> str | None:
        pool_list = self.query_one("#pool-list", ListView)
        if pool_list.highlighted_child is not None:
            return pool_list.highlighted_child.name
        return None

    def _delete_selected_pool(self) -> None:
        name = self._get_selected_pool_name()
        if name:
            mgr = self._get_pool_manager()
            mgr.delete_pool(name)
            self._refresh_pool_list()

    def _add_files_to_selected_pool(self) -> None:
        name = self._get_selected_pool_name()
        if not name:
            self.notify("Select a pool first", severity="warning")
            return
        self.app.push_screen(
            FileInputModal(title="Add Files", placeholder="Paths to WAV files (one per line)"),
            lambda paths: self._do_add_files(name, paths),
        )

    def _do_add_files(self, pool_name: str, paths_text: str | None) -> None:
        if not paths_text:
            return
        files = [Path(p.strip()) for p in paths_text.strip().split("\n") if p.strip()]
        mgr = self._get_pool_manager()
        count = mgr.add_files(pool_name, files)
        self.notify(f"Added {count} files")
        self._refresh_pool_list()

    def _add_directory_to_selected_pool(self) -> None:
        name = self._get_selected_pool_name()
        if not name:
            self.notify("Select a pool first", severity="warning")
            return
        self.app.push_screen(
            FileInputModal(title="Add Directory", placeholder="Path to directory"),
            lambda path: self._do_add_dir(name, path),
        )

    def _do_add_dir(self, pool_name: str, path_text: str | None) -> None:
        if not path_text:
            return
        directory = Path(path_text.strip())
        mgr = self._get_pool_manager()
        count = mgr.add_directory(pool_name, directory)
        self.notify(f"Added {count} files from directory")
        self._refresh_pool_list()

    def _go_to_train(self) -> None:
        name = self._get_selected_pool_name()
        if name:
            # Switch to train tab with pool pre-selected
            tabbed = self.app.query_one("TabbedContent")
            tabbed.active = "train"


class NewPoolModal(Static):
    """Modal for creating a new pool."""

    # This will be replaced with a proper ModalScreen implementation
    # during integration. For now it's a simple input dialog.

    pass


class FileInputModal(Static):
    """Modal for entering file/directory paths."""

    def __init__(self, title: str = "", placeholder: str = "", **kwargs):
        super().__init__(**kwargs)
        self._title = title
        self._placeholder = placeholder
```

Note: The modals (`NewPoolModal`, `FileInputModal`) above are stubs. They should be implemented as proper `ModalScreen` subclasses. Here's the full implementation:

```python
from textual.screen import ModalScreen


class NewPoolModal(ModalScreen[str | None]):
    """Modal for creating a new pool."""

    DEFAULT_CSS = """
    NewPoolModal {
        align: center middle;
    }
    #new-pool-dialog {
        width: 50;
        height: auto;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical(id="new-pool-dialog"):
            yield Label("New Pool")
            yield Label("Name:", classes="form-label")
            yield Input(placeholder="e.g. drum-hits", id="new-pool-name")
            with Horizontal(classes="action-bar"):
                yield Button("Create", variant="primary", id="create-pool-confirm")
                yield Button("Cancel", id="create-pool-cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "create-pool-confirm":
            name = self.query_one("#new-pool-name", Input).value.strip()
            self.dismiss(name if name else None)
        elif event.button.id == "create-pool-cancel":
            self.dismiss(None)


class FileInputModal(ModalScreen[str | None]):
    """Modal for entering file/directory paths."""

    DEFAULT_CSS = """
    FileInputModal {
        align: center middle;
    }
    #file-input-dialog {
        width: 60;
        height: auto;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }
    """

    def __init__(self, title: str = "Input", placeholder: str = "", **kwargs):
        super().__init__(**kwargs)
        self._modal_title = title
        self._placeholder = placeholder

    def compose(self) -> ComposeResult:
        with Vertical(id="file-input-dialog"):
            yield Label(self._modal_title)
            yield Input(placeholder=self._placeholder, id="file-path-input")
            with Horizontal(classes="action-bar"):
                yield Button("OK", variant="primary", id="file-input-ok")
                yield Button("Cancel", id="file-input-cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "file-input-ok":
            value = self.query_one("#file-path-input", Input).value.strip()
            self.dismiss(value if value else None)
        elif event.button.id == "file-input-cancel":
            self.dismiss(None)
```

**Step 4: Update `scropipe/tui/app.py` to use PoolTab**

Replace Pool TabPane placeholder:

```python
from .pool_tab import PoolTab

# In compose():
with TabPane("Pool", id="pool"):
    yield PoolTab()
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/test_tui_pool.py -v`
Expected: All 2 PASS

**Step 6: Commit**

```bash
git add scropipe/tui/pool_tab.py scropipe/tui/app.py tests/test_tui_pool.py
git commit -m "feat: add Pool tab with pool list, detail panel, and CRUD modals"
```

---

## Task 10: Train Tab (Configuration Mode)

**Files:**
- Create: `scropipe/tui/train_tab.py`
- Modify: `scropipe/tui/app.py`
- Create: `tests/test_tui_train.py`

**Step 1: Write the failing test**

```python
"""Tests for the Train tab."""

import pytest
from scropipe.tui.app import ScropipeApp


@pytest.fixture
def app(tmp_path):
    (tmp_path / "pools").mkdir()
    (tmp_path / "models").mkdir()
    return ScropipeApp(
        models_dir=tmp_path / "models",
        pools_dir=tmp_path / "pools",
    )


@pytest.mark.asyncio
async def test_train_tab_has_pool_selector(app):
    async with app.run_test() as pilot:
        tabbed = app.query_one("TabbedContent")
        tabbed.active = "train"
        await pilot.pause()
        pool_select = app.query_one("#train-pool-select")
        assert pool_select is not None


@pytest.mark.asyncio
async def test_train_tab_has_model_name_input(app):
    async with app.run_test() as pilot:
        tabbed = app.query_one("TabbedContent")
        tabbed.active = "train"
        await pilot.pause()
        name_input = app.query_one("#train-model-name")
        assert name_input is not None


@pytest.mark.asyncio
async def test_train_tab_has_start_button(app):
    async with app.run_test() as pilot:
        tabbed = app.query_one("TabbedContent")
        tabbed.active = "train"
        await pilot.pause()
        btn = app.query_one("#start-training-btn")
        assert btn is not None
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_tui_train.py -v`
Expected: FAIL

**Step 3: Create `scropipe/tui/train_tab.py`**

```python
"""Train tab - RAVE model training interface."""

import asyncio
import re
import subprocess
import time
from pathlib import Path
from typing import Optional

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import (
    Button,
    Input,
    Label,
    RadioButton,
    RadioSet,
    Select,
    Sparkline,
    Static,
)

from ..pool_manager import PoolManager
from ..model_manager import ModelManager


class TrainConfigPanel(Static):
    """Training configuration panel (shown before training starts)."""

    def compose(self) -> ComposeResult:
        with Vertical(classes="tab-content"):
            yield Label("Pool:", classes="form-label")
            yield Select([], id="train-pool-select", prompt="Select a pool...")

            yield Label("Model name:", classes="form-label")
            yield Input(placeholder="e.g. drums-v1", id="train-model-name")

            yield Label("Stop Conditions", classes="section-title")
            with RadioSet(id="stop-condition"):
                yield RadioButton("Manual (stop when ready)", value=True, id="stop-manual")
                yield RadioButton("Max steps:", id="stop-max-steps")
                yield RadioButton("Delta target:", id="stop-delta")
            yield Input(value="10000", id="train-max-steps", type="number")
            yield Input(value="0.001", id="train-delta-target", type="number")

            yield Label("RAVE Config", classes="section-title")
            yield Label("Architecture:", classes="form-label")
            yield Select(
                [("v2", "v2"), ("v2_small", "v2_small"), ("discrete", "discrete")],
                value="v2",
                id="train-arch-select",
            )
            yield Label("Checkpoint every:", classes="form-label")
            with Horizontal():
                yield Input(value="500", id="train-val-every", type="number")
                yield Label(" steps")

            yield Static("", id="train-gpu-info")

            with Horizontal(classes="action-bar"):
                yield Button(
                    "Start Training", variant="primary", id="start-training-btn"
                )

    def on_mount(self) -> None:
        self._refresh_pool_list()
        self._detect_gpu()
        # Initially hide conditional inputs
        self.query_one("#train-max-steps").display = False
        self.query_one("#train-delta-target").display = False

    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        if event.radio_set.id == "stop-condition":
            self.query_one("#train-max-steps").display = event.index == 1
            self.query_one("#train-delta-target").display = event.index == 2

    def _refresh_pool_list(self) -> None:
        pools_dir = self.app.pools_dir
        if pools_dir is None:
            return
        mgr = PoolManager(pools_dir)
        pools = mgr.list_pools()
        select = self.query_one("#train-pool-select", Select)
        select.set_options(
            [(f"{p.name} ({p.sample_count} samples)", p.name) for p in pools]
        )

    def _detect_gpu(self) -> None:
        gpu_info = self.query_one("#train-gpu-info", Static)
        try:
            import torch

            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_info.update(f"GPU: {gpu_name}")
            else:
                gpu_info.update("GPU: None (will use CPU)")
        except ImportError:
            gpu_info.update("GPU: torch not installed")


class TrainDashboard(Static):
    """Live training dashboard (shown during training)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loss_history: list[float] = []
        self.current_step = 0
        self.max_steps: Optional[int] = None
        self.start_time: Optional[float] = None

    def compose(self) -> ComposeResult:
        with Vertical(classes="tab-content"):
            yield Label("", id="dash-title", classes="section-title")
            yield Static("", id="dash-info")
            yield Static("", id="dash-metrics")

            yield Label("Loss:", classes="section-title")
            yield Sparkline([], id="dash-sparkline")

            yield Static("", id="dash-timing")
            yield Static("", id="dash-checkpoint")

            with Horizontal(classes="action-bar"):
                yield Button("Stop & Save", variant="warning", id="stop-save-btn")
                yield Button("Stop & Discard", variant="error", id="stop-discard-btn")

    def update_metrics(self, step: int, loss: float, delta: float = 0.0) -> None:
        """Update the dashboard with new training metrics."""
        self.current_step = step
        self.loss_history.append(loss)

        step_str = f"{step}"
        if self.max_steps:
            step_str = f"{step} / {self.max_steps}"
        else:
            step_str = f"{step} / manual"

        metrics = self.query_one("#dash-metrics", Static)
        metrics.update(f"Step: {step_str}    Loss: {loss:.4f}    Δ: {delta:.4f}")

        sparkline = self.query_one("#dash-sparkline", Sparkline)
        sparkline.data = self.loss_history

        if self.start_time:
            elapsed = time.time() - self.start_time
            elapsed_str = self._format_duration(elapsed)
            timing = self.query_one("#dash-timing", Static)
            if self.max_steps and step > 0:
                eta = (elapsed / step) * (self.max_steps - step)
                timing.update(f"Elapsed: {elapsed_str}    ETA: {self._format_duration(eta)}")
            else:
                timing.update(f"Elapsed: {elapsed_str}")

    @staticmethod
    def _format_duration(seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        if h > 0:
            return f"{h}h {m}m"
        return f"{m}m"


class TrainTab(Static):
    """Train tab - switches between config and dashboard views."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._training_process: Optional[subprocess.Popen] = None

    def compose(self) -> ComposeResult:
        yield TrainConfigPanel(id="train-config")
        yield TrainDashboard(id="train-dashboard")

    def on_mount(self) -> None:
        self.query_one("#train-dashboard").display = False

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "start-training-btn":
            self._start_training()
        elif event.button.id == "stop-save-btn":
            self._stop_training(save=True)
        elif event.button.id == "stop-discard-btn":
            self._stop_training(save=False)

    def _start_training(self) -> None:
        """Validate inputs and start training."""
        config = self.query_one("#train-config", TrainConfigPanel)
        pool_select = config.query_one("#train-pool-select", Select)
        name_input = config.query_one("#train-model-name", Input)

        pool_name = pool_select.value
        model_name = name_input.value.strip()

        if pool_name is Select.BLANK:
            self.notify("Select a pool first", severity="warning")
            return
        if not model_name:
            self.notify("Enter a model name", severity="warning")
            return

        # Switch to dashboard view
        self.query_one("#train-config").display = False
        dashboard = self.query_one("#train-dashboard", TrainDashboard)
        dashboard.display = True

        title = dashboard.query_one("#dash-title", Label)
        title.update(f"Training: {model_name}")

        arch = config.query_one("#train-arch-select", Select).value
        info = dashboard.query_one("#dash-info", Static)
        info.update(f"Pool: {pool_name}    Arch: RAVE {arch}")

        dashboard.start_time = time.time()

        # Determine stop condition
        stop_radio = config.query_one("#stop-condition", RadioSet)
        max_steps = None
        if stop_radio.pressed_index == 1:
            max_steps = int(config.query_one("#train-max-steps", Input).value or "10000")
            dashboard.max_steps = max_steps

        val_every = int(config.query_one("#train-val-every", Input).value or "500")

        # Start training in background worker
        self.run_worker(
            self._do_train(
                pool_name=str(pool_name),
                model_name=model_name,
                arch=str(arch),
                max_steps=max_steps,
                val_every=val_every,
            ),
            thread=True,
        )

    async def _do_train(
        self,
        pool_name: str,
        model_name: str,
        arch: str,
        max_steps: Optional[int],
        val_every: int,
    ) -> None:
        """Run RAVE training in a background thread."""
        import tempfile

        pools_dir = self.app.pools_dir
        models_dir = self.app.models_dir
        if not pools_dir or not models_dir:
            return

        mgr = PoolManager(pools_dir)
        samples_dir = mgr.get_samples_dir(pool_name)

        from ..stages import RavePreprocessStage, RaveTrainStage, RaveExportStage

        with tempfile.TemporaryDirectory(prefix="scropipe_train_") as temp_dir:
            temp_path = Path(temp_dir)

            # Preprocess
            dashboard = self.query_one("#train-dashboard", TrainDashboard)
            checkpoint_info = dashboard.query_one("#dash-checkpoint", Static)
            checkpoint_info.update("Preprocessing...")

            preprocess = RavePreprocessStage(temp_path)
            result = preprocess.run(input_dir=samples_dir)
            if not result.success:
                checkpoint_info.update(f"[red]Preprocess failed: {result.message}[/red]")
                return

            # Train
            checkpoint_info.update("Training started...")
            train_stage = RaveTrainStage(temp_path)

            # We need to run training with line-by-line output parsing
            # to update the dashboard. The RaveTrainStage runs rave as
            # a subprocess, so we replicate that here with output parsing.
            from ..utils.discovery import find_tool
            rave_cmd = str(find_tool("rave"))

            cmd = [
                rave_cmd, "train",
                "--config", arch,
                "--db_path", str(result.output_dir),
                "--name", "model",
                "--n_signal", "131072",
                "--workers", "0",
                "--val_every", str(val_every),
            ]

            # GPU detection
            try:
                import torch
                if torch.cuda.is_available():
                    cmd.extend(["--gpu", "0"])
            except ImportError:
                pass

            if max_steps:
                cmd.extend(["--max_steps", str(max_steps)])

            # Run training process
            self._training_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(train_stage.output_dir),
            )
            train_stage.ensure_output_dir()

            prev_loss = None
            for line in self._training_process.stdout:
                # Parse RAVE training output for step/loss
                # Typical: "Step 1000 | loss: 0.1234"
                step_match = re.search(r"(?:step|Step)\s*(\d+)", line, re.IGNORECASE)
                loss_match = re.search(r"loss[:\s]+([0-9.]+)", line, re.IGNORECASE)

                if step_match and loss_match:
                    step = int(step_match.group(1))
                    loss = float(loss_match.group(1))
                    delta = abs(loss - prev_loss) if prev_loss is not None else 0.0
                    prev_loss = loss
                    self.app.call_from_thread(
                        dashboard.update_metrics, step, loss, delta
                    )

                if "checkpoint" in line.lower() or "saving" in line.lower():
                    self.app.call_from_thread(
                        checkpoint_info.update,
                        f"Checkpoint: step {step_match.group(1) if step_match else '?'}"
                    )

            self._training_process.wait()

            if self._training_process.returncode == 0:
                # Export and save model
                self._save_model(
                    train_stage.output_dir, models_dir, model_name, arch,
                    pool_name, samples_dir, max_steps, val_every,
                )
                self.app.call_from_thread(
                    checkpoint_info.update,
                    f"[green]Training complete! Model saved as '{model_name}'[/green]"
                )

    def _save_model(
        self, run_dir, models_dir, model_name, arch,
        pool_name, samples_dir, max_steps, val_every,
    ):
        """Export and save the trained model."""
        import json
        import shutil
        from datetime import datetime
        from ..stages import RaveExportStage

        # Export
        export = RaveExportStage(run_dir.parent)
        export.run(run_dir=run_dir)

        ts_files = list(run_dir.glob("**/*.ts"))
        if not ts_files:
            return

        # Copy to models directory
        model_output = models_dir / model_name
        model_output.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ts_files[0], model_output / "model.ts")

        # Save metadata
        sample_count = len(list(samples_dir.glob("*.wav")))
        metadata = {
            "name": model_name,
            "created": datetime.now().isoformat(),
            "config": arch,
            "total_samples": sample_count,
            "pool_name": pool_name,
            "epochs": max_steps,
            "val_every": val_every,
        }
        with open(model_output / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    def _stop_training(self, save: bool) -> None:
        """Stop the training process."""
        if self._training_process and self._training_process.poll() is None:
            self._training_process.terminate()
            self._training_process.wait(timeout=10)

        if not save:
            # Switch back to config view
            self.query_one("#train-config").display = True
            self.query_one("#train-dashboard").display = False
        else:
            # Save from latest checkpoint
            dashboard = self.query_one("#train-dashboard", TrainDashboard)
            checkpoint = dashboard.query_one("#dash-checkpoint", Static)
            checkpoint.update("[yellow]Saving from last checkpoint...[/yellow]")
```

**Step 4: Update `scropipe/tui/app.py` to use TrainTab**

```python
from .train_tab import TrainTab

# In compose():
with TabPane("Train", id="train"):
    yield TrainTab()
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/test_tui_train.py -v`
Expected: All 3 PASS

**Step 6: Commit**

```bash
git add scropipe/tui/train_tab.py scropipe/tui/app.py tests/test_tui_train.py
git commit -m "feat: add Train tab with config panel and live dashboard"
```

---

## Task 11: Generate Tab

**Files:**
- Create: `scropipe/tui/generate_tab.py`
- Modify: `scropipe/tui/app.py`
- Create: `tests/test_tui_generate.py`

**Step 1: Write the failing test**

```python
"""Tests for the Generate tab."""

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
    # Create a fake model
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
        tabbed.active = "generate"
        await pilot.pause()
        select = app.query_one("#gen-model-select")
        assert select is not None


@pytest.mark.asyncio
async def test_generate_tab_has_input_output_fields(app):
    async with app.run_test() as pilot:
        tabbed = app.query_one("TabbedContent")
        tabbed.active = "generate"
        await pilot.pause()
        input_field = app.query_one("#gen-input-dir")
        output_field = app.query_one("#gen-output-dir")
        assert input_field is not None
        assert output_field is not None


@pytest.mark.asyncio
async def test_generate_tab_has_generate_button(app):
    async with app.run_test() as pilot:
        tabbed = app.query_one("TabbedContent")
        tabbed.active = "generate"
        await pilot.pause()
        btn = app.query_one("#generate-btn")
        assert btn is not None
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_tui_generate.py -v`
Expected: FAIL

**Step 3: Create `scropipe/tui/generate_tab.py`**

```python
"""Generate tab - audio generation from trained models."""

from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import (
    Button,
    DataTable,
    Input,
    Label,
    ProgressBar,
    Select,
    Static,
)

from ..model_manager import ModelManager


class GenerateTab(Static):
    """Generate tab content."""

    def compose(self) -> ComposeResult:
        with Vertical(classes="tab-content"):
            yield Label("Model", classes="section-title")
            yield Select([], id="gen-model-select", prompt="Select a model...")

            yield Label("Input", classes="section-title")
            with Horizontal(classes="form-group"):
                yield Input(
                    placeholder="Path to source samples directory...",
                    id="gen-input-dir",
                )
                yield Button("Browse", id="gen-input-browse")
            yield Static("", id="gen-input-info")

            yield Label("Output", classes="section-title")
            with Horizontal(classes="form-group"):
                yield Input(
                    placeholder="Path to output directory...",
                    id="gen-output-dir",
                )
                yield Button("Browse", id="gen-output-browse")

            yield Label("Models Library", classes="section-title")
            yield DataTable(id="gen-models-table")

            yield Static("", id="gen-status")
            yield ProgressBar(id="gen-progress", show_eta=True, show_percentage=True)

            with Horizontal(classes="action-bar"):
                yield Button("Generate", variant="primary", id="generate-btn")
                yield Button("Delete Model", variant="error", id="gen-delete-model-btn")

    def on_mount(self) -> None:
        self.query_one("#gen-progress", ProgressBar).display = False
        self._setup_table()
        self._refresh_models()

    def _get_model_manager(self) -> ModelManager:
        models_dir = self.app.models_dir
        if models_dir is None:
            models_dir = Path.home() / ".local" / "share" / "scropipe" / "models"
        return ModelManager(models_dir)

    def _setup_table(self) -> None:
        table = self.query_one("#gen-models-table", DataTable)
        table.add_columns("Name", "Arch", "Pool", "Samples", "Size")
        table.cursor_type = "row"

    def _refresh_models(self) -> None:
        mgr = self._get_model_manager()
        models = mgr.list_models()

        # Update select
        select = self.query_one("#gen-model-select", Select)
        select.set_options(
            [(f"{m.name} (RAVE {m.config})", m.name) for m in models]
        )

        # Update table
        table = self.query_one("#gen-models-table", DataTable)
        table.clear()
        for m in models:
            table.add_row(
                m.name,
                m.config,
                m.pool_name or "-",
                str(m.total_samples),
                f"{m.size_mb:.1f} MB",
                key=m.name,
            )

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """When a model is selected in the table, select it in the dropdown too."""
        if event.row_key:
            select = self.query_one("#gen-model-select", Select)
            select.value = event.row_key.value

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "gen-input-dir":
            path = Path(event.value.strip())
            info = self.query_one("#gen-input-info", Static)
            if path.exists() and path.is_dir():
                wav_count = len(list(path.glob("*.wav")))
                info.update(f"Found: {wav_count} files")
            else:
                info.update("")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "generate-btn":
            self._start_generation()
        elif event.button.id == "gen-delete-model-btn":
            self._delete_selected_model()

    def _start_generation(self) -> None:
        select = self.query_one("#gen-model-select", Select)
        input_dir = self.query_one("#gen-input-dir", Input).value.strip()
        output_dir = self.query_one("#gen-output-dir", Input).value.strip()

        if select.value is Select.BLANK:
            self.notify("Select a model first", severity="warning")
            return
        if not input_dir or not Path(input_dir).exists():
            self.notify("Input directory not found", severity="warning")
            return
        if not output_dir:
            self.notify("Specify an output directory", severity="warning")
            return

        status = self.query_one("#gen-status", Static)
        status.update("[yellow]Generating...[/yellow]")
        progress = self.query_one("#gen-progress", ProgressBar)
        progress.display = True

        self.run_worker(
            self._do_generate(str(select.value), input_dir, output_dir),
            thread=True,
        )

    async def _do_generate(self, model_name: str, input_dir: str, output_dir: str) -> None:
        """Run generation in a background thread."""
        mgr = self._get_model_manager()
        model_path = mgr.get_model_path(model_name)

        status = self.query_one("#gen-status", Static)
        progress = self.query_one("#gen-progress", ProgressBar)

        try:
            import torch
            import torchaudio

            rave_model = torch.jit.load(str(model_path))
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            rave_model = rave_model.to(device)

            input_path = Path(input_dir)
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            wav_files = list(input_path.glob("*.wav"))
            total = len(wav_files)

            self.app.call_from_thread(progress.update, total=total, progress=0)

            generated = 0
            for i, wav_file in enumerate(wav_files):
                try:
                    x, sr = torchaudio.load(str(wav_file))
                    if sr != rave_model.sr:
                        x = torchaudio.functional.resample(x, sr, rave_model.sr)
                    x = x.to(device)

                    with torch.no_grad():
                        out = rave_model.forward(x[None])

                    out_path = output_path / f"{wav_file.stem}_gen.wav"
                    torchaudio.save(str(out_path), out[0].cpu(), sample_rate=rave_model.sr)
                    generated += 1
                except Exception:
                    pass

                self.app.call_from_thread(progress.update, progress=i + 1)

            self.app.call_from_thread(
                status.update,
                f"[green]Done! Generated {generated} samples[/green]"
            )

        except ImportError:
            self.app.call_from_thread(
                status.update,
                "[red]ML dependencies not installed. Run: pip install scropipe[ml][/red]"
            )

    def _delete_selected_model(self) -> None:
        table = self.query_one("#gen-models-table", DataTable)
        if table.cursor_row is not None:
            row_key = table.get_row_at(table.cursor_row)
            # The key is the model name
            cell_key = table.coordinate_to_cell_key(table.cursor_coordinate)
            model_name = cell_key.row_key.value
            if model_name:
                mgr = self._get_model_manager()
                mgr.delete_model(model_name)
                self._refresh_models()
                self.notify(f"Deleted model '{model_name}'")
```

**Step 4: Update `scropipe/tui/app.py` to use GenerateTab**

```python
from .generate_tab import GenerateTab

# In compose():
with TabPane("Generate", id="generate"):
    yield GenerateTab()
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/test_tui_generate.py -v`
Expected: All 3 PASS

**Step 6: Commit**

```bash
git add scropipe/tui/generate_tab.py scropipe/tui/app.py tests/test_tui_generate.py
git commit -m "feat: add Generate tab with model selector and generation controls"
```

---

## Task 12: Status Bar Updates

**Files:**
- Modify: `scropipe/tui/app.py`

The status bar should reactively update to show the current pool, model, and GPU status.

**Step 1: Add reactive state to `ScropipeApp`**

In `app.py`, add reactive variables and watchers:

```python
from textual.reactive import reactive

class ScropipeApp(App):
    # ... existing code ...

    active_pool: reactive[str] = reactive("none")
    active_model: reactive[str] = reactive("none")
    gpu_info: reactive[str] = reactive("detecting...")

    def watch_active_pool(self) -> None:
        self._update_status_bar()

    def watch_active_model(self) -> None:
        self._update_status_bar()

    def watch_gpu_info(self) -> None:
        self._update_status_bar()

    def _update_status_bar(self) -> None:
        try:
            bar = self.query_one("#status-bar", Static)
            bar.update(
                f"Pool: {self.active_pool} | Model: {self.active_model} "
                f"| GPU: {self.gpu_info} | Ctrl+Q: Quit"
            )
        except Exception:
            pass

    def on_mount(self) -> None:
        self._detect_gpu()
        if self.models_dir is None or self.pools_dir is None:
            self.push_screen(SetupModal(), self._on_setup_complete)

    def _detect_gpu(self) -> None:
        try:
            import torch
            if torch.cuda.is_available():
                self.gpu_info = torch.cuda.get_device_name(0)
            else:
                self.gpu_info = "CPU"
        except ImportError:
            self.gpu_info = "CPU"
```

**Step 2: Verify the status bar updates**

Run: `python -c "from scropipe.tui.app import ScropipeApp; app = ScropipeApp(); print('OK')"`
Expected: OK (no import errors)

**Step 3: Commit**

```bash
git add scropipe/tui/app.py
git commit -m "feat: add reactive status bar with pool, model, and GPU info"
```

---

## Task 13: Preset Integration

**Files:**
- Modify: `scropipe/tui/split_tab.py`
- Modify: `scropipe/tui/train_tab.py`

Add preset loading/saving to the Split and Train tabs. Uses the existing `load_preset()` function from `cli.py` and saves new presets as TOML files.

**Step 1: Extract `load_preset` and `save_preset` to a shared location**

Move the preset loading logic from `cli.py` into `config.py` so both CLI and TUI can use it:

In `scropipe/config.py`, add:

```python
def load_preset(preset_name: str, presets_dir: Optional[Path] = None) -> dict:
    """Load preset configuration from TOML file."""
    import tomllib

    preset_paths = [
        Path.cwd() / "presets" / f"{preset_name}.toml",
        Path(__file__).parent.parent / "presets" / f"{preset_name}.toml",
        Path.home() / ".config" / "scropipe" / "presets" / f"{preset_name}.toml",
    ]
    if presets_dir:
        preset_paths.insert(0, presets_dir / f"{preset_name}.toml")

    for path in preset_paths:
        if path.exists():
            with open(path, "rb") as f:
                return tomllib.load(f)

    raise FileNotFoundError(f"Preset not found: {preset_name}")


def list_presets(presets_dir: Optional[Path] = None) -> list[str]:
    """List available preset names."""
    preset_dirs = [
        Path.cwd() / "presets",
        Path(__file__).parent.parent / "presets",
        Path.home() / ".config" / "scropipe" / "presets",
    ]
    if presets_dir:
        preset_dirs.insert(0, presets_dir)

    names = set()
    for d in preset_dirs:
        if d.exists():
            for f in d.glob("*.toml"):
                names.add(f.stem)
    return sorted(names)


def save_preset(name: str, config: dict, presets_dir: Optional[Path] = None) -> Path:
    """Save a preset to the user's config directory."""
    target_dir = presets_dir or (Path.home() / ".config" / "scropipe" / "presets")
    target_dir.mkdir(parents=True, exist_ok=True)

    path = target_dir / f"{name}.toml"
    lines = []
    for section, values in config.items():
        lines.append(f"[{section}]")
        for key, value in values.items():
            if isinstance(value, bool):
                lines.append(f"{key} = {'true' if value else 'false'}")
            elif isinstance(value, str):
                lines.append(f'{key} = "{value}"')
            else:
                lines.append(f"{key} = {value}")
        lines.append("")

    path.write_text("\n".join(lines))
    return path
```

**Step 2: Add preset selector to Split tab**

In `split_tab.py`, add a `Select` widget for presets above the mode selector. When a preset is loaded, populate the form fields. When "Save Current" is pressed, save current values as a new preset.

Add to `SplitTab.compose()` (after the Source File section):

```python
yield Label("Preset:", classes="form-label")
with Horizontal(classes="form-group"):
    yield Select([], id="split-preset-select", prompt="None")
    yield Button("Save Current...", id="split-save-preset-btn")
```

Add to `SplitTab.on_mount()`:

```python
self._refresh_presets()
```

Add methods:

```python
def _refresh_presets(self) -> None:
    from ..config import list_presets
    presets = list_presets()
    select = self.query_one("#split-preset-select", Select)
    select.set_options([(p, p) for p in presets])

def on_select_changed(self, event: Select.Changed) -> None:
    if event.select.id == "split-preset-select" and event.value is not Select.BLANK:
        self._apply_preset(str(event.value))

def _apply_preset(self, name: str) -> None:
    from ..config import load_preset
    try:
        preset = load_preset(name)
    except FileNotFoundError:
        return

    split_config = preset.get("split", {})
    mode = split_config.get("mode", "transient")

    # Set mode radio button
    radio_set = self.query_one("#split-mode-selector", RadioSet)
    mode_index = {"transient": 0, "grid": 1, "texture": 2}.get(mode, 0)
    radio_set.pressed_index = mode_index

    # Set parameters
    if mode == "transient":
        if "delta" in split_config:
            self.query_one("#split-delta", Input).value = str(split_config["delta"])
        if "min_length" in split_config:
            self.query_one("#split-min-length", Input).value = str(split_config["min_length"])
        if "max_length" in split_config:
            self.query_one("#split-max-length", Input).value = str(split_config["max_length"])
```

**Step 3: Test presets load**

Run: `pytest tests/test_config.py -v`
Expected: All PASS (add new tests for list_presets/save_preset if desired)

**Step 4: Commit**

```bash
git add scropipe/config.py scropipe/tui/split_tab.py scropipe/tui/train_tab.py
git commit -m "feat: add preset loading and saving to Split and Train tabs"
```

---

## Task 14: Final Polish and Styles

**Files:**
- Modify: `scropipe/tui/styles.tcss`
- Modify: `scropipe/tui/app.py`

**Step 1: Update `styles.tcss` with complete styling**

```css
Screen {
    layout: vertical;
}

#status-bar {
    dock: bottom;
    height: 1;
    background: $surface;
    color: $text-muted;
    padding: 0 1;
}

TabbedContent {
    height: 1fr;
}

.tab-content {
    padding: 1 2;
    overflow-y: auto;
}

.section-title {
    margin-top: 1;
    text-style: bold;
}

.form-group {
    height: auto;
    margin-bottom: 1;
}

.form-group Input {
    width: 1fr;
}

.form-group Button {
    width: auto;
    min-width: 10;
}

.form-label {
    margin-top: 1;
    margin-bottom: 0;
}

.action-bar {
    margin-top: 1;
    height: auto;
}

.action-bar Button {
    margin-right: 1;
}

/* Pool tab layout */
#pool-sidebar {
    width: 30;
    height: 1fr;
    border-right: solid $accent;
    padding: 0 1;
}

#pool-detail {
    width: 1fr;
    padding: 0 1;
}

#pool-list {
    height: 1fr;
}

/* Train dashboard */
#dash-sparkline {
    height: 8;
    margin: 1 0;
}

/* Generate tab */
#gen-models-table {
    height: 10;
    margin: 1 0;
}

#gen-progress {
    margin: 1 0;
}
```

**Step 2: Verify the full app renders correctly**

Run: `python -m scropipe.cli`
Expected: TUI launches with styled tabs, status bar, proper layout. Press Ctrl+Q to exit.

**Step 3: Commit**

```bash
git add scropipe/tui/styles.tcss
git commit -m "feat: complete TUI styling"
```

---

## Task 15: Integration Test

**Files:**
- Create: `tests/test_tui_integration.py`

**Step 1: Write an integration test**

```python
"""Integration tests for the full TUI app."""

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
        # App should render without exceptions
        assert configured_app.title == "scropipe"


@pytest.mark.asyncio
async def test_tab_switching(configured_app):
    async with configured_app.run_test() as pilot:
        tabbed = configured_app.query_one("TabbedContent")

        tabbed.active = "pool"
        await pilot.pause()
        assert tabbed.active == "pool"

        tabbed.active = "train"
        await pilot.pause()
        assert tabbed.active == "train"

        tabbed.active = "generate"
        await pilot.pause()
        assert tabbed.active == "generate"

        tabbed.active = "split"
        await pilot.pause()
        assert tabbed.active == "split"


@pytest.mark.asyncio
async def test_keyboard_tab_switching(configured_app):
    async with configured_app.run_test() as pilot:
        await pilot.press("2")
        await pilot.pause()
        tabbed = configured_app.query_one("TabbedContent")
        assert tabbed.active == "pool"
```

**Step 2: Run all tests**

Run: `pytest tests/ -v`
Expected: All PASS

**Step 3: Commit**

```bash
git add tests/test_tui_integration.py
git commit -m "feat: add TUI integration tests"
```

---

## Summary of Files Created/Modified

### New files:
- `scropipe/config.py` - Config loading/saving
- `scropipe/pool_manager.py` - Pool CRUD operations
- `scropipe/model_manager.py` - Model CRUD operations
- `scropipe/tui/__init__.py` - TUI package init
- `scropipe/tui/app.py` - Main Textual app
- `scropipe/tui/split_tab.py` - Split tab UI
- `scropipe/tui/pool_tab.py` - Pool tab UI
- `scropipe/tui/train_tab.py` - Train tab UI (config + dashboard)
- `scropipe/tui/generate_tab.py` - Generate tab UI
- `scropipe/tui/styles.tcss` - Textual CSS styling
- `tests/test_config.py` - Config tests
- `tests/test_pool_manager.py` - Pool manager tests
- `tests/test_model_manager.py` - Model manager tests
- `tests/test_tui_app.py` - App shell tests
- `tests/test_tui_setup.py` - First-run setup tests
- `tests/test_tui_split.py` - Split tab tests
- `tests/test_tui_pool.py` - Pool tab tests
- `tests/test_tui_train.py` - Train tab tests
- `tests/test_tui_generate.py` - Generate tab tests
- `tests/test_tui_integration.py` - Integration tests

### Modified files:
- `pyproject.toml` - Add textual dependency
- `scropipe/cli.py` - TUI entry point (no-args launches TUI)
