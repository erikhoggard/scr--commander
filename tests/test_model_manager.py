"""Tests for scropipe.model_manager."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scropipe.model_manager import ModelInfo, ModelManager


def _create_fake_model(
    models_dir: Path,
    name: str,
    *,
    config: str = "v2",
    total_samples: int = 100,
    created: str = "2025-01-15T10:30:00",
    pool_name: str | None = None,
) -> Path:
    """Create a fake model directory with model.ts and metadata.json."""
    model_dir = models_dir / name
    model_dir.mkdir(parents=True, exist_ok=True)

    # Create a fake model.ts file with some content so it has a nonzero size
    model_file = model_dir / "model.ts"
    model_file.write_bytes(b"\x00" * 1024)  # 1 KB fake model

    metadata = {
        "name": name,
        "created": created,
        "config": config,
        "total_samples": total_samples,
        "sources": {
            "sample_dirs": ["/fake/samples"],
            "audio_files": [],
        },
    }
    if pool_name is not None:
        metadata["pool_name"] = pool_name

    metadata_path = model_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return model_dir


class TestModelManager:
    def test_list_models_empty(self, tmp_path: Path) -> None:
        """list_models returns an empty list when models_dir has no models."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        manager = ModelManager(models_dir)

        result = manager.list_models()

        assert result == []

    def test_list_models_empty_dir_not_exist(self, tmp_path: Path) -> None:
        """list_models returns an empty list when models_dir does not exist."""
        models_dir = tmp_path / "models"
        manager = ModelManager(models_dir)

        result = manager.list_models()

        assert result == []

    def test_list_models(self, tmp_path: Path) -> None:
        """list_models returns ModelInfo for each model found."""
        models_dir = tmp_path / "models"
        _create_fake_model(models_dir, "drum-kit", config="v2", total_samples=200)
        _create_fake_model(models_dir, "bass-synth", config="v1", total_samples=50)

        manager = ModelManager(models_dir)
        result = manager.list_models()

        assert len(result) == 2
        names = {m.name for m in result}
        assert names == {"drum-kit", "bass-synth"}

        # Check that fields are populated correctly
        drum = next(m for m in result if m.name == "drum-kit")
        assert drum.config == "v2"
        assert drum.total_samples == 200
        assert drum.created == "2025-01-15T10:30:00"
        assert drum.model_path == models_dir / "drum-kit" / "model.ts"
        assert drum.size_mb > 0

    def test_list_models_skips_dirs_without_model_ts(self, tmp_path: Path) -> None:
        """list_models ignores directories that don't contain model.ts."""
        models_dir = tmp_path / "models"
        _create_fake_model(models_dir, "valid-model")

        # Create a directory without model.ts (e.g. leftover temp dir)
        stray_dir = models_dir / "not-a-model"
        stray_dir.mkdir()

        manager = ModelManager(models_dir)
        result = manager.list_models()

        assert len(result) == 1
        assert result[0].name == "valid-model"

    def test_list_models_with_pool_name(self, tmp_path: Path) -> None:
        """list_models populates pool_name when present in metadata."""
        models_dir = tmp_path / "models"
        _create_fake_model(models_dir, "my-model", pool_name="kick-pool")

        manager = ModelManager(models_dir)
        result = manager.list_models()

        assert len(result) == 1
        assert result[0].pool_name == "kick-pool"

    def test_get_model(self, tmp_path: Path) -> None:
        """get_model returns the correct ModelInfo by name."""
        models_dir = tmp_path / "models"
        _create_fake_model(models_dir, "test-model", config="v3", total_samples=42)

        manager = ModelManager(models_dir)
        model = manager.get_model("test-model")

        assert isinstance(model, ModelInfo)
        assert model.name == "test-model"
        assert model.config == "v3"
        assert model.total_samples == 42

    def test_get_model_path(self, tmp_path: Path) -> None:
        """get_model_path returns the path to model.ts."""
        models_dir = tmp_path / "models"
        _create_fake_model(models_dir, "my-model")

        manager = ModelManager(models_dir)
        path = manager.get_model_path("my-model")

        expected = models_dir / "my-model" / "model.ts"
        assert path == expected
        assert path.exists()

    def test_delete_model(self, tmp_path: Path) -> None:
        """delete_model removes the entire model directory."""
        models_dir = tmp_path / "models"
        _create_fake_model(models_dir, "doomed-model")

        manager = ModelManager(models_dir)

        # Verify model exists first
        assert (models_dir / "doomed-model").exists()

        manager.delete_model("doomed-model")

        # Directory should be gone
        assert not (models_dir / "doomed-model").exists()
        # And list_models should no longer include it
        assert manager.list_models() == []

    def test_get_nonexistent_raises(self, tmp_path: Path) -> None:
        """get_model raises KeyError for a model that doesn't exist."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()

        manager = ModelManager(models_dir)

        with pytest.raises(KeyError):
            manager.get_model("ghost-model")

    def test_get_model_path_nonexistent_raises(self, tmp_path: Path) -> None:
        """get_model_path raises KeyError for a model that doesn't exist."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()

        manager = ModelManager(models_dir)

        with pytest.raises(KeyError):
            manager.get_model_path("ghost-model")

    def test_delete_nonexistent_raises(self, tmp_path: Path) -> None:
        """delete_model raises KeyError for a model that doesn't exist."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()

        manager = ModelManager(models_dir)

        with pytest.raises(KeyError):
            manager.delete_model("ghost-model")

    def test_model_without_metadata(self, tmp_path: Path) -> None:
        """A model directory with model.ts but no metadata.json still works."""
        models_dir = tmp_path / "models"
        model_dir = models_dir / "bare-model"
        model_dir.mkdir(parents=True)
        (model_dir / "model.ts").write_bytes(b"\x00" * 2048)

        manager = ModelManager(models_dir)
        result = manager.list_models()

        assert len(result) == 1
        model = result[0]
        assert model.name == "bare-model"
        assert model.config == ""
        assert model.created == ""
        assert model.total_samples == 0
        assert model.size_mb > 0
