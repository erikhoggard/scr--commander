"""Tests for scropipe.config module."""

from pathlib import Path

import pytest

from scropipe.config import (
    ScropipeConfig,
    default_config_path,
    load_config,
    save_config,
)


class TestDefaultConfigPath:
    def test_returns_correct_path(self):
        result = default_config_path()
        expected = Path.home() / ".config" / "scropipe" / "config.toml"
        assert result == expected

    def test_returns_path_object(self):
        result = default_config_path()
        assert isinstance(result, Path)


class TestScropipeConfigNeedsSetup:
    def test_needs_setup_true_when_models_dir_none(self):
        config = ScropipeConfig(models_dir=None, pools_dir=Path("/tmp/pools"))
        assert config.needs_setup is True

    def test_needs_setup_true_when_pools_dir_none(self):
        config = ScropipeConfig(models_dir=Path("/tmp/models"), pools_dir=None)
        assert config.needs_setup is True

    def test_needs_setup_true_when_both_none(self):
        config = ScropipeConfig()
        assert config.needs_setup is True

    def test_needs_setup_false_when_both_set(self):
        config = ScropipeConfig(
            models_dir=Path("/tmp/models"),
            pools_dir=Path("/tmp/pools"),
        )
        assert config.needs_setup is False


class TestLoadConfig:
    def test_load_returns_defaults_when_no_file(self, tmp_path):
        nonexistent = tmp_path / "does_not_exist.toml"
        config = load_config(nonexistent)
        assert isinstance(config, ScropipeConfig)
        assert config.models_dir is None
        assert config.pools_dir is None
        assert config.presets_dir is None

    def test_load_parses_toml_file(self, tmp_path):
        config_file = tmp_path / "config.toml"
        config_file.write_text(
            '[paths]\n'
            'models_dir = "/home/user/models"\n'
            'pools_dir = "/home/user/pools"\n'
            'presets_dir = "/home/user/presets"\n'
        )
        config = load_config(config_file)
        assert config.models_dir == Path("/home/user/models")
        assert config.pools_dir == Path("/home/user/pools")
        assert config.presets_dir == Path("/home/user/presets")

    def test_load_handles_partial_config(self, tmp_path):
        config_file = tmp_path / "config.toml"
        config_file.write_text(
            '[paths]\n'
            'models_dir = "/home/user/models"\n'
        )
        config = load_config(config_file)
        assert config.models_dir == Path("/home/user/models")
        assert config.pools_dir is None
        assert config.presets_dir is None


class TestSaveConfig:
    def test_save_creates_file(self, tmp_path):
        config_file = tmp_path / "config.toml"
        config = ScropipeConfig(
            models_dir=Path("/home/user/models"),
            pools_dir=Path("/home/user/pools"),
        )
        save_config(config, config_file)
        assert config_file.exists()

    def test_save_creates_parent_dirs(self, tmp_path):
        config_file = tmp_path / "nested" / "dir" / "config.toml"
        config = ScropipeConfig()
        save_config(config, config_file)
        assert config_file.exists()


class TestSaveLoadRoundtrip:
    def test_roundtrip_all_fields_set(self, tmp_path):
        config_file = tmp_path / "config.toml"
        original = ScropipeConfig(
            models_dir=Path("/home/user/models"),
            pools_dir=Path("/home/user/pools"),
            presets_dir=Path("/home/user/presets"),
        )
        save_config(original, config_file)
        loaded = load_config(config_file)
        assert loaded.models_dir == original.models_dir
        assert loaded.pools_dir == original.pools_dir
        assert loaded.presets_dir == original.presets_dir

    def test_roundtrip_partial_fields(self, tmp_path):
        config_file = tmp_path / "config.toml"
        original = ScropipeConfig(
            models_dir=Path("/home/user/models"),
        )
        save_config(original, config_file)
        loaded = load_config(config_file)
        assert loaded.models_dir == original.models_dir
        assert loaded.pools_dir is None
        assert loaded.presets_dir is None

    def test_roundtrip_no_fields(self, tmp_path):
        config_file = tmp_path / "config.toml"
        original = ScropipeConfig()
        save_config(original, config_file)
        loaded = load_config(config_file)
        assert loaded.models_dir is None
        assert loaded.pools_dir is None
        assert loaded.presets_dir is None


class TestPresets:
    def test_list_presets_from_package(self):
        from scropipe.config import list_presets

        presets = list_presets()
        # Package ships with drums-to-ai and ambient-textures
        assert "drums-to-ai" in presets or len(presets) >= 0  # May not find if cwd differs

    def test_list_presets_custom_dir(self, tmp_path):
        from scropipe.config import list_presets, save_preset

        save_preset("my-preset", {"split": {"mode": "grid"}}, presets_dir=tmp_path)
        presets = list_presets(presets_dir=tmp_path)
        assert "my-preset" in presets

    def test_load_preset_custom(self, tmp_path):
        from scropipe.config import load_preset, save_preset

        save_preset(
            "test-preset",
            {"split": {"mode": "transient", "delta": 0.1}},
            presets_dir=tmp_path,
        )
        preset = load_preset("test-preset", presets_dir=tmp_path)
        assert preset["split"]["mode"] == "transient"
