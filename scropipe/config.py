"""Persistent configuration for scropipe.

Stores user settings (directory paths) in a TOML file at
~/.config/scropipe/config.toml.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional


@dataclass
class ScropipeConfig:
    """Application configuration.

    Attributes:
        models_dir: Directory where trained models are stored.
        pools_dir: Directory where sample pools are stored.
        presets_dir: Directory where splitter presets are stored.
    """

    models_dir: Optional[Path] = None
    pools_dir: Optional[Path] = None
    presets_dir: Optional[Path] = None

    @property
    def needs_setup(self) -> bool:
        """Return True when essential directories have not been configured."""
        return self.models_dir is None or self.pools_dir is None


def default_config_path() -> Path:
    """Return the default path for the scropipe config file."""
    return Path.home() / ".config" / "scropipe" / "config.toml"


def load_config(path: Optional[Path] = None) -> ScropipeConfig:
    """Load configuration from a TOML file.

    Returns a default ScropipeConfig (all fields None) if the file does not
    exist.
    """
    if path is None:
        path = default_config_path()

    if not path.exists():
        return ScropipeConfig()

    with open(path, "rb") as f:
        data = tomllib.load(f)

    paths = data.get("paths", {})
    return ScropipeConfig(
        models_dir=Path(paths["models_dir"]) if paths.get("models_dir") else None,
        pools_dir=Path(paths["pools_dir"]) if paths.get("pools_dir") else None,
        presets_dir=Path(paths["presets_dir"]) if paths.get("presets_dir") else None,
    )


def _toml_value(value: Optional[Path]) -> str:
    """Format a Path value as a TOML string, or return an empty string to omit it."""
    if value is None:
        return ""
    return f'"{value}"'


def save_config(config: ScropipeConfig, path: Optional[Path] = None) -> None:
    """Save configuration to a TOML file.

    Creates parent directories if they do not exist.  Only writes fields that
    have non-None values.
    """
    if path is None:
        path = default_config_path()

    path.parent.mkdir(parents=True, exist_ok=True)

    lines = ["[paths]"]
    for field in fields(config):
        value = getattr(config, field.name)
        if value is not None:
            # Use forward slashes so TOML doesn't treat backslashes as escapes
            lines.append(f'{field.name} = "{str(value).replace(chr(92), "/")}"')

    path.write_text("\n".join(lines) + "\n")


def load_preset(preset_name: str, presets_dir: Optional[Path] = None) -> dict:
    """Load preset configuration from TOML file."""
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
    names: set[str] = set()
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
    lines: list[str] = []
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
