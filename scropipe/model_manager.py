"""Model manager for discovering and managing trained RAVE models on disk.

Models are stored in a models directory with the following structure:

    models_dir/
      model-name/
        model.ts         # TorchScript model
        metadata.json    # {"name", "created", "config", "total_samples", "sources", ...}
        checkpoints/     # optional
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelInfo:
    """Information about a trained model."""

    name: str
    created: str
    config: str
    total_samples: int
    model_path: Path
    size_mb: float = 0.0
    pool_name: str | None = None


class ModelManager:
    """Manages trained RAVE models stored on disk.

    Args:
        models_dir: Root directory containing model subdirectories.
    """

    def __init__(self, models_dir: Path) -> None:
        self.models_dir = models_dir

    def list_models(self) -> list[ModelInfo]:
        """Scan models_dir for subdirectories containing model.ts.

        Returns:
            List of ModelInfo for each valid model found, sorted by name.
        """
        if not self.models_dir.exists():
            return []

        models: list[ModelInfo] = []
        for model_dir in sorted(self.models_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            model_ts = model_dir / "model.ts"
            if not model_ts.exists():
                continue
            models.append(self._load_model_info(model_dir))

        return models

    def get_model(self, name: str) -> ModelInfo:
        """Get info for a specific model by name.

        Args:
            name: The model directory name.

        Returns:
            ModelInfo for the requested model.

        Raises:
            KeyError: If the model does not exist or has no model.ts file.
        """
        model_dir = self.models_dir / name
        model_ts = model_dir / "model.ts"
        if not model_ts.exists():
            raise KeyError(f"Model not found: {name}")
        return self._load_model_info(model_dir)

    def get_model_path(self, name: str) -> Path:
        """Get the path to a model's TorchScript file.

        Args:
            name: The model directory name.

        Returns:
            Path to the model.ts file.

        Raises:
            KeyError: If the model does not exist or has no model.ts file.
        """
        model_dir = self.models_dir / name
        model_ts = model_dir / "model.ts"
        if not model_ts.exists():
            raise KeyError(f"Model not found: {name}")
        return model_ts

    def delete_model(self, name: str) -> None:
        """Delete a model and its entire directory.

        Args:
            name: The model directory name.

        Raises:
            KeyError: If the model does not exist.
        """
        model_dir = self.models_dir / name
        if not model_dir.exists():
            raise KeyError(f"Model not found: {name}")
        shutil.rmtree(model_dir)

    def _load_model_info(self, model_dir: Path) -> ModelInfo:
        """Load ModelInfo from a model directory.

        Reads metadata.json if it exists; falls back to defaults for missing fields.
        """
        model_ts = model_dir / "model.ts"
        metadata_path = model_dir / "metadata.json"

        name = model_dir.name
        created = ""
        config = ""
        total_samples = 0
        pool_name: str | None = None

        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
            name = metadata.get("name", model_dir.name)
            created = metadata.get("created", "")
            config = metadata.get("config", "")
            total_samples = metadata.get("total_samples", 0)
            pool_name = metadata.get("pool_name")

        size_mb = model_ts.stat().st_size / (1024 * 1024) if model_ts.exists() else 0.0

        return ModelInfo(
            name=name,
            created=created,
            config=config,
            total_samples=total_samples,
            model_path=model_ts,
            size_mb=size_mb,
            pool_name=pool_name,
        )
