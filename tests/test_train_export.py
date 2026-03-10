import json
from pathlib import Path
from scropipe.model_manager import ModelManager


def test_exported_model_visible_in_model_manager(tmp_path):
    """After export, model.ts and metadata.json make the model discoverable."""
    model_dir = tmp_path / "models" / "test-model"
    model_dir.mkdir(parents=True)

    # Simulate what the export step should produce
    (model_dir / "model.ts").write_bytes(b"fake torchscript")
    metadata = {
        "name": "test-model",
        "created": "2026-03-10T00:00:00",
        "config": "v2",
        "total_samples": 0,
        "pool_name": "drums",
    }
    (model_dir / "metadata.json").write_text(json.dumps(metadata))

    mm = ModelManager(tmp_path / "models")
    models = mm.list_models()
    assert len(models) == 1
    assert models[0].name == "test-model"
    assert models[0].config == "v2"
    assert models[0].pool_name == "drums"
