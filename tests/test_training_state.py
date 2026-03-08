import json
from scropipe.training_state import TrainingRunInfo, save_training_run, load_training_run, list_paused_runs


def test_save_and_load_training_run(tmp_path):
    run = TrainingRunInfo(
        model_name="my-model",
        pool_name="drums",
        architecture="v2",
        output_dir=str(tmp_path / "output"),
        status="training",
    )
    run_dir = tmp_path / "models" / "my-model"
    save_training_run(run, run_dir)
    loaded = load_training_run(run_dir)
    assert loaded.model_name == "my-model"
    assert loaded.pool_name == "drums"
    assert loaded.architecture == "v2"
    assert loaded.status == "training"


def test_list_paused_runs_filters_by_status(tmp_path):
    models_dir = tmp_path / "models"

    # Create a paused run
    paused_dir = models_dir / "paused-model"
    paused_dir.mkdir(parents=True)
    run1 = TrainingRunInfo(
        model_name="paused-model",
        pool_name="drums",
        architecture="v2",
        output_dir=str(tmp_path / "out1"),
        status="paused",
    )
    save_training_run(run1, paused_dir)

    # Create a completed run
    done_dir = models_dir / "done-model"
    done_dir.mkdir(parents=True)
    run2 = TrainingRunInfo(
        model_name="done-model",
        pool_name="drums",
        architecture="v2",
        output_dir=str(tmp_path / "out2"),
        status="completed",
    )
    save_training_run(run2, done_dir)

    paused = list_paused_runs(models_dir)
    assert len(paused) == 1
    assert paused[0].model_name == "paused-model"


def test_list_paused_runs_empty_dir(tmp_path):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    assert list_paused_runs(models_dir) == []


def test_list_paused_runs_nonexistent_dir(tmp_path):
    assert list_paused_runs(tmp_path / "nope") == []
