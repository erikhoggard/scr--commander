from scropipe.training_state import (
    TrainingRunInfo,
    find_checkpoint_dir,
    list_paused_runs,
    load_training_run,
    reconcile_stale_runs,
    save_training_run,
)


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


def test_reconcile_stale_runs(tmp_path):
    """Stale 'training' runs should be marked as 'paused'."""
    models_dir = tmp_path / "models"
    run_dir = models_dir / "stale-model"
    run_dir.mkdir(parents=True)
    run = TrainingRunInfo(
        model_name="stale-model",
        pool_name="drums",
        architecture="v2",
        output_dir=str(tmp_path / "out"),
        status="training",
    )
    save_training_run(run, run_dir)

    reconcile_stale_runs(models_dir)

    loaded = load_training_run(run_dir)
    assert loaded.status == "paused"


def test_reconcile_stale_runs_leaves_paused(tmp_path):
    """Already-paused runs stay paused."""
    models_dir = tmp_path / "models"
    run_dir = models_dir / "paused-model"
    run_dir.mkdir(parents=True)
    run = TrainingRunInfo(
        model_name="paused-model",
        pool_name="drums",
        architecture="v2",
        output_dir=str(tmp_path / "out"),
        status="paused",
    )
    save_training_run(run, run_dir)

    reconcile_stale_runs(models_dir)

    loaded = load_training_run(run_dir)
    assert loaded.status == "paused"


def test_find_checkpoint_dir(tmp_path):
    """Verify checkpoint directory discovery finds RAVE checkpoint dirs."""
    output_dir = tmp_path / "training_output"
    ckpt_dir = output_dir / "runs" / "model_abc123" / "version_0" / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "best.ckpt").write_text("fake")

    result = find_checkpoint_dir(output_dir)
    assert result is not None
    assert result.name == "checkpoints"
    assert (result / "best.ckpt").exists()


def test_find_checkpoint_dir_no_checkpoints(tmp_path):
    """An output dir without .ckpt files returns None."""
    output_dir = tmp_path / "training_output"
    output_dir.mkdir()
    result = find_checkpoint_dir(output_dir)
    assert result is None


def test_find_checkpoint_dir_nonexistent(tmp_path):
    """A nonexistent output dir returns None."""
    result = find_checkpoint_dir(tmp_path / "nope")
    assert result is None
