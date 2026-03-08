# tests/test_rave_runner.py
from pathlib import Path

from scropipe.tui.rave_runner import build_preprocess_cmd, build_train_cmd


def test_build_preprocess_cmd():
    cmd = build_preprocess_cmd(
        rave_cmd="/usr/bin/rave",
        input_dir=Path("/data/samples"),
        output_dir=Path("/data/preprocessed"),
    )
    assert cmd == [
        "/usr/bin/rave", "preprocess",
        "--input_path", "/data/samples",
        "--output_path", "/data/preprocessed",
        "--num_signal", "131072",
    ]


def test_build_train_cmd_basic():
    cmd = build_train_cmd(
        rave_cmd="/usr/bin/rave",
        config="v2",
        data_dir=Path("/data/preprocessed"),
        name="my-model",
        val_every=500,
    )
    assert "/usr/bin/rave" in cmd
    assert "train" in cmd
    assert "--config" in cmd
    assert "v2" in cmd
    assert "--val_every" in cmd
    assert "500" in cmd


def test_build_train_cmd_with_checkpoint():
    cmd = build_train_cmd(
        rave_cmd="/usr/bin/rave",
        config="v2",
        data_dir=Path("/data/preprocessed"),
        name="my-model",
        val_every=500,
        ckpt=Path("/checkpoints/best.ckpt"),
    )
    assert "--ckpt" in cmd
    assert "/checkpoints/best.ckpt" in cmd


def test_build_train_cmd_with_max_steps():
    cmd = build_train_cmd(
        rave_cmd="/usr/bin/rave",
        config="v2",
        data_dir=Path("/data/preprocessed"),
        name="my-model",
        val_every=500,
        max_steps=10000,
    )
    assert "--max_steps" in cmd
    assert "10000" in cmd


def test_build_train_cmd_with_gpu():
    cmd = build_train_cmd(
        rave_cmd="/usr/bin/rave",
        config="v2",
        data_dir=Path("/data/preprocessed"),
        name="my-model",
        val_every=500,
        gpu=0,
    )
    assert "--gpu" in cmd
    assert "0" in cmd
