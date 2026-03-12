from scropipe.tui.rave_parser import parse_training_line


def test_parse_epoch_line():
    # Actual RAVE output format (v_num, no loss on stdout)
    line = "Epoch 0:  50%|█████     | 500/1000 [01:23<01:23, 6.00it/s, v_num=0]"
    result = parse_training_line(line)
    assert result is not None
    assert result["epoch"] == 0
    assert result["batch"] == 500
    assert result["steps_per_epoch"] == 1000
    assert result["step"] == 500  # epoch * steps_per_epoch + batch
    assert result["pct"] == 50


def test_parse_epoch_line_v2():
    # Second epoch
    line = "Epoch 1:  10%|█         | 100/1000 [00:30<04:30, 3.33it/s, v_num=0]"
    result = parse_training_line(line)
    assert result is not None
    assert result["epoch"] == 1
    assert result["step"] == 1100  # 1 * 1000 + 100


def test_parse_validation_line():
    line = "Validation: 100%|██████████| 10/10 [00:05<00:00, 2.00it/s]"
    result = parse_training_line(line)
    assert result is not None
    assert result.get("validation") is True


def test_parse_checkpoint_line():
    line = (
        "Epoch 0, global step 500: 'valid' reached 0.12300"
        " (best 0.12300), saving model to"
        " '/runs/model/version_0/checkpoints/best.ckpt'"
    )
    result = parse_training_line(line)
    assert result is not None
    assert result.get("checkpoint") is True


def test_parse_unrelated_line():
    line = "GPU available: True (cuda), used: True"
    result = parse_training_line(line)
    assert result is None


def test_parse_sanity_checking():
    line = "Sanity Checking: 100%|██████████| 2/2 [00:01<00:00, 1.50it/s]"
    result = parse_training_line(line)
    assert result is not None
    assert result.get("validation") is True


def test_parse_empty_line():
    assert parse_training_line("") is None
    assert parse_training_line("\n") is None
