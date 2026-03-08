from scropipe.tui.rave_parser import parse_training_line


def test_parse_epoch_line():
    # Lightning 1.9 style progress line
    line = "Epoch 0:  50%|█████     | 500/1000 [01:23<01:23, 6.00it/s, loss=0.123]"
    result = parse_training_line(line)
    assert result is not None
    assert result["step"] == 500
    assert abs(result["loss"] - 0.123) < 1e-6


def test_parse_epoch_line_v2():
    # Another common Lightning format
    line = "Epoch 1:  10%|█         | 100/1000 [00:30<04:30, 3.33it/s, loss=0.456]"
    result = parse_training_line(line)
    assert result is not None
    assert result["step"] == 100


def test_parse_validation_line():
    line = "Validation: 100%|██████████| 10/10 [00:05<00:00, 2.00it/s]"
    result = parse_training_line(line)
    # Validation lines don't have loss — return None
    assert result is None


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


def test_parse_step_log_line():
    # Some Lightning configs log step-level metrics
    line = "Step 1500: loss=0.0891"
    result = parse_training_line(line)
    assert result is not None
    assert result["step"] == 1500


def test_parse_empty_line():
    assert parse_training_line("") is None
    assert parse_training_line("\n") is None
