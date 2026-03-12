"""Parse RAVE/PyTorch Lightning training output for metrics."""

from __future__ import annotations

import re
from typing import Optional

# Epoch progress bar (actual RAVE output):
#   "Epoch 51:  43%|####2     | 15/35 [00:01<00:01, 13.84it/s, v_num=0]"
_EPOCH_RE = re.compile(
    r"Epoch\s+(\d+):\s+(\d+)%\|[^|]*\|\s*(\d+)/(\d+)\s*\["
)

# Checkpoint save: "saving model to '...'" or ModelCheckpoint messages
_CKPT_RE = re.compile(r"saving model to|ModelCheckpoint")

# Validation line: "Validation DataLoader" or "Sanity Checking"
_VAL_RE = re.compile(r"Validation|Sanity Checking")


def parse_training_line(line: str) -> Optional[dict]:
    """Parse a single line of RAVE training output.

    Returns:
        Dict with parsed info, or None if the line doesn't contain
        training metrics.  Possible keys:

        - ``epoch``  – current epoch number
        - ``step``   – global step (epoch * steps_per_epoch + batch)
        - ``batch``  – batch index within the epoch
        - ``steps_per_epoch`` – total batches per epoch
        - ``pct``    – percentage complete within the epoch
        - ``checkpoint`` – True when a checkpoint is saved
        - ``validation`` – True for validation lines
    """
    line = line.strip()
    if not line:
        return None

    # Checkpoint
    if _CKPT_RE.search(line):
        return {"checkpoint": True}

    # Validation (skip, don't count as training steps)
    if _VAL_RE.search(line):
        return {"validation": True}

    # Epoch progress
    m = _EPOCH_RE.search(line)
    if m:
        epoch = int(m.group(1))
        pct = int(m.group(2))
        batch = int(m.group(3))
        steps_per_epoch = int(m.group(4))
        global_step = epoch * steps_per_epoch + batch
        return {
            "epoch": epoch,
            "step": global_step,
            "batch": batch,
            "steps_per_epoch": steps_per_epoch,
            "pct": pct,
        }

    return None
