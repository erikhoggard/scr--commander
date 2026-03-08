"""Parse RAVE/PyTorch Lightning training output for metrics."""

from __future__ import annotations

import re
from typing import Optional

# Epoch progress: "Epoch 0:  50%|███| 500/1000 [01:23<01:23, 6.00it/s, loss=0.123]"
_EPOCH_RE = re.compile(
    r"Epoch\s+\d+:\s+\d+%\|[^|]*\|\s*(\d+)/\d+\s*\[.*?loss=([\d.]+)"
)

# Step-level log: "Step 1500: loss=0.0891"
_STEP_RE = re.compile(r"Step\s+(\d+):\s*loss=([\d.]+)")

# Checkpoint save: "saving model to '...'"
_CKPT_RE = re.compile(r"saving model to")


def parse_training_line(line: str) -> Optional[dict]:
    """Parse a single line of RAVE training output.

    Returns:
        Dict with parsed info (step, loss, checkpoint), or None if
        the line doesn't contain training metrics.
    """
    line = line.strip()
    if not line:
        return None

    # Check for checkpoint save
    if _CKPT_RE.search(line):
        return {"checkpoint": True}

    # Check epoch progress
    m = _EPOCH_RE.search(line)
    if m:
        return {
            "step": int(m.group(1)),
            "loss": float(m.group(2)),
        }

    # Check step-level log
    m = _STEP_RE.search(line)
    if m:
        return {
            "step": int(m.group(1)),
            "loss": float(m.group(2)),
        }

    return None
