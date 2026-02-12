"""Audio sample splitter - grid chopping, transient detection, and texture gating."""

from .processor import SampleProcessor, main
from .presets import main as batch_main, PRESETS

__all__ = ["SampleProcessor", "main", "batch_main", "PRESETS"]
