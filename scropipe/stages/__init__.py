"""Pipeline stages for scropipe."""

from .base import Stage, StageResult
from .split import SplitStage
from .collect import CollectStage
from .preprocess import PreprocessStage
from .train import TrainStage
from .train_vocoder import TrainVocoderStage
from .generate import GenerateStage

__all__ = [
    "Stage",
    "StageResult",
    "SplitStage",
    "CollectStage",
    "PreprocessStage",
    "TrainStage",
    "TrainVocoderStage",
    "GenerateStage",
]
