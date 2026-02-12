"""Neural Audio Synthesizer - VAE-based audio sample generator.

This module provides ML-based audio synthesis capabilities. It requires
the [ml] optional dependencies to be installed:

    pip install scropipe[ml]

The ML dependencies are lazily imported to allow the base package to
work without torch/librosa.
"""

__version__ = "0.2.0"


def _check_ml_deps():
    """Check if ML dependencies are available."""
    try:
        import torch
        import librosa
        return True
    except ImportError:
        return False


def main():
    """Entry point for the scronchler CLI."""
    if not _check_ml_deps():
        import sys
        print("ERROR: ML dependencies not installed.")
        print("Please run: pip install scropipe[ml]")
        sys.exit(1)

    from .main import main as _main
    _main()


# Lazy exports - these will raise ImportError if ML deps not installed
def __getattr__(name):
    """Lazy import ML modules only when accessed."""
    if name == "audio_utils":
        from . import audio_utils
        return audio_utils
    elif name == "model":
        from . import model
        return model
    elif name == "data_loader":
        from . import data_loader
        return data_loader
    elif name == "train_utils":
        from . import train_utils
        return train_utils
    elif name == "vocoder":
        from . import vocoder
        return vocoder
    elif name == "vocoder_train":
        from . import vocoder_train
        return vocoder_train
    elif name == "rave_utils":
        from . import rave_utils
        return rave_utils
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
