"""Scropipe - Audio pipeline for splitting, collecting, and synthesizing samples.

This package provides:
- scropipe.splitter: Audio splitting (grid, transient, texture)
- scropipe.synth: Neural audio synthesis (VAE, HiFi-GAN vocoder)
- scropipe.stages: Pipeline stages
- scropipe.cli: Command-line interface

Install options:
- pip install scropipe       - splitting only (lightweight)
- pip install scropipe[ml]   - full ML synthesis
- pip install scropipe[ml,dev] - development
"""

__version__ = "0.2.0"
