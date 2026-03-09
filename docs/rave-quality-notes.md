# RAVE Audio Quality Notes

## Current Setup
- Architecture: v2 (default 44.1kHz, ~2048x compression ratio)
- Channels: mono
- Window size: 131072 samples (~6s chunks)

## Training Duration
- 5k steps is far too early for usable quality
- Adversarial phase kicks in ~1M steps
- 2M+ steps needed for good timbral reconstruction
- 3M+ for diminishing returns

## Quality Limitations
RAVE v2 is not a high-fidelity codec. At ~21 latent frames/second with 8-16 dimensions, expect:
- Softened transients
- Reduced high-frequency detail
- Spectral smearing
- Characteristic "neural" quality

## Things to Investigate
- Larger latent dimensions (better quality, less manipulable)
- `v2_small` with fewer compression layers for higher fidelity
- EnCodec or DAC for higher-fidelity neural audio codecs (different use case than timbral transfer)

## Naming Inconsistency
- `stages/rave.py` maps `epochs` to `--max_steps`
- `synth/rave_utils.py` maps `epochs` to `--max_epochs`
- TUI uses `--max_steps` directly
- Should align naming to avoid confusion
