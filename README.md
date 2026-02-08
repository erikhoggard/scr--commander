# Scropipe

Audio pipeline orchestrator for splitting, collecting, and synthesizing audio samples.

Scropipe chains together external tools ([Scrumpler](https://github.com/erikhoggard/scrumpler) for audio splitting, [Scronchler](https://github.com/erikhoggard/scronchler) for AI synthesis) into a streamlined workflow.

## Installation

Requires Python 3.10+

```bash
pip install .
```

For development:
```bash
pip install -e ".[dev]"
```

## External Tools

Scropipe requires these external tools:

- **scrumpler** - Audio splitter (for `split` operations)
- **scronchler** - VAE-based sample synthesis (for `synthesize` operations)

Check tool availability:
```bash
scropipe tools
```

Set custom paths via environment variables:
```bash
export SCRUMPLER_PATH=/path/to/scrumpler
export SCRONCHLER_PATH=/path/to/scronchler
```

## Usage

### Full Pipeline

```bash
# Split audio file(s) into samples
scropipe run -i recording.wav

# Split multiple files
scropipe run -i drums.wav -i percussion.wav

# Include existing samples without splitting
scropipe run -I ~/samples/drums/

# Mix: split files + include existing samples, then synthesize
scropipe run -i recording.wav -I ~/samples/kicks/ --synthesize

# Full pipeline with custom settings
scropipe run -i drums.wav --synthesize --epochs 200 --count 50
```

### Individual Commands

```bash
# Split a single file
scropipe split recording.wav --mode transient -o ./samples

# Collect samples from multiple directories
scropipe collect ./split-output ~/samples/drums -o ./pool

# Run synthesis on existing samples
scropipe synthesize ./samples --epochs 150 --count 30
```

### Using Presets

```bash
scropipe run -i drums.wav --preset drums-to-ai --synthesize
scropipe run -i ambient.wav --preset ambient-textures --synthesize
```

## Commands

### `run`

Run the complete pipeline with flexible input sources.

| Option | Description |
|--------|-------------|
| `-i, --input` | Input audio file to split (repeatable) |
| `-I, --include` | Directory of existing samples to include (repeatable) |
| `-o, --output` | Output directory (default: `./scropipe-output`) |
| `-s, --split` | Split mode: `transient`, `grid`, or `texture` |
| `--synthesize` | Run synthesis stages after collecting |
| `-p, --preset` | Use a preset configuration |
| `-r, --resume` | Resume from existing pipeline state |

Split options:
| Option | Description |
|--------|-------------|
| `--delta` | Transient detection sensitivity (default: 0.07) |
| `--min-length` | Minimum segment length in seconds (default: 0.05) |
| `--max-length` | Maximum segment length in seconds (default: 10.0) |
| `--chunk-length` | Fixed chunk length for grid mode |
| `--bpm` | BPM for musical grid chopping |
| `--bars` | Bars per chunk when using `--bpm` (default: 4) |

Synthesis options:
| Option | Description |
|--------|-------------|
| `-a, --augment` | Augment samples during preprocessing |
| `--max-duration` | Maximum sample duration in seconds (default: 2.0) |
| `-e, --epochs` | Training epochs (default: 100) |
| `-c, --count` | Number of samples to generate (default: 10) |
| `--train-vocoder` | Train HiFi-GAN vocoder for higher quality output |
| `--vocoder-epochs` | Vocoder training epochs (default: 50) |

### `split`

Split a single audio file into samples.

```bash
scropipe split input.wav --mode transient -o ./output
```

### `collect`

Pool samples from multiple directories.

```bash
scropipe collect dir1 dir2 dir3 -o ./pool --symlink
```

### `synthesize`

Run the synthesis pipeline (preprocess, train, generate) on existing samples.

```bash
scropipe synthesize ./samples -o ./output --epochs 150 --count 50
```

### `tools`

Show status of required external tools.

```bash
scropipe tools
```

## Presets

Presets are TOML configuration files. Scropipe looks for presets in:

1. `./presets/<name>.toml`
2. `<package>/presets/<name>.toml`
3. `~/.config/scropipe/presets/<name>.toml`

### Included Presets

**drums-to-ai** - Optimized for drum samples
```toml
[split]
mode = "transient"
delta = 0.1
min_length = 0.05
max_length = 2.0

[preprocess]
augment = true
max_duration = 2.0

[train]
epochs = 150

[generate]
count = 20
```

**ambient-textures** - Optimized for ambient/drone samples
```toml
[split]
mode = "texture"
min_duration = 2.0
max_duration = 10.0

[preprocess]
augment = false
max_duration = 8.0

[train]
epochs = 200

[generate]
count = 10
```

## Pipeline Stages

1. **Split** - Chop audio files using Scrumpler (transient detection, grid, or texture gating)
2. **Collect** - Pool samples from split outputs and included directories
3. **Preprocess** - Convert samples to mel-spectrograms for training
4. **Train** - Train a VAE model on the preprocessed data
5. **Train Vocoder** (optional) - Train HiFi-GAN for higher quality audio (with `--train-vocoder`)
6. **Generate** - Synthesize new samples from the trained model

## Tips for Better Results

VAEs need sufficient training data to learn meaningful representations. If your generated samples sound similar or have artifacts:

### Get More Training Data

```bash
# Lower delta = more aggressive splitting = more samples
scropipe run -i recording.wav --synthesize --delta 0.005

# Split multiple source files
scropipe run -i drums.wav -i percussion.wav -i hits.wav --synthesize

# Include existing sample libraries
scropipe run -i recording.wav -I ~/samples/drums/ --synthesize
```

### Use Data Augmentation

The `--augment` flag creates pitch-shifted and time-stretched variants of your samples:

```bash
scropipe run -i recording.wav --synthesize --augment --epochs 200
```

### Adjust Training Parameters

```bash
# More epochs for better learning (but watch for overfitting)
scropipe run -i drums.wav --synthesize --epochs 300

# Generate more samples to find good ones
scropipe run -i drums.wav --synthesize --count 100
```

### Sample Count Guidelines

- **< 50 samples**: Results will likely be poor or repetitive
- **50-200 samples**: Reasonable results with augmentation
- **200+ samples**: Best results, model can learn meaningful variations

### Neural Vocoder (Optional)

For potentially cleaner audio, train a HiFi-GAN vocoder:

```bash
scropipe run -i drums.wav --synthesize --train-vocoder --vocoder-epochs 50
```

| Dataset Size | Recommendation |
|--------------|----------------|
| < 100 samples | Skip vocoder, use Griffin-Lim |
| 100-300 samples | Vocoder might help, worth trying |
| 300+ samples | Vocoder likely to outperform Griffin-Lim |

The vocoder helps most with tonal/melodic content. For percussive/noise sounds, Griffin-Lim often works fine.

## License

MIT
