# Scropipe

Audio pipeline for splitting, collecting, and synthesizing samples.

Scropipe is a monorepo combining audio splitting (previously scrumpler), neural synthesis (previously scronchler), and RAVE integration into a unified package.

## Installation

Requires Python 3.12+

### Install Options

```bash
# Splitting only (lightweight, no ML dependencies)
pip install scropipe

# Full ML synthesis (includes PyTorch, torchaudio, librosa)
pip install scropipe[ml]

# Development
pip install scropipe[ml,dev]
```

### Linux (with Nix)

The Nix flake handles all dependencies including ML:

```bash
# Enter development shell (installs everything automatically)
nix develop

# Or if using direnv
direnv allow
```

The Nix shell provides:
- Python 3.12 environment with all dependencies
- ML dependencies with ROCm GPU support (AMD) or CPU fallback
- ffmpeg (required by RAVE)
- RAVE CLI with GPU acceleration

### Linux (without Nix)

```bash
# Install system dependencies
sudo apt install ffmpeg  # Debian/Ubuntu
sudo pacman -S ffmpeg    # Arch

# Install PyTorch (NVIDIA GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Or for AMD GPU (ROCm)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2

# Install scropipe with ML support
pip install -e ".[ml]"

# Install RAVE in a separate venv (see RAVE Installation section below)
```

### Windows

**Important:** PyTorch does not ship wheels for Python 3.13+. You **must** use Python 3.12. If you also have a newer Python installed, Windows may resolve `python` to the wrong version — see Troubleshooting below.

```powershell
# Install Python 3.12 from python.org (NOT 3.13+)

# Install ffmpeg (using chocolatey, or download from https://www.gyan.dev/ffmpeg/builds/)
choco install ffmpeg

# Install PyTorch with CUDA (for NVIDIA GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install scropipe with ML support
pip install -e ".[ml]"

# Install RAVE in a separate venv (see RAVE Installation section below)
```

### RAVE Installation

RAVE (`acids-rave`) has broken/pinned dependencies that conflict with modern Python packages. Since scropipe calls RAVE via CLI subprocess, install it in a **separate virtual environment** and then patch the incompatibilities.

#### Linux/macOS

```bash
# Create a separate venv for RAVE
python -m venv ~/.rave-venv

# Install PyTorch first, then RAVE
~/.rave-venv/bin/pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
~/.rave-venv/bin/pip install acids-rave

# Force-reinstall torch to fix version mismatches from RAVE's deps
~/.rave-venv/bin/pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Pin scipy to a version that still has scipy.signal.kaiser
~/.rave-venv/bin/pip install "scipy==1.11.4"

# Add RAVE to your PATH (add to .bashrc/.zshrc for persistence)
export PATH="$HOME/.rave-venv/bin:$PATH"
```

#### Windows (PowerShell)

```powershell
# Create a separate venv for RAVE (MUST be Python 3.12)
python -m venv $HOME\.rave-venv

# Install PyTorch first, then RAVE
& $HOME\.rave-venv\Scripts\pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
& $HOME\.rave-venv\Scripts\pip install acids-rave

# Force-reinstall torch to fix version mismatches from RAVE's deps
& $HOME\.rave-venv\Scripts\pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Pin scipy to a version that still has scipy.signal.kaiser
& $HOME\.rave-venv\Scripts\pip install "scipy==1.11.4"
```

#### Patch RAVE's numpy incompatibility

RAVE's code uses `np.float`, which was removed in numpy 1.24. You need to patch one file in the RAVE venv:

**Linux/macOS:**
```bash
sed -i 's/np\.float)/np.float64)/g' ~/.rave-venv/lib/python3.12/site-packages/rave/dataset.py
```

**Windows (PowerShell):**
```powershell
(Get-Content $HOME\.rave-venv\Lib\site-packages\rave\dataset.py) -replace 'np\.float\)', 'np.float64)' | Set-Content $HOME\.rave-venv\Lib\site-packages\rave\dataset.py
```

Or manually edit `~/.rave-venv/.../rave/dataset.py` line 64 and change `np.float` to `np.float64`.

#### Tell scropipe where RAVE is

Scropipe will find the `rave` command if it's in your PATH, or you can set `RAVE_PATH`:

```bash
# Linux/macOS
export RAVE_PATH="$HOME/.rave-venv/bin/rave"
```

```powershell
# Windows (current session)
$env:RAVE_PATH = "$HOME\.rave-venv\Scripts\rave.exe"

# Windows (permanent)
[Environment]::SetEnvironmentVariable("RAVE_PATH", "$HOME\.rave-venv\Scripts\rave.exe", "User")
```

#### Verify

```bash
scropipe tools    # Should show rave as detected
rave train --help # Should print flags without errors
```

#### Why all this?

RAVE was built against older versions of numpy, scipy, and PyTorch. Installing it with `pip install acids-rave` pulls in dependency versions that conflict with each other and with modern Python. The separate venv + force-reinstall + patch approach is the least painful way to get it working. Blame Python packaging, not yourself.

## Backwards Compatibility

The `scrumpler` and `scronchler` CLIs are still available as entry points:

```bash
# These still work
scrumpler input.wav -o ./samples --mode transient
scrumpler-batch --preset drums
scronchler preprocess -i ./samples -o ./specs
scronchler train -i ./specs -o ./model
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
| `-m, --model` | Model type: `vae` (default) or `rave` |
| `--rave-config` | RAVE config: `v2`, `v2_small`, `discrete` (default: v2) |
| `-a, --augment` | Augment samples during preprocessing (VAE only) |
| `--max-duration` | Maximum sample duration in seconds (default: 2.0) |
| `-e, --epochs` | Training epochs (default: 100) |
| `-c, --count` | Number of samples to generate (default: 10) |
| `--train-vocoder` | Train HiFi-GAN vocoder (VAE only) |
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

Show status of external tools (currently only RAVE).

```bash
scropipe tools
```

Note: scrumpler and scronchler are now built into scropipe and no longer require external binaries.

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

1. **Split** - Chop audio files (transient detection, grid, or texture gating)
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

### RAVE for Melodic Content

For **piano, melodic, or harmonic content**, use RAVE instead of VAE:

```bash
# Use RAVE for high-quality melodic synthesis (requires longer chunks)
scropipe run -i piano.wav --synthesize --model rave --count 20 --chunk-length 6

# Or with the synthesize command
scropipe synthesize ./samples --model rave --rave-config v2
```

| Model | Best For | Training Time | Min Chunk Length |
|-------|----------|---------------|------------------|
| `vae` (default) | Drums, percussion, textures | Fast | 0.5s |
| `rave` | Piano, melodic, harmonic | Several hours | 6s |

RAVE produces much higher quality output for tonal content because it works directly on audio waveforms with a neural vocoder.

#### RAVE Training Workflow

RAVE trains for up to 6 million steps by default, which can take many hours or days. The pipeline runs these stages:

1. **RAVE Preprocess** - converts audio to training database
2. **RAVE Train** - trains the model (long-running)
3. **RAVE Export** - converts checkpoint to usable model
4. **RAVE Generate** - creates your output samples

**Stopping early:** You can stop training with `Ctrl+C` when the loss stabilizes. Early results (50-100 epochs) will be rough but recognizable. Several hundred epochs produces decent quality. Checkpoints are saved every 500 steps (~30 epochs).

If you stop early, the pipeline won't complete the export/generate stages. Run them manually:

```bash
# Export the trained model (finds the latest checkpoint automatically)
rave export --run scropipe-output/03-rave-model/model/

# Find the exported model
ls scropipe-output/03-rave-model/model/**/checkpoints/*.ts
```

**Note:** The `rave generate` CLI has a bug, so use Python directly for generation:

```python
import torch
import torchaudio
from pathlib import Path

model = torch.jit.load("path/to/model.ts")
model = model.to("cuda:0")  # or "cpu"

x, sr = torchaudio.load("input.wav")
if sr != model.sr:
    x = torchaudio.functional.resample(x, sr, model.sr)

with torch.no_grad():
    out = model.forward(x[None].to("cuda:0"))

torchaudio.save("output.wav", out[0].cpu(), sample_rate=model.sr)
```

Or let scropipe handle it - the pipeline uses the Python API directly to avoid the CLI bug.

**Resuming training:** RAVE saves checkpoints periodically. Resume from where you left off:

```bash
# First, find your checkpoints
ls scropipe-output/03-rave-model/model_*/version_*/checkpoints/*.ckpt

# Linux - use the full path to the checkpoint file
rave train --config v2 \
    --db_path scropipe-output/02-rave-data \
    --name model \
    --ckpt scropipe-output/03-rave-model/model_XXXX/version_0/checkpoints/epoch-epoch=0100.ckpt \
    --gpu 0 --workers 0
```

```powershell
# Windows (PowerShell) - find checkpoints first
Get-ChildItem -Recurse scropipe-output/03-rave-model -Filter "*.ckpt"

# Resume with full path to checkpoint file
rave train --config v2 `
    --db_path scropipe-output/02-rave-data `
    --name model `
    --ckpt scropipe-output/03-rave-model/model_XXXX/version_0/checkpoints/epoch-epoch=0100.ckpt `
    --gpu 0
```

**Note:** The `--ckpt` flag requires the full path to a `.ckpt` file, not just the directory.

**Testing during training:** You can export and test a checkpoint while training continues (in another terminal):

```bash
# Export current checkpoint
rave export --run scropipe-output/03-rave-model/model/

# The .ts file will be in the checkpoints folder
ls scropipe-output/03-rave-model/model/**/checkpoints/*.ts

# Generate a test sample using Python (see above) or let training finish
# and run the full pipeline which handles generation automatically
```

#### Windows / NVIDIA GPU Notes

RAVE works on Windows with NVIDIA GPUs, but getting there requires patience:

- **Python 3.12 only** — PyTorch has no wheels for 3.13+ on Windows
- Use `--chunk-length 6` or longer (RAVE's preprocessing has a bug with short audio)
- Training should work with default `--workers 8` (reduce if you hit memory issues)
- You will see warnings about Tensor Cores and `weight_norm` deprecation — these are harmless

```powershell
# Example on Windows
scropipe run -i .\audio\piano.wav --synthesize --model rave --chunk-length 6
```

#### AMD GPU (ROCm) Notes

RAVE works with AMD GPUs via ROCm (Linux only), but requires some adjustments:

- Use `--chunk-length 6` or longer (RAVE's preprocessing has a bug with short audio)
- Training uses `--workers 0` to avoid multiprocessing segfaults
- The `amdgpu.ids: No such file or directory` warning is harmless
- `MIOpen` workspace warnings are normal (ROCm kernel autotuning)

### Neural Vocoder (VAE only)

For VAE with cleaner audio, train a HiFi-GAN vocoder:

```bash
scropipe run -i drums.wav --synthesize --train-vocoder --vocoder-epochs 50
```

| Dataset Size | Recommendation |
|--------------|----------------|
| < 100 samples | Skip vocoder, use Griffin-Lim |
| 100-300 samples | Vocoder might help, worth trying |
| 300+ samples | Vocoder likely to outperform Griffin-Lim |

The vocoder helps most with tonal/melodic content. For percussive/noise sounds, Griffin-Lim often works fine.

## Package Structure

```
scropipe/
├── scropipe/
│   ├── __init__.py
│   ├── cli.py                    # Main CLI
│   ├── pipeline.py               # Pipeline orchestration
│   ├── splitter/                 # Audio splitting (built-in)
│   │   ├── processor.py          # SampleProcessor class
│   │   └── presets.py            # Batch presets
│   ├── synth/                    # Neural synthesis (built-in)
│   │   ├── audio_utils.py        # Audio preprocessing
│   │   ├── model.py              # VAE model
│   │   ├── vocoder.py            # HiFi-GAN vocoder
│   │   └── ...
│   ├── stages/                   # Pipeline stages
│   └── utils/
│       └── discovery.py          # RAVE discovery only
├── pyproject.toml
└── flake.nix
```


## License

MIT
