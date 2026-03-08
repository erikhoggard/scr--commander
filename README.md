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
| `--no-train` | Skip training, go straight to export + generate from existing checkpoint |
| `--seed-dir` | Directory of audio to use as generation input (default: training pool) |

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

### VAE Pipeline (default)

1. **Split** - Chop audio files (transient detection, grid, or texture gating)
2. **Collect** - Pool samples from split outputs and included directories
3. **Preprocess** - Convert samples to mel-spectrograms
4. **Train** - Train a VAE model
5. **Train Vocoder** (optional) - Train HiFi-GAN for higher quality (`--train-vocoder`)
6. **Generate** - Synthesize new samples

### RAVE Pipeline (`--model rave`)

1. **Split** - Same as VAE
2. **Collect** - Same as VAE
3. **RAVE Preprocess** - Convert audio to RAVE training database
4. **RAVE Train** - Train the model (hours/days)
5. **RAVE Export** - Convert checkpoint to `.ts` model
6. **RAVE Generate** - Feed seed audio through model to produce output

## Choosing a Model

| Model | Best For | Training Time | Min Chunk Length |
|-------|----------|---------------|------------------|
| `vae` (default) | Drums, percussion, textures | Minutes | 0.5s |
| `rave` | Piano, melodic, harmonic | Hours | 6s |

```bash
# VAE (fast, good for percussion)
scropipe run -i drums.wav --synthesize --epochs 200 --count 50

# RAVE (slow, better for melodic content)
scropipe run -i piano.wav --synthesize --model rave --count 20 --chunk-length 6
```

## Typical Workflow

### 1. Slice audio

```bash
scropipe run -i piano.wav --chunk-length 6
```

### 2. Slice + train

```bash
scropipe run -i piano.wav --synthesize --model rave --chunk-length 6
```

Training takes hours. Ctrl+C anytime — checkpoints are saved every 500 steps.

### 3. Resume training

```bash
scropipe train -o ./scropipe-output
```

Finds the latest checkpoint and continues training.

### 4. Generate audio

```bash
scropipe generate -o ./scropipe-output --count 20

# Use different seed audio (feed new files through the trained model)
scropipe generate -o ./scropipe-output --count 20 --seed-dir ./other-audio/
```

Exports the model if needed, then generates samples. Run it as many times as you want with different counts or seed audio.

### RAVE Training Tips

- Ctrl+C when loss stabilizes. Resume continues from `last.ckpt`.
- Early (50-100 epochs): rough. Several hundred: decent.
- Use `--chunk-length 6` or longer (RAVE has a bug with short audio).

### VAE Tips

- **More data helps**: split multiple files, include sample libraries with `-I`, use `--augment`
- **50+ samples** for reasonable results, 200+ for best results
- **Vocoder** (`--train-vocoder`): helps with tonal content, not needed for percussion

## GPU Notes

**Windows/NVIDIA:** Python 3.12 only (no PyTorch wheels for 3.13+). Tensor Core and `weight_norm` warnings are harmless.

**AMD/ROCm (Linux only):** Uses `--workers 0` to avoid segfaults. `amdgpu.ids` and `MIOpen` warnings are harmless.

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
