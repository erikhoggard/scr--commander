# Scropipe TUI Design

## Overview

Replace the flag-heavy CLI with a Textual-based TUI as the primary interface. The existing CLI commands are preserved for scripting. The TUI provides a tab-based interface covering three independent phases: Split, Pool, and Train/Generate.

## Decisions

- **Framework:** Textual (by the Rich team, already a dependency)
- **CLI coexistence:** `scropipe` (no args) launches TUI; existing CLI commands remain
- **Navigation:** Tab-based with four tabs (Split, Pool, Train, Generate)
- **Architecture:** Single Textual App with TabbedContent, shared reactive AppState
- **Storage:** User-chosen directories for models and pools, configured on first run
- **Presets:** Load existing presets as templates, save current config as new presets
- **Training UI:** Live dashboard with loss curve, step count, delta, stop controls
- **Generation output:** User chooses input and output directories each time

## Entry Points

- `scropipe` (no args) -> launches TUI
- `scropipe split`, `scropipe collect`, `scropipe train`, `scropipe generate`, etc. -> existing CLI preserved
- Both share the same underlying stage/pipeline code in `scropipe/stages/`

## Configuration

Config file: `~/.config/scropipe/config.toml`

```toml
models_dir = "/path/to/models"
pools_dir = "/path/to/pools"
presets_dir = "/path/to/custom/presets"  # optional override
```

On first launch, a setup modal prompts the user to choose these directories.

## File Layout

```
scropipe/
├── tui/
│   ├── __init__.py
│   ├── app.py           # ScropipeApp, first-run setup modal
│   ├── split_tab.py     # Split phase UI
│   ├── pool_tab.py      # Pool management UI
│   ├── train_tab.py     # Training config + live dashboard
│   ├── generate_tab.py  # Generation UI
│   ├── state.py         # Shared reactive AppState
│   ├── widgets.py       # Reusable custom widgets (file picker, etc.)
│   └── styles.tcss      # Textual CSS styling
├── config.py            # Config loading/saving
```

## Tab Designs

### Split Tab

Lets users load an audio file, choose a splitting mode, configure parameters, and split.

```
┌─ Split ──────────────────────────────────────────┐
│                                                   │
│  Source File                                      │
│  [/home/erik/audio/breakbeat.wav       ] [Browse] │
│                                                   │
│  Preset: [ None ▾ ]  [Save Current...]            │
│                                                   │
│  Splitting Mode                                   │
│  (●) Transient   ( ) Grid   ( ) Texture           │
│                                                   │
│  ── Transient Settings ──────────────────────     │
│  Sensitivity (delta):  [0.07    ]                 │
│  Min length (s):       [0.05    ]                 │
│  Max length (s):       [10.0    ]                 │
│                                                   │
│  Output Directory                                 │
│  [/home/erik/audio/breakbeat_split     ] [Browse] │
│                                                   │
│  [ Split ]    [ Split & Add to Pool ▶ ]           │
└───────────────────────────────────────────────────┘
```

- Selecting a mode shows the relevant parameter controls
- Grid mode shows: chunk length OR bpm+bars inputs
- Texture mode shows: min/max duration, RMS threshold, stability threshold
- "Split & Add to Pool" splits then switches to Pool tab with output added
- Loading a preset populates all fields
- File browser uses Textual's DirectoryTree widget

### Pool Tab

Manages named collections of samples aggregated from multiple sources.

```
┌─ Pool ───────────────────────────────────────────┐
│                                                   │
│  ┌─ Pools ─────────────┐  ┌─ Pool: drum-hits ──┐ │
│  │                      │  │                     │ │
│  │  ▶ drum-hits (47)    │  │  47 samples         │ │
│  │    ambient-v1 (23)   │  │  Total: 2m 34s      │ │
│  │    textures (12)     │  │                     │ │
│  │                      │  │  ── Sources ──────  │ │
│  │                      │  │  breakbeat_split/    │ │
│  │                      │  │    24 samples        │ │
│  │                      │  │  ~/lib/kicks/        │ │
│  │                      │  │    15 samples        │ │
│  │                      │  │  snare-01.wav        │ │
│  │                      │  │  snare-02.wav        │ │
│  │                      │  │  ...                 │ │
│  │                      │  │                     │ │
│  │  [+ New Pool]        │  │  [+ Add Files]      │ │
│  │                      │  │  [+ Add Directory]  │ │
│  │                      │  │  [Delete Pool]      │ │
│  └──────────────────────┘  │  [Train ▶]          │ │
│                            └─────────────────────┘ │
└───────────────────────────────────────────────────┘
```

- Left panel: list of all pools with sample counts
- Right panel: details of selected pool, grouped by source
- A pool can contain samples from many different sources (splits, directories, individual files)
- pool.json tracks each source with provenance (path, date added, sample count)
- Samples are copied into the pool's samples/ directory
- "Add Files" imports individual WAV files
- "Add Directory" imports all WAVs from a directory
- "Train ▶" switches to Train tab with this pool pre-selected
- Pools persist in the configured pools_dir

### Train Tab

Two modes: configuration (before training) and live dashboard (during training).

**Configuration:**

```
┌─ Train ──────────────────────────────────────────┐
│                                                   │
│  Pool:   [ drum-hits (47 samples) ▾ ]             │
│  Model name: [drums-v1              ]             │
│                                                   │
│  Preset: [ None ▾ ]  [Save Current...]            │
│                                                   │
│  ── Stop Conditions ─────────────────────────     │
│  ( ) Max steps:    [10000   ]                     │
│  ( ) Delta target: [0.001   ]                     │
│  (●) Manual (stop when ready)                     │
│                                                   │
│  ── RAVE Config ─────────────────────────────     │
│  Architecture:   [ v2 ▾ ]                         │
│  Checkpoint every: [500] steps                    │
│  Resume from:    [ None ▾ ]                       │
│                                                   │
│  GPU: AMD ROCm (gfx1100)                         │
│                                                   │
│  [ Start Training ]                               │
└───────────────────────────────────────────────────┘
```

**Live dashboard (during training):**

```
┌─ Train ──────────────────────────────────────────┐
│                                                   │
│  Training: drums-v1                               │
│  Pool: drum-hits (47)    Arch: RAVE v2            │
│                                                   │
│  Step: 3,420 / manual     Loss: 0.0231            │
│  Δ: 0.0003                                        │
│                                                   │
│  Loss:                                            │
│  0.50│▇                                           │
│      │▆▄                                          │
│      │ ▃▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁                      │
│  0.02│─────────────────────────                   │
│      └──────────────────────────                  │
│       0              3,420                        │
│                                                   │
│  Elapsed: 1h 23m    Last checkpoint: step 3,000   │
│                                                   │
│  [ Stop & Save ]    [ Stop & Discard ]            │
└───────────────────────────────────────────────────┘
```

- Stop conditions are radio buttons (max steps, delta target, or manual)
- "Resume from" dropdown lists existing checkpoints
- Dashboard updates live via background worker thread
- "Stop & Save" exports model to models library
- "Stop & Discard" kills training without saving
- Loss curve via Sparkline or custom text-based plot widget
- If delta target is set, training auto-stops and saves when reached

### Generate Tab

```
┌─ Generate ───────────────────────────────────────┐
│                                                   │
│  ── Model ───────────────────────────────────     │
│  [ drums-v1 (RAVE v2, trained on drum-hits) ▾ ]   │
│                                                   │
│  ── Input ───────────────────────────────────     │
│  Source samples:                                  │
│  [/home/erik/audio/new-drums/          ] [Browse] │
│  Found: 12 files                                  │
│                                                   │
│  ── Output ──────────────────────────────────     │
│  Output directory:                                │
│  [/home/erik/audio/generated/          ] [Browse] │
│                                                   │
│  ── Models Library ──────────────────────────     │
│  ┌────────────────┬──────────┬───────────────┐    │
│  │ Name           │ Arch     │ Pool          │    │
│  ├────────────────┼──────────┼───────────────┤    │
│  │ ▶ drums-v1     │ RAVE v2  │ drum-hits     │    │
│  │   ambient-v2   │ RAVE v2  │ ambient-v1    │    │
│  │   textures-v1  │ VAE      │ textures      │    │
│  └────────────────┴──────────┴───────────────┘    │
│  [Delete Model]                                   │
│                                                   │
│  [ Generate ]                                     │
└───────────────────────────────────────────────────┘
```

- Model dropdown lists all trained models
- Models table shows full library with metadata
- Selecting a model in the table also selects it in the dropdown
- User chooses both input and output directories each time
- "Generate" processes all input samples, shows progress bar, writes to output dir
- "Delete Model" removes from library with confirmation dialog

## Status Bar

Always visible at bottom:

```
Pool: drum-hits (47) │ Model: drums-v1 │ GPU: AMD ROCm │ Ctrl+Q: Quit
```

Shows the currently active pool and model context. Updates as the user works.

## First-Run Setup

Modal dialog on first launch (no config file detected):

- Prompts for models directory path
- Prompts for pools directory path
- Checks RAVE availability and GPU status
- Saves to `~/.config/scropipe/config.toml`

## Settings

Accessible via Ctrl+S or gear icon in header:

- Change models/pools directories
- Manage presets
- RAVE path configuration
- GPU selection (if multiple available)

## Keyboard Shortcuts

- `1-4` or `Ctrl+1-4`: Switch tabs
- `Ctrl+Q`: Quit
- `Ctrl+S`: Settings
- `Tab/Shift+Tab`: Navigate within a tab

## Dependencies

New dependency: `textual>=0.50.0` (already compatible since Rich is a dependency)

## Shared Code

The TUI calls the same underlying code as the CLI:

- `scropipe/stages/` - all pipeline stages unchanged
- `scropipe/splitter/` - audio splitting logic unchanged
- `scropipe/synth/` - synthesis logic unchanged
- `scropipe/pipeline.py` - pipeline orchestration unchanged

The TUI is purely a presentation layer on top of the existing stage/pipeline code.
