# Training Cancel/Resume Design

## Problem

Training RAVE models can take hours. Users need to be able to interrupt training and resume it later without losing progress.

## Scope

- RAVE training only (VAE is being phased out)
- TUI-managed subprocess (training stops when TUI exits)
- Graceful shutdown preserving checkpoints

## Architecture

### Cancel Flow

1. User clicks **"Stop Training"** (single button, replaces current Save/Discard pair)
2. TUI sends **SIGINT** to the `rave train` subprocess
3. PyTorch Lightning intercepts SIGINT and saves a final checkpoint before exiting
4. Dashboard shows "Stopping..." state while waiting for process exit
5. **Timeout**: if process doesn't exit within 10s after SIGINT, escalate to SIGTERM
6. Checkpoints are always preserved (no discard option)
7. Switch back to config panel with status: "Training paused at step N"

### Resume Flow

**Sidecar metadata**: When training starts, write `training_run.json` to the model's directory:

```json
{
  "model_name": "my-model",
  "pool_name": "drums",
  "architecture": "v2",
  "output_dir": "/path/to/output",
  "status": "training|paused|completed",
  "started": "2026-03-08T12:00:00"
}
```

**Status transitions**:
- `training` -> set on start
- `paused` -> set on cancel (SIGINT) or detected on TUI relaunch (process no longer running)
- `completed` -> set when `rave train` exits with code 0

**Resume UI**:
1. "Resume Training" button alongside "Start Training" in config panel
2. On click, show list of models with `status: paused` and checkpoint files present
3. Display: model name, pool, architecture, last checkpoint step
4. On selection, launch `rave train --ckpt <checkpoint_dir>` with original output_dir as cwd

**RAVE resume mechanism**: The `--ckpt` flag accepts a directory path. RAVE finds the most recent `.ckpt` file and restores full training state (model weights, optimizer, schedulers, step counter). Config is auto-loaded from `config.gin` near the checkpoint.

### Training Worker (Replacing Stub)

The current `_training_worker()` in `train_tab.py` is a stub that simulates training. Replace with:

1. **Preprocessing**: Run `rave preprocess` if no preprocessed data exists for the pool. Show "Preprocessing..." status. Uses existing `RavePreprocessStage`.

2. **Training subprocess**: Launch `rave train` via `subprocess.Popen` with:
   - `stdout=PIPE, stderr=STDOUT` for metric capture
   - `cwd=output_dir` (RAVE writes checkpoints relative to cwd)

3. **Metric parsing**: Worker thread reads stdout line-by-line, parses Lightning's progress output via regex to extract step number, loss values, and checkpoint events. Parser should be tolerant of format variations.

4. **UI updates**: Call `dashboard.update_metrics()`, `dashboard.update_timing()`, `dashboard.update_checkpoint()` via `app.call_from_thread()`.

5. **On cancel**: SIGINT -> wait 10s -> SIGTERM fallback. Update sidecar to `paused`.

6. **On completion**: Process exits code 0. Update sidecar to `completed`.

### Error Handling

| Scenario | Behavior |
|---|---|
| TUI exits while training | SIGINT to subprocess; sidecar stays `training`; detected as `paused` on next launch |
| RAVE fails to start | Show error in status bar, return to config view |
| Preprocessing fails | Show error, return to config |
| Resume with bad checkpoint | RAVE fails to start, show error |
| Disk full | RAVE crashes, last good checkpoint preserved, user can resume after freeing space |
| No paused runs | "Resume Training" button disabled |

### Files Modified

- `scropipe/tui/train_tab.py` — Replace stub worker, update cancel UI, add resume UI
- `scropipe/stages/rave.py` — May need minor adjustments for Popen (currently uses `subprocess.run`)

### Files Added

- None expected (sidecar JSON files are runtime data, not source files)

## Key Decisions

1. **SIGINT not SIGTERM** — Lightning saves a checkpoint on SIGINT
2. **Always keep checkpoints** — No discard option; cleanup via model management
3. **JSON sidecar for state** — Simple, human-readable, decoupled from RAVE internals
4. **Managed subprocess** — Training stops when TUI exits (simplicity over detached processes)
