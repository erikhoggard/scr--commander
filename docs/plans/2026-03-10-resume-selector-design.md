# Resume Training Run Selector Design

**Problem:** "Resume Training" blindly picks the most recent paused run with no user choice.

**Solution:** Add a Select dropdown listing paused runs. User picks one, then clicks Resume.

**Select format:** `"model_name (architecture, paused)"` e.g. `"whitpiano_ultramodel (v2, paused)"`

**Edge cases:**
- No paused runs → Select empty, Resume shows "No paused training runs found"
- No selection → "Please select a run to resume"

**Changes:**
- `TrainConfigPanel.compose()` — add Label + Select for resumable runs above Resume button
- `TrainConfigPanel` — add `_populate_resumable_runs()`, call from `on_mount` and tab refresh
- `TrainTab._resume_training()` — read selected value instead of `paused[-1]`
