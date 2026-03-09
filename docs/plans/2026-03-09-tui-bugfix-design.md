# TUI Bugfix & Flow Repair Design

## Problems

1. **Browse buttons on Split tab do nothing** — `on_button_pressed` has no handler for `split-browse-source` or `split-browse-output`
2. **"Split & Add to Pool" is stubbed** — runs the split but never calls `pool_manager.add_files()`; no pool selector exists on Split tab
3. **Train tab shows 0 samples** — pool dropdown populated on mount only, never refreshed; shows stale `sample_count` from initial load
4. **Pool tab Add Files/Directory modals** — use text-input `FileInputModal`; should also offer a `DirectoryTree` visual browser

## Solutions

### 1. Reusable BrowseModal

New widget: `scropipe/tui/browse_modal.py`

- Textual `ModalScreen` containing a `DirectoryTree` widget
- Constructor params: `title: str`, `select_type: "file" | "directory"`, `start_path: Path`
- For file mode: user navigates tree and selects a file; modal returns the file path
- For directory mode: user navigates tree and selects/confirms a directory; modal returns directory path
- OK and Cancel buttons
- Returns selected path via `self.dismiss(result)` callback pattern

Used by: Split tab (Browse buttons), Pool tab (Add Files, Add Directory), Generate tab (Browse buttons).

### 2. Split tab Browse buttons

Wire `split-browse-source` to open `BrowseModal(select_type="file")`. On dismiss, populate `#split-source-input` with the returned path.

Wire `split-browse-output` to open `BrowseModal(select_type="directory")`. On dismiss, populate `#split-output-input`.

### 3. Split & Add to Pool

Add a pool `Select` dropdown to the Split tab (above the action buttons). Populate it from `PoolManager.list_pools()` on compose and refresh on tab focus.

In `_do_split()`, when `add_to_pool=True` and split succeeds:
- Get selected pool name from the dropdown
- Collect output WAV paths from `result.details` (or scan `output_dir`)
- Call `pool_manager.add_files(pool_name, wav_paths)` with source type "split"

### 4. Train tab stale pool data

Call `_populate_pools()` when the Train tab receives focus (Textual `on_show` or watch for `TabbedContent.Active` message). This re-reads from disk and updates the Select widget options.

Preserve the user's current selection if the pool still exists after refresh.

### 5. Pool tab Browse modals

Replace or augment `FileInputModal` usage:
- "Add Files" button: opens `BrowseModal(select_type="file")`, allows selecting one or more WAV files
- "Add Directory" button: opens `BrowseModal(select_type="directory")`, returns directory path
- Keep `FileInputModal` as a fallback for users who prefer typing paths directly, or remove it if the BrowseModal covers both use cases

## Scope

- No changes to `PoolManager`, `SplitStage`, or backend logic (they already work)
- All fixes are in the TUI layer (`scropipe/tui/`)
- New file: `browse_modal.py`
- Modified files: `split_tab.py`, `pool_tab.py`, `train_tab.py`, `generate_tab.py`
