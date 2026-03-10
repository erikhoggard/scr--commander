# Path Input UX Improvements

## Problem

Two friction points in the TUI:

1. **Tab completion conflict**: Users instinctively press Tab to complete paths (terminal muscle memory), but Tab moves focus to the next widget. The existing PathSuggester uses right-arrow to accept, which fights habits.
2. **File browser can't reach other drives**: BrowseModal starts at `Path.home()` with no way to navigate above it. On Windows with multiple drives, users are stuck.

## Design

### 1. PathInput Widget

Custom `Input` subclass that intercepts Tab for path cycling.

**Behavior:**

- On Tab: gather all filesystem entries matching the current partial path, cycle to the next match. If only one match, complete it and append `/` if it's a directory.
- On Shift+Tab: normal focus behavior (move to previous widget).
- Tracks cycle state: resets when the user types anything new.
- Replace all path-related `Input()` usages with `PathInput()` across split_tab, pool_tab, generate_tab, and app.py (setup modal).

PathSuggester stays as-is for right-arrow ghost suggestions. Tab cycling and right-arrow acceptance are complementary.

### 2. BrowseModal Root Navigation

**Changes:**

- Set `DirectoryTree` root to the drive root of the start path (e.g., `C:\` if start path is `C:\Users\micro\Music`), or `/` on Unix.
- Pre-expand the tree nodes down to the start path so the user sees their context immediately.
- Add a `Select` dropdown at the top listing available drive letters (Windows only, hidden on Unix).
- When the user picks a different drive, replace the DirectoryTree with one rooted at that drive.

**Drive detection:** Iterate `A:` through `Z:` checking `Path("X:/").exists()`.

### 3. Testing

- Unit test PathInput tab cycling logic with mocked filesystem.
- Unit test drive detection.
- Unit test BrowseModal initializes with correct root and expands to start path.

### 4. Files Changed

- **New:** `scropipe/tui/path_input.py` — PathInput widget
- **Modified:** `scropipe/tui/browse_modal.py` — root navigation + drive selector
- **Modified:** `scropipe/tui/split_tab.py` — swap Input to PathInput
- **Modified:** `scropipe/tui/pool_tab.py` — swap Input to PathInput
- **Modified:** `scropipe/tui/generate_tab.py` — swap Input to PathInput
- **Modified:** `scropipe/tui/app.py` — swap Input to PathInput in setup modal
- **New:** `tests/test_path_input.py`
- **New:** `tests/test_browse_modal.py`
