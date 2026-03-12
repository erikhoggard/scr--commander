"""Filesystem path suggester for Textual Input widgets."""

from __future__ import annotations

from pathlib import Path

from textual.suggester import Suggester


class PathSuggester(Suggester):
    """Suggests filesystem paths as the user types.

    Completes directory and file names based on the current input value.
    Press right-arrow to accept a suggestion.
    """

    def __init__(self, directories_only: bool = False) -> None:
        super().__init__(use_cache=False, case_sensitive=False)
        self.directories_only = directories_only

    async def get_suggestion(self, value: str) -> str | None:
        if not value:
            return None

        path = Path(value)

        # Expand ~ to home directory
        try:
            expanded = path.expanduser()
        except (RuntimeError, ValueError):
            expanded = path

        # If the path as-typed is an existing directory and ends with a separator,
        # suggest the first child entry.
        if expanded.is_dir() and value.endswith(("/", "\\")):
            return self._first_child(expanded, value)

        # Otherwise, treat the last component as a partial name to complete.
        parent = expanded.parent
        if not parent.is_dir():
            return None

        partial = expanded.name.lower()
        return self._complete_partial(parent, partial, value)

    def _first_child(self, directory: Path, original: str) -> str | None:
        """Suggest the first child of a directory."""
        try:
            children = sorted(directory.iterdir(), key=lambda p: p.name.lower())
            if self.directories_only:
                children = [c for c in children if c.is_dir()]
            if children:
                return original + children[0].name
        except PermissionError:
            pass
        return None

    def _complete_partial(
        self, parent: Path, partial: str, original: str
    ) -> str | None:
        """Complete a partial filename within a directory."""
        try:
            matches = sorted(
                (
                    entry
                    for entry in parent.iterdir()
                    if entry.name.lower().startswith(partial)
                    and (not self.directories_only or entry.is_dir())
                ),
                key=lambda p: p.name.lower(),
            )
        except PermissionError:
            return None

        if not matches:
            return None

        # Use the first match; preserve the prefix the user typed
        best = matches[0]
        # Replace only the last component
        prefix = original[: len(original) - len(partial)]
        return prefix + best.name
