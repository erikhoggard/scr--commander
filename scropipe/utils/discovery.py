"""Tool discovery utilities for finding external binaries.

Only RAVE is an external tool now - scrumpler and scronchler are built-in.
"""

import os
import shutil
from pathlib import Path
from typing import Optional


class ToolNotFoundError(Exception):
    """Raised when a required tool cannot be found."""

    def __init__(self, tool_name: str, env_var: str):
        self.tool_name = tool_name
        self.env_var = env_var
        super().__init__(
            f"Could not find '{tool_name}'. "
            f"Either set {env_var} environment variable or ensure it's in PATH."
        )


# Environment variable names for tool paths
# Only rave is an external tool now
TOOL_ENV_VARS = {
    "rave": "RAVE_PATH",
}


def find_tool(name: str) -> Path:
    """Find a tool binary by name.

    Search order:
    1. Environment variable (e.g., RAVE_PATH)
    2. PATH lookup

    Args:
        name: Tool name (e.g., "rave")

    Returns:
        Path to the tool binary.

    Raises:
        ToolNotFoundError: If the tool cannot be found.
    """
    env_var = TOOL_ENV_VARS.get(name, f"{name.upper().replace('-', '_')}_PATH")

    # Check environment variable first
    env_path = os.environ.get(env_var)
    if env_path:
        path = Path(env_path)
        if path.is_file() and os.access(path, os.X_OK):
            return path
        # Maybe it's a directory containing the binary
        if path.is_dir():
            binary = path / name
            if binary.is_file() and os.access(binary, os.X_OK):
                return binary

    # Fall back to PATH lookup
    which_result = shutil.which(name)
    if which_result:
        return Path(which_result)

    raise ToolNotFoundError(name, env_var)


def find_all_tools() -> dict[str, Optional[Path]]:
    """Find all external tools and return their paths.

    Returns:
        Dict mapping tool names to their paths (or None if not found).
    """
    tools = {}
    # Only rave is an external tool now
    try:
        tools["rave"] = find_tool("rave")
    except ToolNotFoundError:
        tools["rave"] = None
    return tools


def check_rave_available() -> bool:
    """Check if RAVE is available.

    Returns:
        True if rave can be found.
    """
    try:
        find_tool("rave")
        return True
    except ToolNotFoundError:
        return False
