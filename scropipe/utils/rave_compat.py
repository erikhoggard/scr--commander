"""Wrap RAVE CLI commands for subprocess execution.

Ensures scipy compatibility patches are applied before RAVE imports,
working around scipy >= 1.14 removing ``scipy.signal.kaiser``.
"""

from __future__ import annotations

import sys

# Single-line bootstrap that patches scipy then runs RAVE's CLI.
# Uses exec() to handle the try/except block within -c.
_BOOTSTRAP = (
    "import sys;"
    "exec('try:\\n from scipy.signal import kaiser\\n"
    "except ImportError:\\n from scipy.signal.windows import kaiser\\n"
    " import scipy.signal as _s; _s.kaiser = kaiser');"
    "sys.argv = sys.argv[1:];"
    "from scripts.main_cli import main; main()"
)


def wrap_rave_cmd(original_cmd: list[str]) -> list[str]:
    """Wrap a rave CLI command so scipy is patched before rave imports.

    Replaces e.g. ``['rave', 'train', ...]`` with
    ``['python', '-c', '<bootstrap>', 'rave', 'train', ...]``.
    """
    rave_args = original_cmd[1:]
    return [sys.executable, "-c", _BOOTSTRAP, "rave"] + rave_args
