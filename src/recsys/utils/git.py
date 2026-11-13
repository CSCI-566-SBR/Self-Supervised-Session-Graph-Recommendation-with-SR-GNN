"""Version control and Git environment utilities."""

from __future__ import annotations

import subprocess


def get_git_commit() -> str | None:
    """Retrieve the current HEAD git commit hash if running in a git repository."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
