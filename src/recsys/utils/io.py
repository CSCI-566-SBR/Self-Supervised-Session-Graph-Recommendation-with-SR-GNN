"""Serialization, TOML/pickle file reading, and directory management utilities."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any


def load_pickle(path: str | Path) -> Any:
    """Read an existing pickle artifact."""
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def save_pickle(value: Any, path: str | Path) -> None:
    """Atomically write a pickle artifact, creating parents if missing."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_toml(path: str | Path) -> dict[str, Any]:
    """Parse a TOML file with tomllib or lightweight fallback."""
    path = Path(path)
    try:
        import tomllib
        with path.open("rb") as f:
            return tomllib.load(f)
    except ModuleNotFoundError:
        try:
            import tomli
            with path.open("rb") as f:
                return tomli.load(f)
        except ModuleNotFoundError:
            pass

    # Simple key-value / section parser fallback
    data: dict[str, Any] = {}
    current: dict[str, Any] = data
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip()
            current = data.setdefault(section, {})
        elif "=" in line:
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.split("#")[0].strip()
            if val.startswith('"') and val.endswith('"'):
                current[key] = val[1:-1]
            elif val.startswith("'") and val.endswith("'"):
                current[key] = val[1:-1]
            elif val.isdigit():
                current[key] = int(val)
            elif val.lower() == "true":
                current[key] = True
            elif val.lower() == "false":
                current[key] = False
            else:
                try:
                    current[key] = float(val)
                except ValueError:
                    current[key] = val
    return data


def prepare_output_dir(output_dir: Path) -> None:
    """Create the output directory defensively, raising FileExistsError if non-empty."""
    if output_dir.exists():
        raise FileExistsError(
            f"output directory already exists; refusing to overwrite: {output_dir}"
        )
    output_dir.mkdir(parents=True)
