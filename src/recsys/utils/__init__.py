"""Central utilities package for system, hardware, I/O, reproducibility, and data helpers."""

from .artifact import format_results_dict, validate_artifact_identity, validate_robustness_artifacts
from .git import get_git_commit
from .hashing import sha256, sha256_file
from .io import load_pickle, load_toml, prepare_output_dir, save_pickle

__all__ = [
    "format_results_dict",
    "get_git_commit",
    "load_pickle",
    "load_toml",
    "make_data_loader",
    "prepare_output_dir",
    "resolve_device",
    "resolve_num_workers",
    "save_pickle",
    "set_random_seed",
    "sha256",
    "sha256_file",
    "validate_artifact_identity",
    "validate_robustness_artifacts",
]


def __getattr__(name: str):
    if name == "make_data_loader":
        from .data import make_data_loader
        return make_data_loader
    if name in {"resolve_device", "resolve_num_workers"}:
        from .device import resolve_device, resolve_num_workers
        return resolve_device if name == "resolve_device" else resolve_num_workers
    if name == "set_random_seed":
        from .reproducibility import set_random_seed
        return set_random_seed
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

