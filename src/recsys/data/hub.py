"""Upload/download frozen processed artifacts with the Hugging Face Hub.

The dependency is optional. Install it with ``uv sync --extra hub`` and log in
with ``hf auth login`` before uploading private project artifacts.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .processed import load_processed_dataset


def _hub_api() -> tuple[Any, Any]:
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as error:
        raise RuntimeError(
            "Hugging Face support is optional; run `uv sync --extra hub` first"
        ) from error
    return HfApi, snapshot_download


def validate_pipeline_tree(root: str | Path) -> None:
    """Validate every model-facing directory before upload or after download."""
    root = Path(root)
    load_processed_dataset(root / "clean")
    noisy_root = root / "noisy"
    variants = sorted(path for path in noisy_root.iterdir() if path.is_dir())
    if not variants:
        raise ValueError(f"no noisy dataset directories found in {noisy_root}")
    for variant in variants:
        load_processed_dataset(variant)


def upload_pipeline(
    source: str | Path,
    repo_id: str,
    *,
    private: bool = True,
    revision: str | None = None,
) -> str:
    """Validate and upload a complete generated pipeline tree."""
    source = Path(source).expanduser().resolve()
    validate_pipeline_tree(source)
    HfApi, _ = _hub_api()
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    result = api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=source,
        revision=revision,
        commit_message="Upload frozen SR-GNN processed artifacts",
        ignore_patterns=["**/.cache/**", "**/preprocessing.log"],
    )
    return str(result)


def download_pipeline(
    repo_id: str,
    destination: str | Path,
    *,
    revision: str | None = None,
) -> Path:
    """Download a trusted frozen artifact snapshot and validate its contract."""
    _, snapshot_download = _hub_api()
    destination = Path(destination).expanduser().resolve()
    result = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=destination,
    )
    validate_pipeline_tree(result)
    return Path(result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    upload = subparsers.add_parser("upload")
    upload.add_argument("--source", required=True, type=Path)
    upload.add_argument("--repo-id", required=True)
    upload.add_argument("--revision")
    upload.add_argument("--public", action="store_true")
    download = subparsers.add_parser("download")
    download.add_argument("--repo-id", required=True)
    download.add_argument("--destination", required=True, type=Path)
    download.add_argument("--revision")
    args = parser.parse_args()
    if args.command == "upload":
        print(
            upload_pipeline(
                args.source,
                args.repo_id,
                private=not args.public,
                revision=args.revision,
            )
        )
    else:
        print(download_pipeline(args.repo_id, args.destination, revision=args.revision))


if __name__ == "__main__":
    main()
