"""Automated robustness evaluation across clean and all 9 noisy test splits.

Loads a trained model checkpoint and evaluates ranking metrics across:
  - clean
  - deletion (5%, 10%, 20%)
  - mixed (5%, 10%, 20%)
  - substitution (5%, 10%, 20%)

Usage:
    python scripts/evaluate_robustness.py --config configs/eval_robustness_sasrec_diginetica.toml
    python scripts/evaluate_robustness.py --config configs/eval_robustness_srgnn_diginetica.toml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any
import torch

from recsys.data.hub import download_pipeline
from recsys.data.processed import load_processed_dataset
from recsys.models.registry import get_model_builder
from recsys.training import evaluate_ranking
from recsys.utils import (
    get_git_commit,
    load_toml,
    make_data_loader,
    resolve_device,
    resolve_num_workers,
    sha256_file,
)

DEFAULT_REPO_ID = "CSCI-566-SB/srgnn-processed-diginetica"

CONDITIONS = [
    "clean",
    "deletion_0.05",
    "deletion_0.10",
    "deletion_0.20",
    "mixed_0.05",
    "mixed_0.10",
    "mixed_0.20",
    "substitution_0.05",
    "substitution_0.10",
    "substitution_0.20",
]


def ensure_datasets_available(
    data_root: Path,
    repo_id: str,
    revision: str | None = None,
) -> None:
    """Download clean and noisy splits via HF Hub if missing."""
    clean_dir = data_root / "clean"
    noisy_dir = data_root / "noisy"
    missing = not clean_dir.is_dir() or not (clean_dir / "test.txt").is_file()
    if not missing:
        for condition in CONDITIONS[1:]:
            cond_dir = noisy_dir / condition
            if not cond_dir.is_dir() or not (cond_dir / "test.txt").is_file():
                missing = True
                break

    if missing:
        print(f"Dataset splits missing in {data_root}. Downloading from Hugging Face: {repo_id} (revision: {revision})...")
        download_pipeline(repo_id, data_root, revision=revision)
        print("Download and artifact verification completed.")


def resolve_checkpoint(experiment_dir: Path, explicit_checkpoint: str | None = None) -> Path:
    """Locate checkpoint file, checking explicit path, best_model.pt, and model.pt."""
    candidates = []
    if explicit_checkpoint:
        candidates.append(Path(explicit_checkpoint) if Path(explicit_checkpoint).is_absolute() else experiment_dir / explicit_checkpoint)
    candidates.extend([
        experiment_dir / "best_model.pt",
        experiment_dir / "model.pt",
    ])

    for path in candidates:
        if path.is_file():
            return path

    checked = "\n".join(f"  - {p}" for p in candidates)
    raise FileNotFoundError(
        f"Checkpoint weights not found. Looked for:\n{checked}\n"
        "Ensure training completed or checkpoint path is configured correctly."
    )


def run_robustness_evaluation(
    experiment_dir: Path,
    data_root: Path,
    repo_id: str = DEFAULT_REPO_ID,
    revision: str | None = None,
    device_arg: str = "auto",
    batch_size: int = 100,
    checkpoint_name: str | None = None,
    output_file: Path | None = None,
    num_workers_arg: int | None = None,
) -> dict[str, dict[str, Any]]:
    experiment_dir = Path(experiment_dir)
    results_path = experiment_dir / "results.json"
    checkpoint_path = resolve_checkpoint(experiment_dir, checkpoint_name)

    if not results_path.is_file():
        raise FileNotFoundError(f"Experiment results file not found: {results_path}")

    ensure_datasets_available(data_root, repo_id, revision=revision)

    with results_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    cfg = meta.get("config", {})
    model_type = cfg.get("model_type", "srgnn")
    model_params = cfg.get("model_params", {})
    seed = cfg.get("seed", 2028)

    device = resolve_device(device_arg)
    num_workers = resolve_num_workers(device, requested=num_workers_arg)

    print(f"Initializing {model_type.upper()} model using vocabulary from {data_root / 'clean'}...")
    clean_ds = load_processed_dataset(data_root / "clean")
    builder = get_model_builder(model_type)
    model, collate_fn = builder(model_params, clean_ds)

    print(f"Loading checkpoint weights: {checkpoint_path}...")
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    all_metrics: dict[str, dict[str, Any]] = {}

    print(f"\nEvaluating {model_type.upper()} on device: {device} (Workers: {num_workers})")
    print("=" * 95)
    header = (
        f"{'Condition':<20} | {'Recall@10':<10} | {'Recall@20':<10} | "
        f"{'MRR@10':<8} | {'MRR@20':<8} | {'NDCG@10':<8} | {'NDCG@20':<8} | {'Time':<6}"
    )
    print(header)
    print("-" * 95)

    clean_targets: list[int] | None = None

    for cond in CONDITIONS:
        cond_dir = data_root / "clean" if cond == "clean" else data_root / "noisy" / cond
        ds = load_processed_dataset(cond_dir)

        # Runtime target and length invariance verification
        if cond == "clean":
            clean_targets = [target for _, target in ds.test]
        else:
            if len(ds.test) != len(clean_ds.test):
                raise ValueError(
                    f"Example count mismatch in {cond}: {len(ds.test)} vs clean {len(clean_ds.test)}"
                )
            noisy_targets = [target for _, target in ds.test]
            if noisy_targets != clean_targets:
                raise ValueError(f"Target invariance violated in noisy condition: {cond}")

        test_loader = make_data_loader(
            examples=ds.test,
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
        )

        t0 = time.perf_counter()
        res = evaluate_ranking(model, test_loader, device=device, k=20, cutoffs=[10, 20])
        elapsed = time.perf_counter() - t0

        r10 = res.metrics.get("recall@10", 0.0)
        r20 = res.metrics.get("recall@20", res.recall_at_k)
        m10 = res.metrics.get("mrr@10", 0.0)
        m20 = res.metrics.get("mrr@20", res.mrr_at_k)
        n10 = res.metrics.get("ndcg@10", 0.0)
        n20 = res.metrics.get("ndcg@20", res.ndcg_at_k)

        test_file = cond_dir / "test.txt"
        test_sha = sha256_file(test_file) if test_file.is_file() else None

        if cond == "clean":
            corruption_type = "clean"
            corruption_rate = 0.0
        else:
            parts = cond.split("_", 1)
            corruption_type = parts[0]
            corruption_rate = float(parts[1]) if len(parts) > 1 else 0.0

        all_metrics[cond] = {
            "corruption_type": corruption_type,
            "corruption_rate": corruption_rate,
            "test_checksum_sha256": test_sha,
            "recall@10": r10,
            "recall@20": r20,
            "mrr@10": m10,
            "mrr@20": m20,
            "ndcg@10": n10,
            "ndcg@20": n20,
            "loss": res.val_loss,
            "examples": len(ds.test),
            "eval_time_seconds": elapsed,
        }

        if cond == "clean":
            all_metrics["clean"].update({
                "git_commit": get_git_commit(),
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "checkpoint_file": checkpoint_path.name,
                "model_type": model_type,
                "model_params": model_params,
                "artifact_repository": repo_id,
                "artifact_revision": revision,
                "device": str(device),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            })

        print(
            f"{cond:<20} | {r10*100:6.2f}%    | {r20*100:6.2f}%    | "
            f"{m10*100:5.2f}%  | {m20*100:5.2f}%  | {n10*100:5.2f}%  | {n20*100:5.2f}%  | {elapsed:4.1f}s"
        )

    out_path = Path(output_file) if output_file else experiment_dir / "robustness_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    print("=" * 95)
    print(f"Results successfully saved to: {out_path}\n")
    return all_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to evaluation TOML config (e.g. configs/eval_robustness_sasrec_diginetica.toml)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override ('cuda', 'cpu', or 'auto')",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="DataLoader worker count override (defaults to min(4, CPU count) on CUDA, 0 on CPU)",
    )
    args = parser.parse_args()

    doc = load_toml(args.config)

    # Support either dedicated [evaluation] section or training [experiment] config
    if "evaluation" in doc:
        eval_cfg = doc["evaluation"]
        data_cfg = doc.get("data", {})
        experiment_dir = Path(eval_cfg["experiment_dir"])
        data_root = Path(eval_cfg.get("data_root", "data/processed/diginetica"))
        repo_id = data_cfg.get("artifact_repository", DEFAULT_REPO_ID)
        device = args.device or eval_cfg.get("device", "auto")
        batch_size = int(eval_cfg.get("batch_size", 100))
        checkpoint = eval_cfg.get("checkpoint")
        num_workers = args.num_workers if args.num_workers is not None else eval_cfg.get("num_workers")
        output_file = Path(eval_cfg["output_file"]) if "output_file" in eval_cfg else None
    elif "experiment" in doc:
        exp_cfg = doc["experiment"]
        data_cfg = doc.get("data", {})
        train_cfg = doc.get("training", {})
        experiment_dir = Path(exp_cfg["output_dir"])
        proc_dir = Path(data_cfg.get("processed_data_dir", "data/processed/diginetica/clean"))
        data_root = proc_dir.parent if proc_dir.name in ("clean", "noisy") else proc_dir
        repo_id = data_cfg.get("artifact_repository", DEFAULT_REPO_ID)
        device = args.device or train_cfg.get("device", "auto")
        batch_size = int(train_cfg.get("batch_size", 100))
        num_workers = args.num_workers if args.num_workers is not None else train_cfg.get("num_workers")
        checkpoint = None
        output_file = experiment_dir / "robustness_results.json"
    else:
        raise ValueError(
            f"Config file {args.config} must contain either [evaluation] or [experiment] section."
        )

    revision = data_cfg.get("artifact_revision")

    run_robustness_evaluation(
        experiment_dir=experiment_dir,
        data_root=data_root,
        repo_id=repo_id,
        revision=revision,
        device_arg=device,
        batch_size=batch_size,
        checkpoint_name=checkpoint,
        output_file=output_file,
        num_workers_arg=num_workers,
    )


if __name__ == "__main__":
    main()
