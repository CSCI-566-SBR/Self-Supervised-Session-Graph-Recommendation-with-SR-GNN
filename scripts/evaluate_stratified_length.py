"""Stratified robustness & accuracy evaluation across session length buckets (Short, Medium, Long)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
import torch

from recsys.data.processed import load_processed_dataset, ProcessedDataset
from recsys.evaluation.plotting import plot_stratified_results
from recsys.models.registry import get_model_builder
from recsys.training.baseline import evaluate_ranking
from recsys.utils import make_data_loader, resolve_device, resolve_num_workers


BUCKETS = {
    "short": (1, 3),      # Short: 1-3 clicks
    "medium": (4, 6),     # Medium: 4-6 clicks
    "long": (7, 1000),    # Long: 7+ clicks
}


def evaluate_model_by_length(
    experiment_dir: Path,
    data_root: Path,
    device_arg: str = "auto",
    num_workers_arg: int | None = None,
    batch_size: int = 100,
) -> dict[str, Any]:
    experiment_dir = Path(experiment_dir)
    data_root = Path(data_root)

    results_path = experiment_dir / "results.json"
    checkpoint_path = experiment_dir / "best_model.pt"
    if not results_path.is_file():
        raise FileNotFoundError(f"results.json not found in {experiment_dir}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"best_model.pt not found in {experiment_dir}")

    with results_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    cfg = meta.get("config", {})
    model_type = cfg.get("model_type", "srgnn")
    model_params = cfg.get("model_params", {})
    seed = cfg.get("seed", 2028)

    device = resolve_device(device_arg)
    num_workers = resolve_num_workers(device, requested=num_workers_arg)

    clean_ds: ProcessedDataset = load_processed_dataset(data_root / "clean")
    builder = get_model_builder(model_type)
    model, collate_fn = builder(model_params, clean_ds)

    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    all_test_examples = clean_ds.test
    bucket_results: dict[str, Any] = {}

    print(f"Evaluating {model_type.upper()} on {experiment_dir.name} stratified by session length...")
    for b_name, (min_len, max_len) in BUCKETS.items():
        subset = [
            (prefix, target) for prefix, target in all_test_examples
            if min_len <= len(prefix) <= max_len
        ]
        if not subset:
            continue

        loader = make_data_loader(
            subset,
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
        )

        res = evaluate_ranking(model, loader, device=device, k=20, cutoffs=(10, 20))
        bucket_results[b_name] = {
            "min_length": min_len,
            "max_length": max_len,
            "count": len(subset),
            "fraction": len(subset) / len(all_test_examples),
            "recall@10": res.metrics.get("recall@10", res.recall_at_k),
            "recall@20": res.metrics.get("recall@20", res.recall_at_k),
            "mrr@10": res.metrics.get("mrr@10", res.mrr_at_k),
            "mrr@20": res.metrics.get("mrr@20", res.mrr_at_k),
            "ndcg@10": res.metrics.get("ndcg@10", res.ndcg_at_k),
            "ndcg@20": res.metrics.get("ndcg@20", res.ndcg_at_k),
        }
        print(f"  [{b_name.upper():<6}] ({len(subset):>5} sessions) -> Recall@20: {bucket_results[b_name]['recall@20']:.4f} | MRR@20: {bucket_results[b_name]['mrr@20']:.4f}")

    out_file = experiment_dir / "stratified_length_results.json"
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(bucket_results, f, indent=2)
    print(f"Saved stratified results to: {out_file}\n")
    return bucket_results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiments", nargs="+", type=Path, help="Experiment directories containing best_model.pt and results.json")
    parser.add_argument("--data-root", type=Path, required=True, help="Data root directory containing clean/ (e.g. data/processed/diginetica)")
    parser.add_argument("--output-plot", type=Path, default=None, help="Optional output PNG path for the comparison bar chart")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num-workers", type=int, default=None)
    args = parser.parse_args()

    results_map: dict[str, dict[str, Any]] = {}
    for exp_p in args.experiments:
        name = "SR-GNN-DA" if "deletion_aug" in exp_p.name else ("SASRec" if "sasrec" in exp_p.name else "SR-GNN")
        res = evaluate_model_by_length(exp_p, args.data_root, device_arg=args.device, num_workers_arg=args.num_workers)
        results_map[name] = res

    ds_name = "RetailRocket" if "retailrocket" in str(args.data_root).lower() else "Diginetica"
    plot_path = args.output_plot or (Path("experiments") / f"{ds_name.lower()}_stratified_length.png")
    plot_stratified_results(results_map, plot_path, dataset_name=ds_name)


if __name__ == "__main__":
    main()
