"""Flexible chart generation for training curves and model comparison from results.json."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence


def load_experiment_history(path: Path | str) -> dict[str, Any]:
    """Load an experiment results.json file and extract metadata and history."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"results file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    label = data.get("experiment") or path.parent.name
    model_type = data.get("model_type") or data.get("config", {}).get("model_type") or "model"
    history = data.get("history", [])

    return {
        "label": label,
        "model_type": model_type.upper(),
        "history": history,
        "best_epoch": data.get("best_epoch"),
        "best_validation": data.get("best_validation", {}),
        "test": data.get("test", {}),
    }


def plot_experiment_curves(
    results_paths: Sequence[Path | str],
    output_path: Path | str,
    title: str | None = None,
    cutoffs: Sequence[int] | int = (10, 20),
    cutoff: int | None = None,
) -> Path:
    """Plot multi-panel training and validation curves from one or more results.json files.

    Supports plotting multiple cutoffs (e.g. @10 and @20) simultaneously on ranking subplots.
    Handles missing metrics or absent validation loss gracefully.
    """

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if cutoff is not None:
        target_cutoffs = [cutoff]
    elif isinstance(cutoffs, int):
        target_cutoffs = [cutoffs]
    else:
        target_cutoffs = sorted(list(set(cutoffs)))

    experiments = [load_experiment_history(p) for p in results_paths]
    if not experiments:
        raise ValueError("at least one results.json path must be provided")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_loss, ax_recall = axes[0, 0], axes[0, 1]
    ax_mrr, ax_ndcg = axes[1, 0], axes[1, 1]

    colors = plt.cm.tab10.colors

    # Define line styles and markers for different cutoffs
    cutoff_styles = {
        10: {"linestyle": "--", "marker": "^"},
        20: {"linestyle": "-", "marker": "o"},
    }

    for exp_idx, exp in enumerate(experiments):
        base_color = colors[exp_idx % len(colors)]
        label_prefix = f"[{exp['model_type']}] {exp['label']}" if len(experiments) > 1 else exp["model_type"]
        history = exp["history"]
        if not history:
            continue

        epochs = [entry.get("epoch", i + 1) for i, entry in enumerate(history)]

        # 1. Loss curves (Train and optional Val Loss)
        train_loss = [entry.get("train_loss") for entry in history]
        if any(v is not None for v in train_loss):
            ax_loss.plot(
                epochs,
                train_loss,
                color=base_color,
                linestyle="-",
                marker="o",
                markersize=3,
                label=f"{label_prefix} Train",
            )

        val_loss = [entry.get("val_loss") for entry in history]
        if any(v is not None for v in val_loss):
            val_epochs = [epochs[i] for i, v in enumerate(val_loss) if v is not None]
            val_values = [v for v in val_loss if v is not None]
            ax_loss.plot(
                val_epochs,
                val_values,
                color=base_color,
                linestyle="--",
                marker="s",
                markersize=3,
                label=f"{label_prefix} Val",
            )

        # 2. Ranking metric curves for each requested cutoff
        def _extract_metric(metric_name: str) -> tuple[list[int], list[float]]:
            ep_list, val_list = [], []
            for i, entry in enumerate(history):
                val_dict = entry.get("validation", {})
                value = val_dict.get(metric_name)
                if value is not None:
                    ep_list.append(epochs[i])
                    val_list.append(float(value))
            return ep_list, val_list

        for c_idx, k in enumerate(target_cutoffs):
            style = cutoff_styles.get(k, {"linestyle": "-" if c_idx == len(target_cutoffs) - 1 else "--", "marker": "s"})
            
            # If single experiment, use distinct colors for different cutoffs; otherwise keep experiment color
            if len(experiments) == 1 and len(target_cutoffs) > 1:
                plot_color = colors[c_idx % len(colors)]
                metric_label = f"@{k}"
            else:
                plot_color = base_color
                metric_label = f"{label_prefix} @{k}" if len(target_cutoffs) > 1 else label_prefix

            rec_eps, rec_vals = _extract_metric(f"recall@{k}")
            if rec_vals:
                ax_recall.plot(
                    rec_eps,
                    rec_vals,
                    color=plot_color,
                    linestyle=style["linestyle"],
                    marker=style["marker"],
                    markersize=3,
                    label=f"Recall{metric_label}",
                )

            mrr_eps, mrr_vals = _extract_metric(f"mrr@{k}")
            if mrr_vals:
                ax_mrr.plot(
                    mrr_eps,
                    mrr_vals,
                    color=plot_color,
                    linestyle=style["linestyle"],
                    marker=style["marker"],
                    markersize=3,
                    label=f"MRR{metric_label}",
                )

            ndcg_eps, ndcg_vals = _extract_metric(f"ndcg@{k}")
            if ndcg_vals:
                ax_ndcg.plot(
                    ndcg_eps,
                    ndcg_vals,
                    color=plot_color,
                    linestyle=style["linestyle"],
                    marker=style["marker"],
                    markersize=3,
                    label=f"NDCG{metric_label}",
                )

        # Mark best epoch if available
        best_epoch = exp.get("best_epoch")
        if best_epoch and best_epoch in epochs:
            ax_recall.axvline(best_epoch, color=base_color, linestyle=":", alpha=0.4)
            ax_mrr.axvline(best_epoch, color=base_color, linestyle=":", alpha=0.4)

    cutoffs_str = "/".join(str(k) for k in target_cutoffs)

    # Styling subplots
    ax_loss.set_title("Cross-Entropy Loss Progression", fontsize=12, fontweight="bold")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.grid(True, linestyle="--", alpha=0.6)
    ax_loss.legend(loc="upper right", frameon=True)

    ax_recall.set_title(f"Validation Recall@{cutoffs_str}", fontsize=12, fontweight="bold")
    ax_recall.set_xlabel("Epoch")
    ax_recall.set_ylabel("Recall")
    ax_recall.grid(True, linestyle="--", alpha=0.6)
    ax_recall.legend(loc="lower right", frameon=True)

    ax_mrr.set_title(f"Validation MRR@{cutoffs_str}", fontsize=12, fontweight="bold")
    ax_mrr.set_xlabel("Epoch")
    ax_mrr.set_ylabel("MRR")
    ax_mrr.grid(True, linestyle="--", alpha=0.6)
    ax_mrr.legend(loc="lower right", frameon=True)

    ax_ndcg.set_title(f"Validation NDCG@{cutoffs_str}", fontsize=12, fontweight="bold")
    ax_ndcg.set_xlabel("Epoch")
    ax_ndcg.set_ylabel("NDCG")
    ax_ndcg.grid(True, linestyle="--", alpha=0.6)
    ax_ndcg.legend(loc="lower right", frameon=True)

    main_title = title or f"Experiment Training & Validation Progression (@{cutoffs_str})"
    fig.suptitle(main_title, fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path
