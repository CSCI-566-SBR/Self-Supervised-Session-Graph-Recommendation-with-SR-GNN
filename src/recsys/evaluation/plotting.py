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

    raw_label = data.get("experiment") or path.parent.name
    if "deletion_aug" in raw_label:
        clean_label = "SR-GNN-DA"
    elif "sasrec" in raw_label:
        clean_label = "SASRec"
    elif "srgnn" in raw_label:
        clean_label = "SR-GNN"
    else:
        clean_label = raw_label
    model_type = data.get("model_type") or data.get("config", {}).get("model_type") or "model"
    history = data.get("history", [])

    return {
        "label": clean_label,
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
    labels: Sequence[str] | None = None,
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

    if labels and len(labels) == len(experiments):
        for exp, lbl in zip(experiments, labels):
            exp["label"] = lbl

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
        label_prefix = exp["label"]
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
                label=f"{label_prefix} (Train)",
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
                label=f"{label_prefix} (Val)",
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
                    label=metric_label,
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
                    label=metric_label,
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
                    label=metric_label,
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

    ax_recall.set_title(f"Validation Recall@{cutoffs_str}", fontsize=12, fontweight="bold")
    ax_recall.set_xlabel("Epoch")
    ax_recall.set_ylabel("Recall")
    ax_recall.grid(True, linestyle="--", alpha=0.6)

    ax_mrr.set_title(f"Validation MRR@{cutoffs_str}", fontsize=12, fontweight="bold")
    ax_mrr.set_xlabel("Epoch")
    ax_mrr.set_ylabel("MRR")
    ax_mrr.grid(True, linestyle="--", alpha=0.6)

    ax_ndcg.set_title(f"Validation NDCG@{cutoffs_str}", fontsize=12, fontweight="bold")
    ax_ndcg.set_xlabel("Epoch")
    ax_ndcg.set_ylabel("NDCG")
    ax_ndcg.grid(True, linestyle="--", alpha=0.6)

    handles, labels = ax_recall.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.95), ncol=len(labels), frameon=True, fontsize=10)

    main_title = title or f"Experiment Training & Validation Progression (@{cutoffs_str})"
    fig.suptitle(main_title, fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path


def load_robustness_results(path: Path | str, label: str | None = None) -> dict[str, Any]:
    """Load robustness_results.json and infer model label and metadata."""
    path = Path(path)
    if path.is_dir():
        path = path / "robustness_results.json"
    if not path.is_file():
        raise FileNotFoundError(f"Robustness results file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    inferred_label = label
    dataset = None
    companion = path.parent / "results.json"
    if companion.is_file():
        try:
            with companion.open("r", encoding="utf-8") as f:
                meta = json.load(f)
                dataset = meta.get("dataset")
                if not inferred_label:
                    m_type = meta.get("config", {}).get("model_type") or meta.get("model_type")
                    if m_type:
                        inferred_label = m_type.upper()
        except Exception:
            pass
    if not dataset:
        if "retailrocket" in str(path).lower():
            dataset = "RetailRocket"
        elif "diginetica" in str(path).lower():
            dataset = "Diginetica"
    if not inferred_label:
        inferred_label = path.parent.name

    return {
        "label": inferred_label,
        "dataset": dataset,
        "path": path,
        "data": data,
    }


def plot_robustness_curves(
    results_paths: Sequence[Path | str],
    output_path: Path | str,
    metrics: Sequence[str] = ("recall@20", "mrr@20"),
    labels: Sequence[str] | None = None,
    title: str | None = None,
) -> Path:
    """Plot degradation curves comparing models across noise regimes (deletion, substitution, mixed)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    loaded_exps = []
    for idx, p in enumerate(results_paths):
        lbl = labels[idx] if labels and idx < len(labels) else None
        loaded_exps.append(load_robustness_results(p, label=lbl))

    if not loaded_exps:
        raise ValueError("At least one robustness results file must be provided")

    corruptions = [
        ("deletion", "Item Deletion"),
        ("substitution", "Item Substitution"),
        ("mixed", "Mixed Corruption"),
    ]
    ratios = [0.0, 0.05, 0.10, 0.20]
    ratio_labels = ["0% (Clean)", "5%", "10%", "20%"]

    n_rows = len(metrics)
    n_cols = len(corruptions)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5 * n_cols, 4 * n_rows),
        squeeze=False,
        sharex=True,
        sharey="row",
    )

    colors = plt.cm.tab10.colors
    markers = ["o", "s", "^", "D", "v", "P"]

    for row_idx, metric in enumerate(metrics):
        metric_lower = metric.lower()
        is_pct = not metric_lower.endswith("loss")
        scale = 100.0 if is_pct else 1.0
        unit = "(%)" if is_pct else ""
        metric_name = metric.upper() if "ndcg" in metric_lower or "mrr" in metric_lower else metric.title()
        display_metric = f"{metric_name} {unit}".strip()

        row_y_vals: list[float] = []

        for col_idx, (corr_key, corr_title) in enumerate(corruptions):
            ax = axes[row_idx, col_idx]

            for exp_idx, exp in enumerate(loaded_exps):
                data = exp["data"]
                y_vals = []
                for r in ratios:
                    cond_name = "clean" if r == 0.0 else f"{corr_key}_{r:.2f}"
                    val = data.get(cond_name, {}).get(metric_lower)
                    if val is not None:
                        y_vals.append(float(val) * scale)
                    else:
                        y_vals.append(None)

                valid_x = [x for x, y in zip(ratios, y_vals) if y is not None]
                valid_y = [y for y in y_vals if y is not None]
                row_y_vals.extend(valid_y)

                color = colors[exp_idx % len(colors)]
                marker = markers[exp_idx % len(markers)]

                ax.plot(
                    valid_x,
                    valid_y,
                    color=color,
                    marker=marker,
                    linewidth=2.0,
                    markersize=6,
                    label=exp["label"],
                )

            ax.set_xticks(ratios)
            ax.set_xticklabels(ratio_labels, fontsize=10)
            ax.grid(True, linestyle="--", alpha=0.6)

            if row_idx == 0:
                ax.set_title(corr_title, fontsize=13, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(display_metric, fontsize=11, fontweight="bold")
            if row_idx == n_rows - 1:
                ax.set_xlabel("Noise Ratio", fontsize=11)

        # Explicitly enforce identical y-limits across all columns in this metric row
        if row_y_vals:
            min_y = min(row_y_vals)
            max_y = max(row_y_vals)
            pad = (max_y - min_y) * 0.08 if max_y > min_y else 1.0
            row_ylim = (min_y - pad, max_y + pad)
            for col_idx in range(n_cols):
                axes[row_idx, col_idx].set_ylim(row_ylim)

    if not title:
        datasets = {exp.get("dataset") for exp in loaded_exps if exp.get("dataset")}
        ds_name = "RetailRocket" if "retailrocket" in str(list(datasets)[0]).lower() else "Diginetica" if len(datasets) == 1 else ""
        ds_str = f" ({ds_name})" if ds_name else ""
        main_title = f"Model Robustness Under Synthetic Noise Corruption{ds_str}"
    else:
        main_title = title

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.95), ncol=len(loaded_exps), frameon=True, fontsize=10)
    fig.suptitle(main_title, fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path


def plot_robustness_bars(
    results_paths: Sequence[Path | str],
    output_path: Path | str,
    metric: str = "recall@20",
    labels: Sequence[str] | None = None,
    title: str | None = None,
) -> Path:
    """Plot grouped bar charts comparing models across all 10 conditions side-by-side."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    loaded_exps = []
    for idx, p in enumerate(results_paths):
        lbl = labels[idx] if labels and idx < len(labels) else None
        loaded_exps.append(load_robustness_results(p, label=lbl))

    conditions = [
        "clean",
        "deletion_0.05", "deletion_0.10", "deletion_0.20",
        "substitution_0.05", "substitution_0.10", "substitution_0.20",
        "mixed_0.05", "mixed_0.10", "mixed_0.20",
    ]
    cond_labels = [
        "Clean",
        "Del 5%", "Del 10%", "Del 20%",
        "Sub 5%", "Sub 10%", "Sub 20%",
        "Mix 5%", "Mix 10%", "Mix 20%",
    ]

    metric_lower = metric.lower()
    is_pct = not metric_lower.endswith("loss")
    scale = 100.0 if is_pct else 1.0
    unit = "(%)" if is_pct else ""

    x = np.arange(len(conditions))
    n_models = len(loaded_exps)
    total_width = 0.8
    bar_width = total_width / n_models

    fig, ax = plt.subplots(figsize=(14, 6))
    colors = plt.cm.tab10.colors

    for i, exp in enumerate(loaded_exps):
        offset = (i - n_models / 2 + 0.5) * bar_width
        vals = [
            float(exp["data"].get(c, {}).get(metric_lower, 0.0)) * scale
            for c in conditions
        ]
        ax.bar(
            x + offset,
            vals,
            width=bar_width,
            label=exp["label"],
            color=colors[i % len(colors)],
            edgecolor="black",
            linewidth=0.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(cond_labels, rotation=25, ha="right", fontsize=10)
    ax.set_ylabel(f"{metric.upper()} {unit}".strip(), fontsize=12, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.legend(loc="upper right", frameon=True)

    if not title:
        datasets = {exp.get("dataset") for exp in loaded_exps if exp.get("dataset")}
        ds_name = "RetailRocket" if "retailrocket" in str(list(datasets)[0]).lower() else "Diginetica" if len(datasets) == 1 else ""
        ds_str = f" ({ds_name})" if ds_name else ""
        main_title = f"Robustness Comparison Across Corruption Regimes ({metric.upper()}){ds_str}"
    else:
        main_title = title
    ax.set_title(main_title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path


def plot_stratified_results(
    results_map: dict[str, dict[str, Any]],
    output_path: Path | str,
    dataset_name: str = "Dataset",
) -> Path:
    """Render grouped bar chart comparing architectures across Short, Medium, Long sessions."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    models = list(results_map.keys())
    bucket_keys = ["short", "medium", "long"]
    bucket_labels = ["Short (1–3)", "Medium (4–6)", "Long (7+)"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax_rec, ax_mrr = axes[0], axes[1]

    colors = {"SR-GNN": "#1f77b4", "SASRec": "#ff7f0e", "SR-GNN-DA": "#2ca02c"}
    x = np.arange(len(bucket_keys))
    n_mod = len(models)
    bar_width = 0.75 / n_mod

    all_rec_vals: list[float] = []
    all_mrr_vals: list[float] = []

    for m_idx, mo in enumerate(models):
        data = results_map[mo]
        offset = (m_idx - n_mod / 2 + 0.5) * bar_width
        color = colors.get(mo, plt.cm.tab10.colors[m_idx % 10])

        rec_vals = [data.get(b, {}).get("recall@20", 0.0) * 100.0 for b in bucket_keys]
        mrr_vals = [data.get(b, {}).get("mrr@20", 0.0) * 100.0 for b in bucket_keys]
        all_rec_vals.extend(rec_vals)
        all_mrr_vals.extend(mrr_vals)

        bars_r = ax_rec.bar(x + offset, rec_vals, width=bar_width, label=mo, color=color, edgecolor="black", linewidth=0.6)
        bars_m = ax_mrr.bar(x + offset, mrr_vals, width=bar_width, label=mo, color=color, edgecolor="black", linewidth=0.6)

        for bar, v in zip(bars_r, rec_vals):
            ax_rec.text(bar.get_x() + bar.get_width() / 2, v + 0.5, f"{v:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")
        for bar, v in zip(bars_m, mrr_vals):
            ax_mrr.text(bar.get_x() + bar.get_width() / 2, v + 0.3, f"{v:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

    max_rec = max(all_rec_vals) if all_rec_vals else 50.0
    max_mrr = max(all_mrr_vals) if all_mrr_vals else 20.0
    ax_rec.set_ylim(0, max_rec * 1.15)
    ax_mrr.set_ylim(0, max_mrr * 1.15)

    for ax, title, ylabel in [(ax_rec, "Recall@20 by Session Length", "Recall@20 (%)"),
                              (ax_mrr, "MRR@20 by Session Length", "MRR@20 (%)")]:
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(bucket_labels, fontsize=10, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=11, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.6)

    handles, labels = ax_rec.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=len(models), frameon=True, fontsize=10)
    fig.suptitle(f"{dataset_name}: Architectural Accuracy Stratified by Session Length", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Stratified comparison plot saved to: {output_path.resolve()}\n")
    return output_path
