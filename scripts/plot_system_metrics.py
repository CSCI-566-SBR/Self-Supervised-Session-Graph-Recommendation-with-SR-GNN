"""Generate publication-ready Robustness Heatmaps, Grouped System Efficiency,
Noise Fragility Asymmetry, Augmentation Delta Gain, and Multi-Axis Radar charts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CONDITIONS = [
    ("clean", "Clean"),
    ("deletion_0.05", "Del 5%"),
    ("deletion_0.10", "Del 10%"),
    ("deletion_0.20", "Del 20%"),
    ("substitution_0.05", "Sub 5%"),
    ("substitution_0.10", "Sub 10%"),
    ("substitution_0.20", "Sub 20%"),
    ("mixed_0.05", "Mix 5%"),
    ("mixed_0.10", "Mix 10%"),
    ("mixed_0.20", "Mix 20%"),
]

MODEL_COLORS = {
    "SR-GNN": "#1f77b4",       # Blue
    "SASRec": "#ff7f0e",       # Orange
    "SR-GNN-DA": "#2ca02c",    # Green
    "SGL": "#9467bd",          # Purple (ready for contrastive model)
    "SGL-DA": "#8c564b",       # Brown
}


def load_model_data(exp_dir: Path) -> dict[str, Any]:
    exp_dir = Path(exp_dir)
    results_path = exp_dir / "results.json"
    rob_path = exp_dir / "robustness_results.json"

    meta = {}
    if results_path.is_file():
        with results_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)

    rob = {}
    if rob_path.is_file():
        with rob_path.open("r", encoding="utf-8") as f:
            rob = json.load(f)

    dataset_raw = meta.get("dataset") or ("retailrocket" if "retailrocket" in exp_dir.name else "diginetica")
    dataset = "RetailRocket" if "retailrocket" in str(dataset_raw).lower() else "Diginetica"

    model_type = meta.get("model_type") or meta.get("config", {}).get("model_type", "model")
    if "deletion_aug" in exp_dir.name:
        label = "SR-GNN-DA"
    elif "sasrec" in exp_dir.name or model_type == "sasrec":
        label = "SASRec"
    elif "sgl" in exp_dir.name:
        label = "SGL"
    else:
        label = "SR-GNN"

    # Compute inference throughput (sessions per second)
    clean_rob = rob.get("clean", {})
    eval_time = float(clean_rob.get("eval_time_seconds", 0) or 0)
    examples = int(clean_rob.get("examples", 0) or 0)
    throughput = (examples / eval_time) if eval_time > 0 and examples > 0 else 0.0

    return {
        "exp_dir": exp_dir,
        "label": label,
        "dataset": dataset,
        "meta": meta,
        "robustness": rob,
        "throughput": throughput,
    }


def plot_robustness_heatmap(
    experiments: list[dict[str, Any]],
    output_path: Path,
    metric: str = "recall@20",
    retention: bool = True,
) -> Path:
    """Plot a heatmap of model robustness across noise conditions."""
    n_models = len(experiments)
    n_conds = len(CONDITIONS)

    matrix = np.zeros((n_models, n_conds))
    row_labels = []
    col_labels = [c[1] for c in CONDITIONS]

    for i, exp in enumerate(experiments):
        row_labels.append(f"{exp['dataset']} - {exp['label']}")
        rob = exp["robustness"]
        clean_val = float(rob.get("clean", {}).get(metric, 1e-6))
        for j, (cond_key, _) in enumerate(CONDITIONS):
            val = float(rob.get(cond_key, {}).get(metric, 0.0))
            if retention:
                matrix[i, j] = (val / clean_val) * 100.0 if clean_val > 0 else 0.0
            else:
                matrix[i, j] = val * 100.0

    fig, ax = plt.subplots(figsize=(12, max(4, n_models * 0.9 + 1.5)))
    cmap = "Blues" if not retention else "YlGnBu"
    im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=np.nanmin(matrix) * 0.98, vmax=np.nanmax(matrix) * 1.01)

    ax.set_xticks(np.arange(n_conds))
    ax.set_yticks(np.arange(n_models))
    ax.set_xticklabels(col_labels, fontsize=11, fontweight="medium")
    ax.set_yticklabels(row_labels, fontsize=11, fontweight="bold")

    for i in range(n_models):
        for j in range(n_conds):
            val = matrix[i, j]
            txt = f"{val:.1f}%" if retention else f"{val:.2f}%"
            text_color = "white" if val > (matrix.max() + matrix.min()) / 2 else "black"
            ax.text(j, i, txt, ha="center", va="center", color=text_color, fontsize=10, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.03)
    cbar.set_label("Relative Retention (%)" if retention else f"{metric.upper()} (%)", fontsize=11, fontweight="bold")

    metric_name = metric.upper()
    title_suffix = "Relative Retention (% of Clean)" if retention else f"Absolute Performance ({metric_name})"
    ax.set_title(f"Robustness Heatmap Across Synthetic Noise Regimes\n{title_suffix}", fontsize=14, fontweight="bold", pad=15)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_grouped_system_efficiency(
    experiments: list[dict[str, Any]],
    output_path: Path,
) -> Path:
    """Grouped Bar Chart (X-axis: Dataset, Hue: Model) across Memory, Training Time, Epoch Latency, Throughput."""
    datasets = sorted(list({e["dataset"] for e in experiments}))
    models = sorted(list({e["label"] for e in experiments}), key=lambda m: ["SR-GNN", "SASRec", "SR-GNN-DA", "SGL"].index(m) if m in ["SR-GNN", "SASRec", "SR-GNN-DA", "SGL"] else 99)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_mem, ax_thru = axes[0, 0], axes[0, 1]
    ax_epoch, ax_time = axes[1, 0], axes[1, 1]

    n_groups = len(datasets)
    n_bars = len(models)
    total_width = 0.75
    bar_width = total_width / n_bars
    x = np.arange(n_groups)

    # Helper to extract matrix [dataset_idx, model_idx]
    def extract_grid(extractor_fn):
        grid = np.zeros((n_groups, n_bars))
        for d_idx, ds in enumerate(datasets):
            for m_idx, mo in enumerate(models):
                matching = [e for e in experiments if e["dataset"] == ds and e["label"] == mo]
                if matching:
                    grid[d_idx, m_idx] = extractor_fn(matching[0])
        return grid

    # 1. Peak GPU Memory (MB)
    mem_grid = extract_grid(lambda e: float(e["meta"].get("peak_gpu_memory_bytes", 0) or 0) / (1024 * 1024))
    # 2. Inference Throughput (Sessions / Sec)
    thru_grid = extract_grid(lambda e: float(e["throughput"]))
    # 3. Epoch Latency (Seconds)
    epoch_grid = extract_grid(lambda e: np.mean([h.get("epoch_seconds", 0) for h in e["meta"].get("history", []) if h.get("epoch_seconds")]) if e["meta"].get("history") else 0.0)
    # 4. Total Training Time (Minutes)
    time_grid = extract_grid(lambda e: float(e["meta"].get("runtime_seconds", 0) or 0) / 60.0)

    panels = [
        (ax_mem, mem_grid, "Peak GPU Memory Footprint", "Memory (MB)", "{:.1f} MB"),
        (ax_thru, thru_grid, "Inference Serving Throughput", "Throughput (Sessions / Sec)", "{:,.0f} s/s"),
        (ax_epoch, epoch_grid, "Average Training Time per Epoch", "Latency (Seconds / Epoch)", "{:.1f}s"),
        (ax_time, time_grid, "Total Training Duration", "Time (Minutes)", "{:.1f} min"),
    ]

    for ax, grid, title, ylabel, fmt in panels:
        max_val = float(grid.max()) if grid.size > 0 else 1.0
        ax.set_ylim(0, max(max_val * 1.15, 1.0))
        for m_idx, mo in enumerate(models):
            offset = (m_idx - n_bars / 2 + 0.5) * bar_width
            color = MODEL_COLORS.get(mo, plt.cm.tab10.colors[m_idx % 10])
            vals = grid[:, m_idx]
            bars = ax.bar(x + offset, vals, width=bar_width, label=mo, color=color, edgecolor="black", linewidth=0.6)
            for bar, val in zip(bars, vals):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, val + max(grid.max() * 0.02, 0.5), fmt.format(val),
                            ha="center", va="bottom", fontsize=8, fontweight="bold")

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, fontsize=11, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.6)

    handles, labels = ax_mem.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.95), ncol=len(models), frameon=True, fontsize=10)
    fig.suptitle("System Efficiency and Real-Time Serving Benchmark (Grouped by Dataset)", fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_noise_fragility_bars(
    experiments: list[dict[str, Any]],
    output_path: Path,
    metric: str = "recall@20",
) -> Path:
    """Grouped bar chart showing Relative % Drop at 20% Noise across Deletion, Substitution, Mixed."""
    datasets = sorted(list({e["dataset"] for e in experiments}))
    corruptions = [("deletion_0.20", "Item Deletion (20%)"),
                   ("substitution_0.20", "Item Substitution (20%)"),
                   ("mixed_0.20", "Mixed Noise (20%)")]

    fig, axes = plt.subplots(1, len(datasets), figsize=(7 * len(datasets), 5), sharey=True)
    if len(datasets) == 1:
        axes = [axes]

    for ax, ds in zip(axes, datasets):
        ds_exps = [e for e in experiments if e["dataset"] == ds]
        models = [e["label"] for e in ds_exps]
        n_corr = len(corruptions)
        n_mod = len(models)
        x = np.arange(n_corr)
        total_width = 0.75
        bar_width = total_width / n_mod

        for m_idx, exp in enumerate(ds_exps):
            mo = exp["label"]
            rob = exp["robustness"]
            clean_val = float(rob.get("clean", {}).get(metric, 1e-6))
            drops = []
            for cond_key, _ in corruptions:
                c_val = float(rob.get(cond_key, {}).get(metric, 0.0))
                rel_drop = max(0.0, (1.0 - (c_val / clean_val))) * 100.0 if clean_val > 0 else 0.0
                drops.append(rel_drop)

            offset = (m_idx - n_mod / 2 + 0.5) * bar_width
            color = MODEL_COLORS.get(mo, plt.cm.tab10.colors[m_idx % 10])
            bars = ax.bar(x + offset, drops, width=bar_width, label=mo, color=color, edgecolor="black", linewidth=0.6)
            for bar, drop in zip(bars, drops):
                ax.text(bar.get_x() + bar.get_width() / 2, drop + 0.4, f"-{drop:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

        ax.set_title(f"{ds} (Relative Drop at 20% Noise)", fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([c[1] for c in corruptions], fontsize=10, fontweight="medium")
        ax.set_ylabel(f"Relative {metric.upper()} Drop (%)", fontsize=11, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.6)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.95), ncol=len(models), frameon=True, fontsize=10)
    fig.suptitle("Noise Fragility Asymmetry: Click Omission vs. False Click Injection", fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_augmentation_delta(
    experiments: list[dict[str, Any]],
    output_path: Path,
) -> Path:
    """Plot Δ (SR-GNN-DA minus baseline SR-GNN) across noise conditions to isolate data augmentation benefits."""
    datasets = sorted(list({e["dataset"] for e in experiments}))
    fig, axes = plt.subplots(len(datasets), 2, figsize=(14, 4 * len(datasets)), sharex=True)
    if len(datasets) == 1:
        axes = np.expand_dims(axes, 0)

    cond_keys = [c[0] for c in CONDITIONS]
    cond_labels = [c[1] for c in CONDITIONS]
    x = np.arange(len(CONDITIONS))

    for row_idx, ds in enumerate(datasets):
        baseline = next((e for e in experiments if e["dataset"] == ds and e["label"] == "SR-GNN"), None)
        aug = next((e for e in experiments if e["dataset"] == ds and e["label"] == "SR-GNN-DA"), None)
        if not baseline or not aug:
            continue

        base_rob = baseline["robustness"]
        aug_rob = aug["robustness"]

        for col_idx, (metric, m_title) in enumerate([("recall@20", "Δ Recall@20 (%)"), ("mrr@20", "Δ MRR@20 (%)")]):
            ax = axes[row_idx, col_idx]
            deltas = []
            for k in cond_keys:
                b_val = float(base_rob.get(k, {}).get(metric, 0.0)) * 100.0
                a_val = float(aug_rob.get(k, {}).get(metric, 0.0)) * 100.0
                deltas.append(a_val - b_val)

            bar_colors = ["#2ca02c" if d >= 0 else "#d62728" for d in deltas]
            bars = ax.bar(x, deltas, color=bar_colors, edgecolor="black", width=0.6, linewidth=0.6)
            ax.axhline(0, color="black", linewidth=1.0, linestyle="--")

            min_d = min(deltas) if deltas else -0.1
            max_d = max(deltas) if deltas else 0.1
            span = max(max_d - min_d, 0.1)
            y_bottom = (min_d - span * 0.35) if min_d < 0 else (-span * 0.1)
            y_top = (max_d + span * 0.35) if max_d > 0 else (span * 0.1)
            ax.set_ylim(y_bottom, y_top)

            for bar, d in zip(bars, deltas):
                if d >= 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, d + span * 0.03, f"{d:+.2f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")
                else:
                    ax.text(bar.get_x() + bar.get_width() / 2, d - span * 0.03, f"{d:+.2f}%", ha="center", va="top", fontsize=8, fontweight="bold")

            ax.set_title(f"{ds}: {m_title} [SR-GNN-DA vs. SR-GNN]", fontsize=12, fontweight="bold")
            ax.set_ylabel(m_title, fontsize=10, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels(cond_labels, rotation=25, ha="right", fontsize=9)
            ax.tick_params(axis="x", pad=6)
            ax.grid(axis="y", linestyle="--", alpha=0.6)

    fig.suptitle("Data Augmentation Net Differential: SR-GNN-DA Empirical Gain Over Baseline", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_radar_chart(
    experiments: list[dict[str, Any]],
    output_path: Path,
) -> Path:
    """Multi-Axis Radar Chart comparing models across 5 holistic axes."""
    datasets = sorted(list({e["dataset"] for e in experiments}))
    axes_labels = ["Clean MRR@20", "Noise Retention", "Training Speed", "Inference Throughput", "Memory Footprint"]
    num_vars = len(axes_labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # close polygon

    fig, subaxes = plt.subplots(1, len(datasets), figsize=(7 * len(datasets), 7), subplot_kw=dict(polar=True))
    if len(datasets) == 1:
        subaxes = [subaxes]

    for ax, ds in zip(subaxes, datasets):
        ds_exps = [e for e in experiments if e["dataset"] == ds]

        # Gather raw values across models in this dataset to min-max normalize onto [0.2, 1.0] scale
        raw_vals = {"mrr": [], "retention": [], "train_speed": [], "throughput": [], "memory": []}
        for exp in ds_exps:
            rob = exp["robustness"]
            clean_mrr = float(rob.get("clean", {}).get("mrr@20", 0.0))
            # Retention across all 9 noise conditions
            clean_rec = float(rob.get("clean", {}).get("recall@20", 1e-6))
            ret_vals = [float(rob.get(k, {}).get("recall@20", 0.0)) / clean_rec for k, _ in CONDITIONS[1:]]
            avg_ret = np.mean(ret_vals) * 100.0 if ret_vals else 0.0

            # Training speed (inverse of average epoch time)
            history = exp["meta"].get("history", [])
            durations = [h.get("epoch_seconds", 0) for h in history if h.get("epoch_seconds")]
            avg_epoch_time = np.mean(durations) if durations else 1.0
            train_speed = 100.0 / avg_epoch_time

            throughput = float(exp["throughput"])
            peak_mem = float(exp["meta"].get("peak_gpu_memory_bytes", 1) or 1) / (1024 * 1024)
            # Memory efficiency (inverse of memory)
            mem_eff = 1000.0 / peak_mem

            raw_vals["mrr"].append(clean_mrr)
            raw_vals["retention"].append(avg_ret)
            raw_vals["train_speed"].append(train_speed)
            raw_vals["throughput"].append(throughput)
            raw_vals["memory"].append(mem_eff)

        # Normalization helper
        def norm(val, key):
            arr = raw_vals[key]
            min_v, max_v = min(arr), max(arr)
            if max_v == min_v:
                return 0.8
            return 0.3 + 0.7 * ((val - min_v) / (max_v - min_v))

        for idx, exp in enumerate(ds_exps):
            mo = exp["label"]
            rob = exp["robustness"]
            clean_mrr = float(rob.get("clean", {}).get("mrr@20", 0.0))
            clean_rec = float(rob.get("clean", {}).get("recall@20", 1e-6))
            ret_vals = [float(rob.get(k, {}).get("recall@20", 0.0)) / clean_rec for k, _ in CONDITIONS[1:]]
            avg_ret = np.mean(ret_vals) * 100.0 if ret_vals else 0.0

            history = exp["meta"].get("history", [])
            durations = [h.get("epoch_seconds", 0) for h in history if h.get("epoch_seconds")]
            train_speed = 100.0 / (np.mean(durations) if durations else 1.0)
            throughput = float(exp["throughput"])
            peak_mem = float(exp["meta"].get("peak_gpu_memory_bytes", 1) or 1) / (1024 * 1024)
            mem_eff = 1000.0 / peak_mem

            scores = [
                norm(clean_mrr, "mrr"),
                norm(avg_ret, "retention"),
                norm(train_speed, "train_speed"),
                norm(throughput, "throughput"),
                norm(mem_eff, "memory"),
            ]
            scores += scores[:1]

            color = MODEL_COLORS.get(mo, plt.cm.tab10.colors[idx % 10])
            ax.plot(angles, scores, linewidth=2.0, linestyle="solid", label=mo, color=color)
            ax.fill(angles, scores, color=color, alpha=0.15)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(axes_labels, fontsize=10, fontweight="bold")
        ax.set_ylim(0.1, 1.05)
        ax.set_yticklabels([])
        ax.set_title(f"{ds} (Systemic Balance)", fontsize=13, fontweight="bold", pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)

    fig.suptitle("Holistic Architecture Synthesis (5-Axis Multi-Dimensional Radar)", fontsize=15, fontweight="bold", y=1.03)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiments", nargs="+", type=Path, help="Paths to experiment directories containing results.json and robustness_results.json")
    parser.add_argument("--heatmap-output", type=Path, default=Path("experiments/robustness_heatmap.png"), help="Path for heatmap image")
    parser.add_argument("--efficiency-output", type=Path, default=Path("experiments/system_efficiency.png"), help="Path for system efficiency image")
    parser.add_argument("--fragility-output", type=Path, default=Path("experiments/noise_fragility.png"), help="Path for noise fragility asymmetry image")
    parser.add_argument("--delta-output", type=Path, default=Path("experiments/augmentation_delta.png"), help="Path for augmentation delta gain image")
    parser.add_argument("--radar-output", type=Path, default=Path("experiments/radar_synthesis.png"), help="Path for radar synthesis image")
    parser.add_argument("--metric", type=str, default="recall@20", help="Metric for heatmap (default: recall@20)")
    args = parser.parse_args()

    loaded = [load_model_data(p) for p in args.experiments]

    # 1. Heatmap
    p_hm = plot_robustness_heatmap(loaded, args.heatmap_output, metric=args.metric)
    print(f"1. Robustness Heatmap generated: {p_hm.resolve()}")

    # 2. Grouped System Efficiency with Inference Throughput
    p_eff = plot_grouped_system_efficiency(loaded, args.efficiency_output)
    print(f"2. Grouped System Efficiency generated: {p_eff.resolve()}")

    # 3. Noise Fragility Asymmetry
    p_frag = plot_noise_fragility_bars(loaded, args.fragility_output, metric=args.metric)
    print(f"3. Noise Fragility Asymmetry generated: {p_frag.resolve()}")

    # 4. Augmentation Delta
    p_delta = plot_augmentation_delta(loaded, args.delta_output)
    print(f"4. Augmentation Delta Gain generated: {p_delta.resolve()}")

    # 5. Multi-Axis Radar Synthesis
    p_radar = plot_radar_chart(loaded, args.radar_output)
    print(f"5. Radar Chart Synthesis generated: {p_radar.resolve()}")


if __name__ == "__main__":
    main()
