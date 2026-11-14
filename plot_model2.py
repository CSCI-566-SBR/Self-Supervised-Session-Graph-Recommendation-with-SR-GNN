"""Reproducible scientific plots from saved completed result JSON files."""
from pathlib import Path
import json


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    root = Path(__file__).resolve().parent
    output = root / "model2_results"
    output.mkdir(exist_ok=True)
    styles = ("-o", "--s", "-^", "--D", "-v", "--x")
    for dataset in ("diginetica", "retailrocket"):
        runs = []
        for p in sorted((root / "experiments").rglob("robustness_results.json")):
            meta_path = p.parent / "results.json"
            if not meta_path.exists(): continue
            meta = json.loads(meta_path.read_text())
            if meta.get("dataset") != dataset: continue
            label = p.parent.name.replace("_global_recursive", "")
            runs.append((label, json.loads(p.read_text())))
        if not runs: continue
        fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
        for col, kind in enumerate(("deletion", "mixed", "substitution")):
            for row, metric in enumerate(("recall@20", "mrr@20")):
                ax = axes[row, col]
                for index, (label, values) in enumerate(runs):
                    names = ["clean"] + [f"{kind}_{r:.2f}" for r in (.05, .1, .2)]
                    ax.plot([0, 5, 10, 20], [values[n][metric] * 100 for n in names],
                            styles[index % len(styles)], label=label, linewidth=1.6, markersize=4)
                ax.set_title(kind.title())
                ax.set_ylabel(metric.upper() + " (%)")
                ax.grid(alpha=.25)
                ax.set_xticks([0, 5, 10, 20])
                if row == 1: ax.set_xlabel("Test corruption (%)")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=7)
        fig.suptitle(f"{dataset.title()} robustness — single seed, saved result files")
        fig.tight_layout(rect=(0, .11, 1, .96))
        for extension in ("png", "pdf"):
            fig.savefig(output / f"{dataset}_robustness.{extension}", dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(output / f"{dataset}_robustness.png")


if __name__ == "__main__": main()
