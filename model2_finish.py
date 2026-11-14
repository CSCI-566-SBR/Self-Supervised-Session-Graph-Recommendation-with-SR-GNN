"""Finish Model2 on team data. Run: python model2_finish.py --device cuda"""
from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from model2_io import audit_data, import_runs, prepare_data, read_json, report, write_json


def locate(explicit, filename, downloads):
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    for directory in (downloads, ROOT, ROOT / "inputs", ROOT.parent):
        path = directory / filename
        if path.is_file():
            return path
    if filename == "CSCI-566-SBsrgnn-processed-diginetica.zip":
        previous = (Path.home() / "OneDrive/Documents/Projects/DL_CSCI566/Self_Supervised_SR_GNN/Model2_Team_Diginetica")
        for parent in (previous, previous / "Model2_Team_Diginetica"):
            path = parent / "data/diginetica"
            if (path / "pipeline_manifest.json").is_file():
                return path
    raise FileNotFoundError(f"Cannot find {filename}. Put it in {downloads}, or supply its path using --diginetica, --retailrocket, or --saved-runs.")


def prepare(args):
    data = {}
    for name in ("diginetica", "retailrocket"):
        target = ROOT / "data/processed" / name
        if not target.exists():
            source = locate(getattr(args, name), f"CSCI-566-SBsrgnn-processed-{name}.zip", args.downloads)
            target = source if source.is_dir() else prepare_data(source, target)
        config = ROOT / f"configs/model2_{name}.toml"
        result = audit_data(target, config)
        write_json(ROOT / f"evidence/{name}_audit.json", result)
        data[name] = target
        print(f"PASS {name}: {result['num_items']:,} items; {result['examples']}", flush=True)
    legacy = ROOT / "experiments/laptop_diginetica"
    if not all((legacy / n / "best_model.pt").exists()
               for n in ("clean_seed2028", "model2_deletion_seed2028")):
        source = locate(args.saved_runs, "outputs.zip", args.downloads)
        if not source.is_file():
            raise ValueError("--saved-runs must point to outputs.zip")
        import_runs(source, legacy)
    print("Saved Diginetica checkpoints are ready; originals remain unchanged.", flush=True)
    return data, legacy


def run_retail(args, data):
    from recsys.model2.runner import train_run
    from recsys.model2.finish import evaluate_full
    config = ROOT / "configs/model2_retailrocket.toml"
    with config.open("rb") as f:
        base = tomllib.load(f)
    settings = [("model2_retailrocket_seed2028", .5)]
    if args.retail_clean:
        settings.insert(0, ("clean_retailrocket_local_seed2028", 0.0))
    for label, fraction in settings:
        output = ROOT / "experiments" / label
        completed_training = (output / "run.json").is_file() and read_json(output / "run.json")["status"] == "trained_not_tested"
        if not completed_training:
            values = SimpleNamespace(
                epochs=base["training"]["max_epochs"], patience=base["training"]["patience"],
                batch_size=base["training"]["batch_size"], hidden_size=base["model"]["hidden_size"],
                cpu_threads=args.cpu_threads, debug_limit=0, seed=2028,
                data=data, output=output, kind="deletion", ratio=.1, fraction=fraction,
                noise_seed=2026, deterministic=False, device=args.device,
                resume=(output / "last_training.pt").exists(),
                num_workers=args.num_workers, fused=not args.unfused)
            train_run(values, baseline_config=config)
        evaluate_full(output, data, "retailrocket", config, args.device, args.num_workers)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("all", "prepare", "metrics", "retail", "report"), default="all")
    p.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    p.add_argument("--diginetica", type=Path, help="Dataset ZIP or already-correct pipeline directory")
    p.add_argument("--retailrocket", type=Path, help="Dataset ZIP or already-correct pipeline directory")
    p.add_argument("--saved-runs", type=Path, help="Your original outputs.zip")
    p.add_argument("--downloads", type=Path, default=Path.home() / "Downloads")
    p.add_argument("--num-workers", type=int, default=0, help="0 avoids Windows multiprocessing overhead")
    p.add_argument("--cpu-threads", type=int, default=6)
    p.add_argument("--retail-clean", action="store_true", help="Also train a clean RetailRocket baseline on this machine")
    p.add_argument("--unfused", action="store_true", help="Use unfused Adam; use the same choice for matched training runs")
    p.add_argument("--skip-checks", action="store_true", help="Skip repeat neural smoke tests after an earlier successful check")
    args = p.parse_args()
    if args.num_workers < 0 or args.cpu_threads < 1:
        p.error("workers must be nonnegative and CPU threads positive")
    if args.stage == "report":
        print(report(ROOT / "experiments", ROOT / "model2_results/comparison_full.csv"))
        return
    data, legacy = prepare(args)
    if args.stage == "prepare":
        return
    if importlib.util.find_spec("torch") is None:
        raise SystemExit("Data preparation passed. Run GPU steps with the Python environment where your earlier torch CUDA doctor check passed.")
    try:
        import torch
        import matplotlib
    except ImportError as error:
        raise SystemExit(f"Missing dependency: {error}. Install matplotlib with: python -m pip install matplotlib. Keep your working CUDA torch installation.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable in this Python environment. Use the same Python that passed your original doctor check.")
    torch.set_num_threads(args.cpu_threads)
    print(f"PyTorch {torch.__version__}; device={args.device}", flush=True)
    if not args.skip_checks:
        subprocess.run([sys.executable, str(ROOT / "check_model2.py")], check=True, cwd=ROOT)
    from recsys.model2.finish import evaluate_full
    if args.stage in ("all", "metrics"):
        for name in ("clean_seed2028", "model2_deletion_seed2028"):
            evaluate_full(legacy / name, data["diginetica"], "diginetica",
                          ROOT / "configs/model2_diginetica.toml", args.device, args.num_workers)
    if args.stage in ("all", "retail"):
        run_retail(args, data["retailrocket"])
    destination = report(ROOT / "experiments", ROOT / "model2_results/comparison_full.csv")
    from plot_model2 import main as plot_completed_results
    plot_completed_results()
    print(f"FINISHED. Share the model2_results folder and the new experiment results. Table: {destination}", flush=True)


if __name__ == "__main__":
    main()
