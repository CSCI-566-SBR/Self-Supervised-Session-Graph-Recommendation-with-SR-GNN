"""Create a separate Model2 branch and copy only code, JSON/CSV results and plots.

Run on the laptop: python publish_model2.py --push
Git must already be installed. Git handles its normal GitHub authentication.
No datasets/checkpoints are copied. No force-push, merge, or global Git changes.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import math
from pathlib import Path
import shutil
import subprocess
import sys
from urllib.parse import quote

REPOSITORY = "https://github.com/CSCI-566-SBR/Self-Supervised-Session-Graph-Recommendation-with-SR-GNN.git"
BRANCH = "feature/model2-fixed-noise-results"
ROOT_FILES = ["model2_finish.py", "model2_io.py", "check_model2.py", "plot_model2.py",
              "START_HERE.txt", "SOURCE_REVISION.json"]
RUNS = ["laptop_diginetica/clean_seed2028", "laptop_diginetica/model2_deletion_seed2028",
        "model2_retailrocket_seed2028"]
JSON_FILES = ["results.json", "robustness_results.json", "history.json", "run.json"]


def command(args, cwd=None, capture=False):
    return subprocess.run(args, cwd=cwd, check=True, text=True,
                          stdout=subprocess.PIPE if capture else None).stdout


def validate_source(source):
    required = [*ROOT_FILES, "src/recsys/model2/finish.py", "src/recsys/model2/runner.py",
                "src/recsys/model2/data.py", "src/recsys/model2/__init__.py",
                "configs/model2_diginetica.toml", "configs/model2_retailrocket.toml",
                "tests/test_model2_finish_io.py", "tests/test_model2_finish_neural.py",
                "model2_results/comparison_full.csv", "model2_results/comparison_full.json"]
    required += [f"experiments/{run}/{name}" for run in RUNS for name in JSON_FILES]
    missing = [name for name in required if not (source / name).is_file()]
    if missing:
        raise ValueError("Required source files are missing:\n" + "\n".join(missing))
    with (source / "model2_results/comparison_full.csv").open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    keys = [(r["run"].replace("\\", "/"), r["condition"]) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("The comparison CSV contains duplicate run/condition rows")
    for run in RUNS:
        if sum(name == run for name, _ in keys) != 10:
            raise ValueError(f"Expected ten evaluated conditions for {run}")
    for row in rows:
        for metric in ("recall", "mrr", "ndcg"):
            low, high = float(row[f"{metric}@10"]), float(row[f"{metric}@20"])
            if not (math.isfinite(low) and math.isfinite(high) and 0 <= low <= high <= 1):
                raise ValueError("The CSV contains a missing or invalid ranking metric")
    return rows


def stage_files(source, target):
    copied = []
    relative_files = [Path(name) for name in ROOT_FILES]
    for folder, pattern in [("src/recsys/model2", "*.py"), ("configs", "model2_*.toml"),
                            ("tests", "test_model2_finish_*.py")]:
        relative_files += [f.relative_to(source) for f in sorted((source / folder).glob(pattern))]
    relative_files += [Path("experiments") / run / name for run in RUNS for name in JSON_FILES]
    relative_files += [f.relative_to(source) for f in sorted((source / "model2_results").iterdir())
                       if f.is_file() and f.suffix.lower() in {".csv", ".json", ".png", ".pdf"}]
    for relative in relative_files:
        original = source / relative
        if original.is_symlink() or original.stat().st_size > 20 * 1024 * 1024:
            raise ValueError(f"Unexpected symlink or oversized upload: {relative}")
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, destination)
        copied.append(relative.as_posix())
    publisher = target / "publish_model2.py"
    shutil.copy2(Path(__file__).resolve(), publisher)
    copied.append("publish_model2.py")
    command(["git", "add", "--", *copied], cwd=target)
    staged = command(["git", "diff", "--cached", "--name-only"], cwd=target, capture=True).splitlines()
    if not staged:
        raise ValueError("No new changes to commit")
    for name in staged:
        if name.startswith("data/") or Path(name).suffix.lower() in {".pt", ".pkl", ".zip"}:
            raise ValueError(f"Unexpected data/checkpoint staged: {name}")
    return staged


def publish(source, target, branch=BRANCH, push=False, repository=REPOSITORY):
    source, target = Path(source).resolve(), Path(target).resolve()
    if shutil.which("git") is None:
        raise RuntimeError("Git is not installed or is not on PATH. Install Git through your usual approved setup, reopen PowerShell, and rerun.")
    rows = validate_source(source)
    if target.exists():
        raise FileExistsError(f"Choose a NEW --repo-dir; refusing to overwrite {target}")
    print(f"Validated {len(rows)} CSV rows. Cloning the sasrec branch into {target}...", flush=True)
    command(["git", "clone", "--branch", "sasrec", repository, str(target)])
    existing = command(["git", "ls-remote", "--heads", "origin", f"refs/heads/{branch}"],
                       cwd=target, capture=True).strip()
    if existing:
        raise RuntimeError("That remote branch already exists. Rerun with a different --branch and a new --repo-dir; no existing branch was changed.")
    command(["git", "switch", "-c", branch], cwd=target)
    staged = stage_files(source, target)
    print("Running the same local Model2 checks before committing...", flush=True)
    command([sys.executable, "check_model2.py"], cwd=target)
    command(["git", "diff", "--cached", "--stat"], cwd=target)
    if not push:
        print(f"Prepared {len(staged)} changed files on {branch}. No commit or push was made.")
        print(f"Folder: {target}")
        print('Next: git commit -m "Add fixed-noise SR-GNN experiments and results for both datasets"')
        print(f"Then: git push -u origin {branch}")
        return target
    command(["git", "commit", "-m", "Add fixed-noise SR-GNN experiments and results for both datasets"], cwd=target)
    command(["git", "push", "-u", "origin", f"HEAD:refs/heads/{branch}"], cwd=target)
    print(f"SUCCESS: pushed {branch}. Main and sasrec were not changed.")
    if repository == REPOSITORY:
        repo_url = REPOSITORY.removesuffix(".git")
        print("Open a pull request for team review:")
        print(f"{repo_url}/compare/sasrec...{quote(branch, safe='')}?expand=1")
        print("Base the PR on sasrec while its fixes are unmerged; use main once those fixes have merged.")
    return target


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=Path.home() / "Downloads/Model2_Updated_Run/Model2_Updated")
    p.add_argument("--repo-dir", type=Path)
    p.add_argument("--branch", default=BRANCH)
    p.add_argument("--push", action="store_true", help="Also commit and push the new branch to GitHub")
    args = p.parse_args()
    target = args.repo_dir or Path.home() / "Downloads" / ("Model2_GitHub_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    try:
        publish(args.source, target, args.branch, args.push)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"Stopped: {error}\nYour original training folder and results were not changed.")


if __name__ == "__main__":
    main()
