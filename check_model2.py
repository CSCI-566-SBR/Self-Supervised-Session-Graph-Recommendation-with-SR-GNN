"""Run frozen-data, ranking and neural integration tests; no raw CSV/pandas imports."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parent
env = os.environ.copy()
env["PYTHONPATH"] = os.pathsep.join([str(root), str(root / "src"), str(root / "tests"), env.get("PYTHONPATH", "")])
tests = ["test_model2_finish_io"]
if importlib.util.find_spec("torch"):
    tests += ["test_model2_finish_neural", "test_evaluation", "test_session_graph", "test_srgnn", "test_training_smoke"]
else:
    print("PyTorch absent: running preparation/report tests only; neural checks have NOT run.", flush=True)
raise SystemExit(subprocess.run([sys.executable, "-m", "unittest", "-v", *tests], cwd=root, env=env).returncode)
