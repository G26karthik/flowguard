"""Files a deployment needs, from models.json: each served model's weights and reliability study, plus the
discretisation study. Used by the Docker build (server) and scripts/build_site.py (browser runtime).

Usage: python scripts/served_files.py OUT_DIR   copies flowguard/, web/, models.json and those files into OUT_DIR
"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def served_files(root=ROOT):
    registry = json.loads((root / "models.json").read_text())
    files = []
    for v in registry.values():
        files += v["weights"] + ([v["reliability"]] if v.get("reliability") else [])
    files.append("results/discretization_error.json")
    missing = [f for f in files if not (root / f).exists()]
    if missing:
        sys.exit(f"missing {missing}: train/export/evaluate first (see README)")
    return registry, files


if __name__ == "__main__":
    out = Path(sys.argv[1])
    _, files = served_files()
    for d in ("flowguard", "web"):
        shutil.copytree(ROOT / d, out / d, ignore=shutil.ignore_patterns("__pycache__"))
    for f in files + ["models.json"]:
        (out / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / f, out / f)
    print("bundle at", out, "data files:", files)
