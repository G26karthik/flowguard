"""Assemble the static site (GitHub Pages): web/ + the flowguard modules the browser runtime needs + weights.

Usage: python scripts/build_site.py [out_dir=_site]
The browser runs the same flowguard.core.Service as the FastAPI server, through Pyodide (web/worker.js).
"""
import json
import shutil
import sys
from glob import glob
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "_site"
if out.exists():
    shutil.rmtree(out)
shutil.copytree(ROOT / "web", out)

MODULES = ["__init__", "core", "fields", "guard", "infer", "npfno", "solver"]  # no torch, no FastAPI
files = []
for m in MODULES:
    rel = f"flowguard/{m}.py"
    (out / "flowguard").mkdir(exist_ok=True)
    shutil.copy(ROOT / rel, out / rel)
    files.append(rel)

weights = sorted(str(Path(p).relative_to(ROOT).as_posix()) for p in glob(str(ROOT / "runs/fno_steady_s*/weights.npz")))
if not weights:
    sys.exit("no exported weights: run scripts/export_numpy.py first")
for w in weights:
    (out / w).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / w, out / w)
files += weights

rel = sorted(glob(str(ROOT / "results/reliability_*_ens*.json"))) or sorted(glob(str(ROOT / "results/reliability_*.json")))
extra = {"reliability": rel[0] if rel else None, "discretization": str(ROOT / "results/discretization_error.json")}
manifest = {"files": files, "weights": weights}
for key, p in extra.items():
    if p and Path(p).exists():
        r = Path(p).relative_to(ROOT).as_posix()
        (out / r).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(p, out / r)
        files.append(r)
        manifest[key] = r
(out / "manifest.json").write_text(json.dumps(manifest, indent=1))
index = out / "index.html"
html = index.read_text(encoding="utf-8")
if "<title>" not in html:
    sys.exit("index.html has no <title> to anchor the runtime marker")
index.write_text(html.replace("<title>", '<meta name="flowguard-runtime" content="browser">\n<title>', 1), encoding="utf-8")
(out / ".nojekyll").write_text("")
print("site at", out, "files:", len(files), "weights:", weights)
