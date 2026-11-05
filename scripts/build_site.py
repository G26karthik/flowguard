"""Assemble the static site (GitHub Pages): web/ + the flowguard modules the browser runtime needs + the models
listed in models.json and their evidence files.

Usage: python scripts/build_site.py [out_dir=_site]
The browser runs the same flowguard.core.Registry as the FastAPI server, through Pyodide (web/worker.js).
"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "_site"
if out.exists():
    shutil.rmtree(out)
shutil.copytree(ROOT / "web", out)

registry = json.loads((ROOT / "models.json").read_text())
MODULES = ["__init__", "core", "diagnose", "fields", "guard", "infer", "npfno", "solver"]  # no torch, no FastAPI
files = [f"flowguard/{m}.py" for m in MODULES]
for v in registry.values():
    files += v["weights"] + ([v["reliability"]] if v.get("reliability") else [])
files.append("results/discretization_error.json")
for f in files:
    if not (ROOT / f).exists():
        sys.exit(f"missing {f}: train/export/evaluate first (see README)")
    (out / f).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / f, out / f)

manifest = {"files": files, "registry": registry, "discretization": "results/discretization_error.json"}
(out / "manifest.json").write_text(json.dumps(manifest, indent=1))
(out / ".nojekyll").write_text("")
index = out / "index.html"
html = index.read_text(encoding="utf-8")
if "<title>" not in html:
    sys.exit("index.html has no <title> to anchor the runtime marker")
index.write_text(html.replace("<title>", '<meta name="flowguard-runtime" content="browser">\n<title>', 1), encoding="utf-8")
print("site at", out, "files:", len(files), "models:", list(registry))
