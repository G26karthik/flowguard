"""Assemble the static site (GitHub Pages): web/ + the flowguard modules the browser runtime needs + the models
listed in models.json and their evidence files.

Usage: [FLOWGUARD_API=https://host/] python scripts/build_site.py [out_dir=_site]
The browser runs the same flowguard.core.Registry as the FastAPI server, through Pyodide (web/worker.js).
"""
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "_site"
if out.exists():
    shutil.rmtree(out)
shutil.copytree(ROOT / "web", out)

sys.path.insert(0, str(ROOT / "scripts"))
from served_files import served_files  # noqa: E402

registry, data_files = served_files(ROOT)
MODULES = ["__init__", "core", "diagnose", "fields", "guard", "infer", "npfno", "solver"]  # no torch, no FastAPI
files = [f"flowguard/{m}.py" for m in MODULES] + data_files
for f in files:
    (out / f).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / f, out / f)

manifest = {"files": files, "registry": registry, "discretization": "results/discretization_error.json"}
(out / "manifest.json").write_text(json.dumps(manifest, indent=1))
(out / ".nojekyll").write_text("")
index = out / "index.html"
html = index.read_text(encoding="utf-8")
if "<title>" not in html:
    sys.exit("index.html has no <title> to anchor the runtime marker")
# FLOWGUARD_API (a server URL ending in /): the page uses that server when it answers, else computes in the browser
api = os.environ.get("FLOWGUARD_API", "").strip()
marker = '<meta name="flowguard-runtime" content="browser">\n' + (f'<meta name="flowguard-api" content="{api}">\n' if api else "")
index.write_text(html.replace("<title>", marker + "<title>", 1), encoding="utf-8")
print("site at", out, "files:", len(files), "models:", list(registry))
