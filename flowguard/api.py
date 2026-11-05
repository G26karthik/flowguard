"""FlowGuard HTTP API: thin FastAPI wrapper around flowguard.core.Registry (models.json).

Run: uvicorn flowguard.api:app --port 8000
FLOWGUARD_MODELS (a weights glob) replaces models.json with one PDEBench-domain model; tests use it.
"""
import json
import os
from glob import glob
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from flowguard.core import PDEBENCH_DOMAIN, Registry

ROOT = Path(__file__).resolve().parents[1]


def _read(p):
    return json.loads((ROOT / p).read_text()) if p and (ROOT / p).exists() else None


if os.environ.get("FLOWGUARD_MODELS"):
    registry = {"default": {"label": "Model", "description": "", "weights": sorted(glob(os.environ["FLOWGUARD_MODELS"])),
                            "reliability": None, "domain": PDEBENCH_DOMAIN}}
else:
    registry = json.loads((ROOT / "models.json").read_text())
registry = {k: v for k, v in registry.items() if v["weights"] and all((ROOT / w).exists() for w in v["weights"])}
service = Registry(registry, _read, _read("results/discretization_error.json"), path=lambda w: str(ROOT / w)) if registry else None
app = FastAPI(title="FlowGuard")


def _call(name, req):
    if service is None:
        raise HTTPException(503, "no trained model loaded")
    try:
        return getattr(service, name)(req)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/health")
def health():
    return service.info() if service else {"models": {}}


@app.get("/api/evidence")
def evidence():
    return _call("evidence", {})


@app.post("/api/solve")
def solve_one(req: dict = Body(...)):
    return _call("solve", req)


@app.post("/api/batch")
def batch(req: dict = Body(...)):
    return _call("batch", req)


@app.post("/api/investigate")
def investigate(req: dict = Body(...)):
    return _call("investigate", req)


web = ROOT / "web"
if web.exists():
    app.mount("/", StaticFiles(directory=web, html=True), name="web")
