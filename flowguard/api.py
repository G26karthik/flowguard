"""FlowGuard HTTP API: thin FastAPI wrapper around flowguard.core.Service.

Run: uvicorn flowguard.api:app --port 8000   (FLOWGUARD_MODELS overrides the weights glob)
"""
import json
import os
from glob import glob
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from flowguard.core import Service
from flowguard.infer import Predictor

ROOT = Path(__file__).resolve().parents[1]
MODELS = sorted(glob(os.environ.get("FLOWGUARD_MODELS", str(ROOT / "runs/fno_steady_s*/weights.npz"))))


def _load(pattern):
    hits = sorted(glob(str(ROOT / pattern)))
    return json.loads(Path(hits[0]).read_text()) if hits else None


service = Service(Predictor(MODELS), _load("results/reliability_*_ens*.json") or _load("results/reliability_*.json"),
                  _load("results/discretization_error.json")) if MODELS else None
app = FastAPI(title="FlowGuard")


def _call(fn, req):
    if service is None:
        raise HTTPException(503, "no trained model loaded")
    try:
        return fn(req)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/health")
def health():
    return service.info() if service else {"models": 0}


@app.get("/api/evidence")
def evidence():
    if not service or not service.rel:
        raise HTTPException(404, "no reliability study found")
    return service.rel


@app.post("/api/solve")
def solve_one(req: dict = Body(...)):
    return _call(service.solve if service else None, req)


@app.post("/api/batch")
def batch(req: dict = Body(...)):
    return _call(service.batch if service else None, req)


web = ROOT / "web"
if web.exists():
    app.mount("/", StaticFiles(directory=web, html=True), name="web")
