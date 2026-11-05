"""FlowGuard HTTP API: thin FastAPI wrapper around flowguard.core.Registry (models.json).

Run: uvicorn flowguard.api:app --port 8000
FLOWGUARD_MODELS (a weights glob) replaces models.json with one PDEBench-domain model; tests use it.
FLOWGUARD_CORS: comma-separated origins allowed to call the API from another site (default: the GitHub Pages
site, which uses this server when it is reachable and falls back to computing in the browser).
"""
import json
import os
from glob import glob
from pathlib import Path


def _cpu_quota():
    """CPUs the container may use (cgroup v2 cpu.max); os.cpu_count() reports the host's cores instead."""
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()[:2]
        return None if quota == "max" else max(1, int(quota) // int(period))
    except (OSError, ValueError):
        return None


# BLAS threads must match the CPU quota, set before NumPy loads: with 10 threads on a 1-vCPU quota the network
# took 200 ms per field, with 1 thread 43 ms (measured in Docker, --cpus=1). FLOWGUARD_THREADS overrides.
_threads = os.environ.get("FLOWGUARD_THREADS") or _cpu_quota()
if _threads:
    for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, str(_threads))

import numpy as np  # noqa: E402
from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
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
if service:  # warm up: first-call costs (DFT bases, SciPy imports) are paid at startup, not by the first user
    for s in service.services.values():
        s.p.predict(np.ones((128, 128)))
app = FastAPI(title="FlowGuard")
app.add_middleware(GZipMiddleware, minimum_size=2048, compresslevel=5)
app.add_middleware(CORSMiddleware, allow_origins=os.environ.get("FLOWGUARD_CORS", "https://g26karthik.github.io").split(","),
                   allow_methods=["GET", "POST"], allow_headers=["Content-Type"], max_age=3600)


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
