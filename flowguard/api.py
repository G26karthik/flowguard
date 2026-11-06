"""FlowGuard HTTP API: thin FastAPI wrapper around flowguard.core.Registry (models.json).

Run: uvicorn flowguard.api:app --port 8000
FLOWGUARD_MODELS (a weights glob) replaces models.json with one PDEBench-domain model; tests use it.
FLOWGUARD_CORS: comma-separated origins allowed to call the API from another site (default: the GitHub Pages
site, which uses this server when it is reachable and falls back to computing in the browser).
"""
import json
import os
import threading
import time
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
from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
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


# Abuse limits. The API is public (CORS does not stop direct calls), the replica has 1 vCPU / 2 GiB, and the biggest
# allowed request takes ~27 s. Azure bills the time a replica is awake, not CPU, so these protect availability and
# memory; the monthly cost is bounded by max replicas = 1 and watched with scripts/azure_usage.py.
MAX_BODY = 1 << 20  # a 256x256 field is 350 KB of base64
MAX_WAITING = 4  # requests queued behind the running one; more get 503
WAIT_S = 30
BURST_S, REFILL = 120.0, 0.2  # compute seconds per client: burst, then 0.2 s per second (12 s of compute per minute)
_compute = threading.Semaphore(1)  # one computation at a time: they share one core, and two big ones would not fit in RAM
_state = threading.Lock()
_waiting = 0
_buckets = {}  # client -> [seconds left, time of last update]


def _client(request):
    # Azure's ingress appends the caller's address to X-Forwarded-For; earlier entries are whatever the caller sent
    xff = request.headers.get("x-forwarded-for")
    return xff.rsplit(",", 1)[-1].strip() if xff else (request.client.host if request.client else "unknown")


def _charge(client, seconds=0.0):
    """Refill and charge the client's compute budget; returns the seconds left (negative: over budget)."""
    now = time.monotonic()
    with _state:
        if len(_buckets) > 10_000:  # forget clients whose budget has refilled
            for k in [k for k, (s, t) in _buckets.items() if s + REFILL * (now - t) >= BURST_S]:
                del _buckets[k]
        left, t = _buckets.get(client, (BURST_S, now))
        left = min(BURST_S, left + REFILL * (now - t)) - seconds
        _buckets[client] = [left, now]
        return left


def _call(name, req, request=None):
    global _waiting
    if service is None:
        raise HTTPException(503, "no trained model loaded")
    client = _client(request) if request else None
    if client and (left := _charge(client)) <= 0:
        wait = int(-left / REFILL) + 1
        raise HTTPException(429, f"This address has used a lot of server time in the last few minutes. Try again in {wait} s, "
                                 "or compute in your browser instead (link under the results).", headers={"Retry-After": str(wait)})
    with _state:
        if _waiting >= MAX_WAITING:
            raise HTTPException(503, "The server is busy with other requests. Try again in a few seconds.", headers={"Retry-After": "10"})
        _waiting += 1
    try:
        got = _compute.acquire(timeout=WAIT_S)
    finally:
        with _state:
            _waiting -= 1
    if not got:
        raise HTTPException(503, "The server is busy with other requests. Try again in a few seconds.", headers={"Retry-After": "10"})
    t = time.perf_counter()
    try:
        return getattr(service, name)(req)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    finally:
        _compute.release()
        if client:
            _charge(client, time.perf_counter() - t)


@app.middleware("http")
async def _limit_body(request: Request, call_next):
    if request.method == "POST":
        size = request.headers.get("content-length")
        if size is None:
            return JSONResponse({"detail": "send a Content-Length header"}, status_code=411)
        if not size.isdigit() or int(size) > MAX_BODY:
            return JSONResponse({"detail": f"request body over {MAX_BODY} bytes"}, status_code=413)
    return await call_next(request)


@app.get("/api/health")
def health():
    return service.info() if service else {"models": {}}


@app.get("/api/evidence")
def evidence():
    return _call("evidence", {})


@app.post("/api/solve")
def solve_one(request: Request, req: dict = Body(...)):
    return _call("solve", req, request)


@app.post("/api/batch")
def batch(request: Request, req: dict = Body(...)):
    return _call("batch", req, request)


@app.post("/api/investigate")
def investigate(request: Request, req: dict = Body(...)):
    return _call("investigate", req, request)


web = ROOT / "web"
if web.exists():
    app.mount("/", StaticFiles(directory=web, html=True), name="web")
