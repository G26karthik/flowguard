"""FlowGuard HTTP API: neural-operator Darcy solutions with independent verification.

Fields travel as base64 float32, row-major [i, j] with i along x (PDEBench layout).
Run: uvicorn flowguard.api:app --port 8000
"""
import base64
import json
import os
import time
from glob import glob
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from flowguard.fields import FAMILIES
from flowguard.guard import fft_estimates
from flowguard.infer import Predictor
from flowguard.solver import solve

ROOT = Path(__file__).resolve().parents[1]
MODELS = sorted(glob(os.environ.get("FLOWGUARD_MODELS", str(ROOT / "runs/fno_steady_s*/model.pt"))))
predictor = Predictor(MODELS) if MODELS else None
_rel = sorted(glob(str(ROOT / "results/reliability_*_ens*.json")))
RELIABILITY = json.loads(Path(_rel[0]).read_text()) if _rel else None
MAX_N, MAX_BATCH, MAX_VERIFY = 256, 256, 64

app = FastAPI(title="FlowGuard")


def enc(x):
    return base64.b64encode(np.ascontiguousarray(x, dtype="<f4").tobytes()).decode()


def dec(s, n):
    try:
        x = np.frombuffer(base64.b64decode(s), dtype="<f4")
    except Exception as exc:
        raise HTTPException(400, "field is not valid base64 float32") from exc
    if x.size != n * n:
        raise HTTPException(400, f"field has {x.size} values, expected {n}x{n}")
    return x.reshape(n, n).astype(np.float64)


class FieldSpec(BaseModel):
    family: str | None = Field(None, description="pdebench | grf | channels; omit to send `values`")
    seed: int = 0
    n: int = 128
    a_low: float = 0.1
    a_high: float = 1.0
    values: str | None = Field(None, description="base64 float32 n*n coefficient field")


def make_fields(spec: FieldSpec, count, rng):
    if not 32 <= spec.n <= MAX_N or spec.n % 4:  # FNO keeps 12 modes: needs n >= 24
        raise HTTPException(400, f"n must be a multiple of 4 in [32, {MAX_N}]")
    if spec.values is not None:
        a = dec(spec.values, spec.n)[None]
    elif spec.family in FAMILIES:
        if not 0 < spec.a_low < spec.a_high <= 1e3:
            raise HTTPException(400, "need 0 < a_low < a_high <= 1000")
        kw = {"n": spec.n, "a_low": spec.a_low, "a_high": spec.a_high}
        # PDEBench thresholds bumps at the batch mean; generate in batches of 200 like the dataset
        a = np.concatenate([FAMILIES[spec.family](min(200, count - i), rng, **kw) for i in range(0, count, 200)])
    else:
        raise HTTPException(400, f"unknown family {spec.family!r}")
    if not np.all(np.isfinite(a)) or a.min() <= 0:
        raise HTTPException(400, "coefficient must be finite and strictly positive")
    return a


def input_flags(a):
    """Where this input sits relative to the training data (PDEBench: binary {0.1, 1}, 128x128)."""
    flags = []
    n = a.shape[-1]
    if n != 128:
        flags.append({"code": "resolution", "msg": f"Trained at 128x128; this field is {n}x{n}.", "eval_set": f"res{n}"})
    lo, hi = float(a.min()), float(a.max())
    if hi / lo > 10.0 + 1e-9:
        flags.append({"code": "contrast", "msg": f"Contrast {hi / lo:.0f}x exceeds the training contrast of 10x.", "eval_set": "contrast100"})
    if np.unique(np.round(a, 6)).size > 2:
        flags.append({"code": "continuous", "msg": "Training fields are two-valued; this one varies continuously.", "eval_set": "grf"})
    frac = float((a > np.sqrt(lo * hi)).mean())
    if not 0.2 <= frac <= 0.8:
        flags.append({"code": "fraction", "msg": f"High-permeability area {frac:.0%} is outside the typical training range 20-80%.", "eval_set": None})
    if RELIABILITY:
        for f in flags:
            s = RELIABILITY["sets"].get(f["eval_set"] or "")
            if s:
                f["measured_median_err"] = s["err"]["median"]
    return flags


def bound(fft_err):
    if not RELIABILITY:
        return None
    return float(RELIABILITY["conformal_q90_fft"] * fft_err)


@app.get("/api/health")
def health():
    return {"models": len(MODELS), "reliability_loaded": RELIABILITY is not None}


@app.get("/api/evidence")
def evidence():
    if not RELIABILITY:
        raise HTTPException(404, "no reliability study found")
    return RELIABILITY


class SolveReq(BaseModel):
    field: FieldSpec
    beta: float = 1.0


@app.post("/api/solve")
def solve_one(req: SolveReq):
    if predictor is None:
        raise HTTPException(503, "no trained model loaded")
    if not 0 < abs(req.beta) <= 1e4:
        raise HTTPException(400, "beta must be nonzero with |beta| <= 1e4")
    a = make_fields(req.field, 1, np.random.default_rng(req.field.seed))[0]
    t = time.perf_counter()
    members = predictor.predict(a, req.beta)
    t_nn = time.perf_counter() - t
    u_hat = members[0]
    ind, e_est = fft_estimates(a[None], req.beta, u_hat[None])
    t = time.perf_counter()
    u = solve(a, req.beta, "arith", "ghost")
    t_solve = time.perf_counter() - t
    err = float(np.linalg.norm(u_hat - u) / np.linalg.norm(u))
    spread = float(np.linalg.norm(members.std(0)) / np.linalg.norm(members.mean(0))) if len(members) > 1 else None
    return {
        "n": a.shape[0], "a": enc(a), "u_pred": enc(u_hat), "u_exact": enc(u), "err_field": enc(u_hat - u),
        "err_est_field": enc(-e_est[0]),
        "true_rel_err": err, "fft_err_rel": float(ind["fft_err_rel"][0]), "residual_rel": float(ind["residual_rel"][0]),
        "ensemble_spread": spread, "bound90": bound(float(ind["fft_err_rel"][0])),
        "flags": input_flags(a), "ms": {"nn": 1e3 * t_nn, "exact": 1e3 * t_solve},
        "qoi": {"mean_pred": float(u_hat.mean()), "mean_exact": float(u.mean()), "max_pred": float(u_hat.max()), "max_exact": float(u.max())},
    }


class BatchReq(BaseModel):
    field: FieldSpec
    count: int = 128
    beta: float = 1.0
    verify_risky: int = 8
    verify_random: int = 8


@app.post("/api/batch")
def batch(req: BatchReq):
    """Predict many fields; spend the exact-solve budget on the riskiest plus a random audit sample."""
    if predictor is None:
        raise HTTPException(503, "no trained model loaded")
    if req.field.values is not None or req.field.family is None:
        raise HTTPException(400, "batch mode needs a generator family")
    if not 1 <= req.count <= MAX_BATCH or req.verify_risky < 0 or req.verify_random < 0 or req.verify_risky + req.verify_random > min(MAX_VERIFY, req.count):
        raise HTTPException(400, f"count in [1, {MAX_BATCH}], verify budgets >= 0 and sum <= min({MAX_VERIFY}, count)")
    if not 0 < abs(req.beta) <= 1e4:
        raise HTTPException(400, "beta must be nonzero with |beta| <= 1e4")
    rng = np.random.default_rng(req.field.seed)
    a = make_fields(req.field, req.count, rng)
    t = time.perf_counter()
    u_hat = predictor.predict(a, req.beta)[0]
    t_nn = time.perf_counter() - t
    t = time.perf_counter()
    ind, _ = fft_estimates(a, req.beta, u_hat)
    t_ind = time.perf_counter() - t
    order = np.argsort(-ind["fft_err_rel"])
    risky = order[:req.verify_risky]
    rest = order[req.verify_risky:]
    audit = rng.choice(rest, size=min(req.verify_random, len(rest)), replace=False)
    t = time.perf_counter()
    true = {}
    for i in np.concatenate([risky, audit]).astype(int):
        u = solve(a[i], req.beta, "arith", "ghost")
        true[int(i)] = float(np.linalg.norm(u_hat[i] - u) / np.linalg.norm(u))
    t_solve = time.perf_counter() - t
    rows = [{"i": i, "fft_err_rel": float(ind["fft_err_rel"][i]), "bound90": bound(float(ind["fft_err_rel"][i])),
             "true_rel_err": true.get(i), "role": "risky" if i in set(risky.tolist()) else ("audit" if i in set(audit.tolist()) else None),
             "qoi_mean": float(u_hat[i].mean())} for i in range(req.count)]
    audit_err = [true[int(i)] for i in audit]
    return {
        "rows": rows, "flags": input_flags(a[0]),
        "audit": {"n": len(audit_err), "median": float(np.median(audit_err)) if audit_err else None,
                  "max": float(np.max(audit_err)) if audit_err else None},
        "ms": {"nn_total": 1e3 * t_nn, "indicators_total": 1e3 * t_ind, "exact_total": 1e3 * t_solve,
               "exact_per_solve": 1e3 * t_solve / max(1, len(true))},
        "preview": {str(i): enc(a[i]) for i in order[:4].tolist()}, "n": a.shape[-1],
    }


web = ROOT / "web"
if web.exists():
    app.mount("/", StaticFiles(directory=web, html=True), name="web")
