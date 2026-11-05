"""FlowGuard service logic, independent of the transport: FastAPI (api.py) and Pyodide (web) both call it.

Fields travel as base64 float32, row-major [i, j] with i along x (PDEBench layout).
'Reference' below means the finite-volume solution on the same grid with the discretisation the network
was trained on: an approximation of the PDE, not its exact solution (see results/discretization_error.json).
Invalid input raises ValueError.
"""
import base64
import hashlib
import threading
import time
from collections import OrderedDict

import numpy as np

from flowguard.diagnose import investigate
from flowguard.fields import FAMILIES
from flowguard.guard import fft_estimates
from flowguard.solver import solve

MAX_N, MAX_BATCH, MAX_VERIFY = 256, 256, 64


def enc(x):
    return base64.b64encode(np.ascontiguousarray(x, dtype="<f4").tobytes()).decode()


def dec(s, n):
    try:
        x = np.frombuffer(base64.b64decode(s, validate=True), dtype="<f4")
    except Exception as exc:
        raise ValueError("field is not valid base64 float32") from exc
    if x.size != n * n:
        raise ValueError(f"field has {x.size} values, expected {n}x{n}")
    return x.reshape(n, n).astype(np.float64)


def _num(d, key, default, lo, hi, kind=float):
    v = d.get(key, default)
    try:
        v = kind(v)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be a number") from exc
    if not lo <= v <= hi:
        raise ValueError(f"{key} must be in [{lo}, {hi}]")
    return v


def make_fields(field, count, rng):
    n = _num(field, "n", 128, 32, MAX_N, int)  # FNO keeps 12 modes: needs n >= 24
    if n % 4:
        raise ValueError("n must be a multiple of 4")
    if field.get("values") is not None:
        a = dec(field["values"], n)[None]
    elif field.get("family") in FAMILIES:
        a_low = _num(field, "a_low", 0.1, 1e-4, 1e3)
        a_high = _num(field, "a_high", 1.0, 1e-4, 1e3)
        if not a_low < a_high:
            raise ValueError("need a_low < a_high")
        kw = {"n": n, "a_low": a_low, "a_high": a_high}
        # PDEBench thresholds bumps at the batch mean; generate in batches of 200 like the dataset
        a = np.concatenate([FAMILIES[field["family"]](min(200, count - i), rng, **kw) for i in range(0, count, 200)])
    else:
        raise ValueError(f"unknown family {field.get('family')!r}; send 'family' or 'values'")
    if not np.all(np.isfinite(a)) or a.min() <= 0:
        raise ValueError("coefficient must be finite and strictly positive")
    # fields travel as float32: solve exactly the field the client holds, so a round trip hits the cache
    return a.astype(np.float32).astype(np.float64)


PDEBENCH_DOMAIN = {"grid": 128, "contrast_max": 10, "continuous": False, "eval_sets": {"contrast": "contrast100", "continuous": "grf"}}


def input_flags(a, reliability, domain=PDEBENCH_DOMAIN):
    """Where this input sits relative to the model's training data (`domain`, from models.json)."""
    flags = []
    n = a.shape[-1]
    sets = domain.get("eval_sets", {})
    if n != domain["grid"]:
        flags.append({"code": "resolution", "msg": f"Trained at {domain['grid']}x{domain['grid']}; this field is {n}x{n}.", "eval_set": f"res{n}"})
    lo, hi = float(a.min()), float(a.max())
    if hi / lo > domain["contrast_max"] * (1 + 1e-9):
        flags.append({"code": "contrast", "msg": f"Contrast {hi / lo:.0f}x exceeds the training range (up to {domain['contrast_max']:g}x).",
                      "eval_set": sets.get("contrast")})
    if not domain["continuous"] and np.unique(np.round(a, 6)).size > 2:
        flags.append({"code": "continuous", "msg": "Training fields are two-valued; this one varies continuously.", "eval_set": sets.get("continuous")})
    frac = float((a > np.sqrt(lo * hi)).mean())
    if not 0.2 <= frac <= 0.8:
        flags.append({"code": "fraction", "msg": f"High-permeability area {frac:.0%} is outside the typical training range 20-80%.", "eval_set": None})
    if reliability:
        for f in flags:
            s = reliability["sets"].get(f["eval_set"] or "")
            if s:
                f["measured_median_err"] = s["err"]["median"]
    return flags


class _LRU:
    """Small thread-safe LRU cache. get() returns (value, was_cached); a miss computes outside the lock."""

    def __init__(self, size):
        self.size, self.d, self.lock = size, OrderedDict(), threading.Lock()

    def get(self, key, make):
        with self.lock:
            if key in self.d:
                self.d.move_to_end(key)
                return self.d[key], True
        v = make()
        with self.lock:
            self.d[key] = v
            while len(self.d) > self.size:
                self.d.popitem(last=False)
        return v, False


def _field_key(a):
    return hashlib.sha1(np.ascontiguousarray(a, dtype=np.float64).tobytes()).hexdigest() + str(a.shape)


_REFERENCE = _LRU(32)  # unit-source reference solutions, shared by models with the same discretisation


class Service:
    def __init__(self, predictor, reliability=None, discretization=None, domain=PDEBENCH_DOMAIN):
        self.p = predictor
        self.rel = reliability
        self.disc = discretization  # results/discretization_error.json, shown next to the reference
        self.domain = domain
        self._pred_cache = _LRU(32)  # unit-source network outputs (members) per field

    def _bound(self, est):
        return None if not self.rel else float(self.rel["conformal_q90_fft"] * est)

    def _beta(self, req):
        beta = _num(req, "beta", 1.0, -1e4, 1e4)
        if beta == 0:
            raise ValueError("beta must be nonzero")
        return beta

    def info(self):
        return {"models": len(self.p.models), "discretisation": [self.p.face, self.p.bc],
                "reliability_loaded": self.rel is not None, "discretization_study": self.disc}

    def _field(self, req):
        field = req.get("field") or {}
        a = make_fields(field, 1, np.random.default_rng(_num(field, "seed", 0, 0, 2**32 - 1, int)))[0]
        return a, field.get("values") is None

    def solve(self, req):
        """One field. Network and reference results are cached per field at unit source: the problem is linear in
        beta (u_beta = beta u_1, exactly, for the network as served and for the solver), so a beta change or a model
        switch does not redo work that does not depend on it. The response omits the coefficient field when the
        client sent it, and the measured error field (u_pred - u_ref), which the client computes."""
        a, generated = self._field(req)
        beta = self._beta(req)
        key = _field_key(a)
        t = time.perf_counter()
        members, nn_cached = self._pred_cache.get(key, lambda: self.p.predict(a, 1.0))
        t_nn = time.perf_counter() - t
        t = time.perf_counter()
        u1, ref_cached = _REFERENCE.get((self.p.face, self.p.bc, key), lambda: solve(a, 1.0, self.p.face, self.p.bc))
        t_ref = time.perf_counter() - t
        members, u = members * beta, u1 * beta
        u_hat = members[0]
        ind, e_est = fft_estimates(a[None], beta, u_hat[None], self.p.face, self.p.bc)
        err = float(np.linalg.norm(u_hat - u) / np.linalg.norm(u))
        spread = float(np.linalg.norm(members.std(0)) / np.linalg.norm(members.mean(0))) if len(members) > 1 else None
        out = {
            "n": a.shape[0], "beta": beta, "u_pred": enc(u_hat), "u_ref": enc(u),
            "err_est_field": enc(-e_est[0]), "discretisation": [self.p.face, self.p.bc],
            "rel_err_vs_ref": err, "fft_err_rel": float(ind["fft_err_rel"][0]), "residual_rel": float(ind["residual_rel"][0]),
            "ensemble_spread": spread, "bound90": self._bound(float(ind["fft_err_rel"][0])),
            "flags": input_flags(a, self.rel, self.domain),
            "ms": {"nn": 1e3 * t_nn, "reference": 1e3 * t_ref}, "cached": {"nn": nn_cached, "reference": ref_cached},
            "reference_discretization_err": ((self.disc or {}).get(f"pixel_{self.p.face}_{self.p.bc}") or {}).get("rel_128_vs_512_median"),
            "qoi": {"mean_pred": float(u_hat.mean()), "mean_ref": float(u.mean()), "max_pred": float(u_hat.max()), "max_ref": float(u.max())},
        }
        if generated:
            out["a"] = enc(a)
        return out

    def investigate(self, req):
        """Counterfactual experiments on this field (flowguard.diagnose): which off-training property of the input
        explains the network's error. Every conclusion is a measured error on an edited input."""
        a, _ = self._field(req)
        test = ((self.rel or {}).get("sets") or {}).get("pdebench_test")
        id_err = test["err"]["median"] if test else 0.006
        t = time.perf_counter()
        r = investigate(self.p, a, domain=self.domain, id_err=id_err, geometry=True)
        return {**r, "id_err": id_err, "domain": {k: self.domain[k] for k in ("grid", "contrast_max", "continuous")},
                "ms": 1e3 * (time.perf_counter() - t)}

    def batch(self, req):
        """Predict many fields; spend the reference-solve budget on the riskiest plus a random audit sample."""
        field = req.get("field") or {}
        if field.get("values") is not None or field.get("family") is None:
            raise ValueError("batch mode needs a generator family")
        count = _num(req, "count", 128, 1, MAX_BATCH, int)
        k_risky = _num(req, "verify_risky", 8, 0, MAX_VERIFY, int)
        k_audit = _num(req, "verify_random", 8, 0, MAX_VERIFY, int)
        if k_risky + k_audit > min(MAX_VERIFY, count):
            raise ValueError(f"verify budgets must sum to at most min({MAX_VERIFY}, count)")
        beta = self._beta(req)
        rng = np.random.default_rng(_num(field, "seed", 0, 0, 2**32 - 1, int))
        a = make_fields(field, count, rng)
        t = time.perf_counter()
        u_hat = self.p.predict(a, beta)[0]
        t_nn = time.perf_counter() - t
        t = time.perf_counter()
        ind, _ = fft_estimates(a, beta, u_hat, self.p.face, self.p.bc)
        t_ind = time.perf_counter() - t
        order = np.argsort(-ind["fft_err_rel"])
        risky = order[:k_risky]
        rest = order[k_risky:]
        audit = rng.choice(rest, size=min(k_audit, len(rest)), replace=False)
        t = time.perf_counter()
        ref_err = {}
        for i in np.concatenate([risky, audit]).astype(int):
            u = solve(a[i], beta, self.p.face, self.p.bc)
            ref_err[int(i)] = float(np.linalg.norm(u_hat[i] - u) / np.linalg.norm(u))
        t_ref = time.perf_counter() - t
        risky_set, audit_set = set(risky.tolist()), set(audit.tolist())
        rows = [{"i": i, "fft_err_rel": float(ind["fft_err_rel"][i]), "bound90": self._bound(float(ind["fft_err_rel"][i])),
                 "rel_err_vs_ref": ref_err.get(i), "role": "risky" if i in risky_set else ("audit" if i in audit_set else None),
                 "qoi_mean": float(u_hat[i].mean())} for i in range(count)]
        audit_err = [ref_err[int(i)] for i in audit]
        return {
            "rows": rows, "flags": input_flags(a[0], self.rel, self.domain),
            "audit": {"n": len(audit_err), "median": float(np.median(audit_err)) if audit_err else None,
                      "max": float(np.max(audit_err)) if audit_err else None},
            "ms": {"nn_total": 1e3 * t_nn, "indicators_total": 1e3 * t_ind, "reference_total": 1e3 * t_ref,
                   "reference_per_solve": 1e3 * t_ref / max(1, len(ref_err))},
            "n": a.shape[-1],
        }


class Registry:
    """Several served models (models.json): each request names one with "model"; the first entry is the default."""

    def __init__(self, registry, read_json, discretization=None, path=lambda p: p):
        from flowguard.infer import Predictor

        self.meta = {k: {"label": v["label"], "description": v["description"], "domain": v["domain"]} for k, v in registry.items()}
        self.services = {k: Service(Predictor([path(w) for w in v["weights"]]), read_json(v.get("reliability")), discretization, v["domain"])
                         for k, v in registry.items()}
        self.default = next(iter(registry))

    def _pick(self, req):
        name = req.get("model") or self.default
        if name not in self.services:
            raise ValueError(f"unknown model {name!r}; available: {sorted(self.services)}")
        return name, self.services[name]

    def solve(self, req):
        name, s = self._pick(req)
        return {**s.solve(req), "model": name}

    def batch(self, req):
        name, s = self._pick(req)
        return {**s.batch(req), "model": name}

    def investigate(self, req):
        name, s = self._pick(req)
        return {**s.investigate(req), "model": name}

    def evidence(self, req=None):
        return {"default": self.default,
                "models": {k: {**self.meta[k], "reliability": s.rel} for k, s in self.services.items()}}

    def info(self):
        return {"default": self.default, "models": {k: {**self.meta[k], **s.info()} for k, s in self.services.items()}}
