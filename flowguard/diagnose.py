"""Diagnostic experiments: why is the network wrong on this input?

Counterfactual experiments edit the input back toward the training data along one axis, re-run network and
reference, and report how much the error drops:
  contrast    compress log-contrast to the training maximum, keeping the geometry
  grid        resample to the training grid (nearest cell)
  continuous  threshold at the geometric mid-value (only for models trained on two-valued fields)
Whatever remains after undoing every applicable axis is attributed to geometry (shape and layout), either by a
fixed threshold (legacy rule) or, with the geometry experiment, by comparison with training-family shapes that
have the same two values on the same grid.
`domain` (models.json) says what "training-like" means for a model; the default is PDEBench's
(128x128, two values, contrast 10).
Label-free probe: the PDE and the square domain are symmetric under the 8 rotations/reflections, so the
spread of the network's symmetry-transformed answers measures its inconsistency without a reference solve.
"""
import numpy as np

from flowguard.fields import pdebench
from flowguard.solver import solve

TRAIN_N, TRAIN_CONTRAST = 128, 10.0
PDEBENCH = {"grid": TRAIN_N, "contrast_max": TRAIN_CONTRAST, "continuous": False}


def _undo_contrast(a, cmax=TRAIN_CONTRAST):
    hi, c = a.max(), a.max() / a.min()
    if c <= cmax * 1.001:
        return None
    return hi * (a / hi) ** (np.log(cmax) / np.log(c))


def _undo_grid(a, n_train=TRAIN_N):
    n = a.shape[0]
    if n == n_train:
        return None
    idx = ((np.arange(n_train) + 0.5) * n / n_train).astype(int)
    return a[np.ix_(idx, idx)]


def _undo_continuous(a, allowed=False):
    if allowed or np.unique(np.round(a, 6)).size <= 2:
        return None
    lo, hi = a.min(), a.max()
    return np.where(a > np.sqrt(lo * hi), hi, lo)


UNDO = {"contrast": _undo_contrast, "grid": _undo_grid, "continuous": _undo_continuous}
_DOMAIN_KEY = {"contrast": "contrast_max", "grid": "grid", "continuous": "continuous"}
ORDER = ("grid", "continuous", "contrast")  # grid first: the other edits act on the training grid
GEOMETRY_PROBES = 4
# in units of one network run + one reference solve
COST = {"contrast": 1, "grid": 1, "continuous": 1, "all": 1, "symmetry": 0.2, "geometry": GEOMETRY_PROBES,
        "keep_contrast": 1, "keep_grid": 1, "keep_continuous": 1}


def undo_axis(k, a, domain=None):
    """Move `a` back into the training domain along axis k; None when it is already inside on that axis."""
    return UNDO[k](a, (domain or PDEBENCH)[_DOMAIN_KEY[k]])


def _undo(a, axes, domain=None):
    used = []
    for k in ORDER:
        if k in axes:
            b = undo_axis(k, a, domain)
            if b is not None:
                a, used = b, used + [k]
    return a, used


def _d4(x, k):
    return np.flip(np.rot90(x, k % 4), 0) if k >= 4 else np.rot90(x, k)


def _d4_inv(x, k):
    return np.rot90(np.flip(x, 0), -(k % 4)) if k >= 4 else np.rot90(x, -k)


def symmetry_spread(predictor, a):
    """Relative spread of the network over the 8 symmetries of the square (0 for an equivariant model)."""
    preds = np.stack([_d4_inv(predictor.predict(_d4(a, k))[0], k) for k in range(8)])
    return float(np.linalg.norm(preds.std(0)) / np.linalg.norm(preds.mean(0)))


def network_errors(predictor, a):
    """Relative L2 error of the network against the reference, for a batch (B, N, N)."""
    u_hat = predictor.predict(a)[0]
    out = []
    for i in range(len(a)):
        u = solve(a[i], 1.0, predictor.face, predictor.bc)
        out.append(np.linalg.norm(u_hat[i] - u) / np.linalg.norm(u))
    return np.array(out)


def network_error(predictor, a):
    return float(network_errors(predictor, a[None])[0])


def geometry_comparators(b, k=GEOMETRY_PROBES, seed=0):
    """Training-family (PDEBench bump) shapes with the two extreme values of `b`, on its grid."""
    return pdebench(k, np.random.default_rng(seed), n=b.shape[0], a_low=float(b.min()), a_high=float(b.max()))


def run_experiment(predictor, a, name, domain=None):
    """One experiment. Returns evidence dict; 'applicable' False when the input already matches training on that axis."""
    if name == "symmetry":
        return {"experiment": name, "applicable": True, "symmetry_spread": symmetry_spread(predictor, a)}
    if name == "all":
        b, used = _undo(a, UNDO, domain)
        if not used:
            return {"experiment": name, "applicable": False}
        return {"experiment": name, "applicable": True, "undone": used, "err_after": network_error(predictor, b)}
    if name == "geometry":
        # training-like baseline, then swap its shape for training-family shapes with the same values and grid
        b, used = _undo(a, UNDO, domain)
        errs = network_errors(predictor, geometry_comparators(b))
        return {"experiment": name, "applicable": True, "undone": used, "err_after": float(np.median(errs)),
                "err_range": [float(errs.min()), float(errs.max())]}
    if name.startswith("keep_"):
        # undo every off-training axis except one: that axis's effect measured from the training-like baseline
        k = name[5:]
        if undo_axis(k, a, domain) is None:
            return {"experiment": name, "applicable": False}
        b, used = _undo(a, [x for x in UNDO if x != k], domain)
        if not used:
            return {"experiment": name, "applicable": False, "note": "only this axis is off-training: same as the input"}
        return {"experiment": name, "applicable": True, "undone": used, "err_after": network_error(predictor, b)}
    b = undo_axis(name, a, domain)
    if b is None:
        return {"experiment": name, "applicable": False}
    return {"experiment": name, "applicable": True, "err_after": network_error(predictor, b)}


def attribute(err0, evidence, id_err=0.006, ratio=1.5, geo_ratio=2.0):
    """Ablation rule from a training-like baseline (every off-training axis undone):
    axis k is a cause if adding it back alone raises the baseline error by >= `ratio`.
    Geometry: with a geometry experiment in the evidence, a cause if the baseline error is >= `geo_ratio` times
    the larger of the comparator error (training-family shapes, same values and grid) and `id_err`. Without it
    (legacy rule), a cause if the baseline is still > 3x the in-distribution median `id_err`.
    When only one axis is off-training, adding it back reproduces the input, so its error is err0."""
    ev = {e["experiment"]: e for e in evidence}
    alle = ev.get("all")
    base = alle["err_after"] if alle and alle["applicable"] else err0
    off = alle["undone"] if alle and alle["applicable"] else []
    causes = set()
    for k in off:
        kept = ev.get(f"keep_{k}")
        err_k = kept["err_after"] if kept and kept["applicable"] else (err0 if len(off) == 1 else None)
        if err_k is not None and err_k / max(base, 1e-12) >= ratio:
            causes.add(k)
    geo = ev.get("geometry")
    if geo and geo["applicable"]:
        if base >= geo_ratio * max(geo["err_after"], id_err):
            causes.add("geometry")
    elif base > 3 * id_err:
        causes.add("geometry")
    return causes


def investigate(predictor, a, domain=None, id_err=0.006, geometry=False, ratio=1.5, geo_ratio=2.0):
    """Fixed policy: the joint undo, then add back each off-training axis alone (skipped when only one is off).
    geometry=True adds the geometry experiment, skipped when the baseline error is already within `geo_ratio`
    of `id_err` (it could not then name geometry as a cause). Defaults reproduce the legacy rule."""
    err0 = network_error(predictor, a)
    evidence = [run_experiment(predictor, a, "all", domain)]
    if evidence[0]["applicable"] and len(evidence[0]["undone"]) > 1:
        evidence += [run_experiment(predictor, a, f"keep_{k}", domain) for k in evidence[0]["undone"]]
    base = evidence[0]["err_after"] if evidence[0]["applicable"] else err0
    if geometry and base >= geo_ratio * id_err:
        evidence.append(run_experiment(predictor, a, "geometry", domain))
    cost = sum(COST[e["experiment"]] for e in evidence if e["applicable"])
    causes = attribute(err0, evidence, id_err, ratio, geo_ratio)
    return {"err": err0, "baseline_err": base, "evidence": evidence, "causes": sorted(causes), "cost": cost}
