"""Diagnostic experiments: why is the network wrong on this input?

Counterfactual experiments edit the input back toward the training data (PDEBench: 128x128, two values,
contrast 10) along one axis, re-run network and reference, and report how much the error drops:
  contrast  compress log-contrast to 10x, keeping the geometry
  grid      resample to 128x128 (nearest cell)
  binarize  threshold at the geometric mid-value
Whatever remains after undoing every applicable axis is attributed to geometry (shape and layout).
Label-free probe: the PDE and the square domain are symmetric under the 8 rotations/reflections, so the
spread of the network's symmetry-transformed answers measures its inconsistency without a reference solve.
"""
import numpy as np

from flowguard.solver import solve

TRAIN_N, TRAIN_CONTRAST = 128, 10.0


def _undo_contrast(a):
    hi, c = a.max(), a.max() / a.min()
    if c <= TRAIN_CONTRAST * 1.001:
        return None
    return hi * (a / hi) ** (np.log(TRAIN_CONTRAST) / np.log(c))


def _undo_grid(a):
    n = a.shape[0]
    if n == TRAIN_N:
        return None
    idx = ((np.arange(TRAIN_N) + 0.5) * n / TRAIN_N).astype(int)
    return a[np.ix_(idx, idx)]


def _undo_continuous(a):
    if np.unique(np.round(a, 6)).size <= 2:
        return None
    lo, hi = a.min(), a.max()
    return np.where(a > np.sqrt(lo * hi), hi, lo)


UNDO = {"contrast": _undo_contrast, "grid": _undo_grid, "continuous": _undo_continuous}
ORDER = ("grid", "continuous", "contrast")  # grid first: the other edits act on the 128 grid
# in units of one network run + one reference solve
COST = {"contrast": 1, "grid": 1, "continuous": 1, "all": 1, "symmetry": 0.2,
        "keep_contrast": 1, "keep_grid": 1, "keep_continuous": 1}


def _undo(a, axes):
    used = []
    for k in ORDER:
        if k in axes:
            b = UNDO[k](a)
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


def network_error(predictor, a):
    u_hat = predictor.predict(a)[0]
    u = solve(a, 1.0, predictor.face, predictor.bc)
    return float(np.linalg.norm(u_hat - u) / np.linalg.norm(u))


def run_experiment(predictor, a, name):
    """One experiment. Returns evidence dict; 'applicable' False when the input already matches training on that axis."""
    if name == "symmetry":
        return {"experiment": name, "applicable": True, "symmetry_spread": symmetry_spread(predictor, a)}
    if name == "all":
        b, used = _undo(a, UNDO)
        if not used:
            return {"experiment": name, "applicable": False}
        return {"experiment": name, "applicable": True, "undone": used, "err_after": network_error(predictor, b)}
    if name.startswith("keep_"):
        # undo every off-training axis except one: that axis's effect measured from the training-like baseline
        k = name[5:]
        if UNDO[k](a) is None:
            return {"experiment": name, "applicable": False}
        b, used = _undo(a, [x for x in UNDO if x != k])
        if not used:
            return {"experiment": name, "applicable": False, "note": "only this axis is off-training: same as the input"}
        return {"experiment": name, "applicable": True, "undone": used, "err_after": network_error(predictor, b)}
    b = UNDO[name](a)
    if b is None:
        return {"experiment": name, "applicable": False}
    return {"experiment": name, "applicable": True, "err_after": network_error(predictor, b)}


def attribute(err0, evidence, id_err=0.006, ratio=1.5):
    """Ablation rule from a training-like baseline (every off-training axis undone):
    axis k is a cause if adding it back alone raises the baseline error by >= `ratio`;
    geometry is a cause if the baseline itself is still > 3x the in-distribution median `id_err`.
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
    if base > 3 * id_err:
        causes.add("geometry")
    return causes


def investigate(predictor, a):
    """Fixed policy: the joint undo, then add back each off-training axis alone (skipped when only one is off)."""
    err0 = network_error(predictor, a)
    evidence = [run_experiment(predictor, a, "all")]
    if evidence[0]["applicable"] and len(evidence[0]["undone"]) > 1:
        evidence += [run_experiment(predictor, a, f"keep_{k}") for k in evidence[0]["undone"]]
    cost = sum(COST[e["experiment"]] for e in evidence if e["applicable"])
    return {"err": err0, "evidence": evidence, "causes": sorted(attribute(err0, evidence)), "cost": cost}
