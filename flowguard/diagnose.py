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
COST = {"contrast": 1, "grid": 1, "continuous": 1, "all": 1, "symmetry": 0.2}  # in units of (network + reference solve)


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
        b, used = a, []
        for k in ("grid", "continuous", "contrast"):  # grid first: the other edits act on the 128 grid
            nb = UNDO[k](b)
            if nb is not None:
                b, used = nb, used + [k]
        if not used:
            return {"experiment": name, "applicable": False}
        return {"experiment": name, "applicable": True, "undone": used, "err_after": network_error(predictor, b)}
    b = UNDO[name](a)
    if b is None:
        return {"experiment": name, "applicable": False}
    return {"experiment": name, "applicable": True, "err_after": network_error(predictor, b)}


def attribute(err0, evidence, id_err=0.006, ratio=1.5):
    """Fixed rule: an axis is a cause if undoing it alone cuts the error by >= `ratio`; geometry is a cause if,
    after undoing every applicable axis, the error is still > 3x the in-distribution median `id_err`."""
    causes = set()
    for e in evidence:
        if e["experiment"] in UNDO and e["applicable"] and err0 / max(e["err_after"], 1e-12) >= ratio:
            causes.add(e["experiment"])
    alle = next((e for e in evidence if e["experiment"] == "all"), None)
    if alle is not None and (not alle["applicable"] or alle["err_after"] > 3 * id_err):
        causes.add("geometry")
    return causes


def investigate(predictor, a):
    """Fixed policy: run every applicable undo experiment and the joint undo, then attribute."""
    err0 = network_error(predictor, a)
    evidence = [run_experiment(predictor, a, k) for k in ("contrast", "grid", "continuous", "all")]
    cost = sum(COST[e["experiment"]] for e in evidence if e["applicable"])
    return {"err": err0, "evidence": evidence, "causes": sorted(attribute(err0, evidence)), "cost": cost}
