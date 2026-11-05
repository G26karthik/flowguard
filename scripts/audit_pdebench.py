"""Audit PDEBench Darcy labels against an independent steady-state solve.

Usage: python scripts/audit_pdebench.py data/raw/2D_DarcyFlow_beta1.0_Train.hdf5 [n_samples]
Writes a JSON summary next to results/audit_<file>.json.
"""
import json
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flowguard.solver import residual, solve  # noqa: E402

path = Path(sys.argv[1])
k = int(sys.argv[2]) if len(sys.argv) > 2 else 100

with h5py.File(path, "r") as h:
    layout = {key: [list(h[key].shape), str(h[key].dtype)] for key in h.keys()}
    attrs = {key: float(np.asarray(v)) for key, v in h.attrs.items()}
    print("keys", layout, "attrs", attrs)
    beta = attrs.get("beta")
    n_total = h["nu"].shape[0]
    idx = np.sort(np.random.default_rng(0).choice(n_total, size=min(k, n_total), replace=False))
    nu = h["nu"][idx].astype(np.float64)
    u = h["tensor"][idx][:, 0].astype(np.float64)
    xc = h["x-coordinate"][:]

print("x-coordinate first/last", xc[:2], xc[-1], "n", len(xc))
print("unique nu values", np.unique(nu)[:10], "fraction high", float((nu > 0.5).mean()))
print("label stats min/max/mean", u.min(), u.max(), u.mean())
print("label boundary mean |u| (edge cells)", float(np.abs(np.concatenate([u[:, 0], u[:, -1], u[:, :, 0], u[:, :, -1]], 1)).mean()))

out = {"file": path.name, "layout": layout, "attrs": attrs, "n_checked": len(idx)}
for face in ("arith", "harm"):
    for bc in ("face", "ghost"):
        rel, resl = [], []
        for a_i, u_i in zip(nu, u):
            s = solve(a_i, beta, face, bc)
            rel.append(np.linalg.norm(u_i - s) / np.linalg.norm(s))
            resl.append(np.linalg.norm(residual(a_i, u_i, beta, face, bc)) / (beta * u_i.shape[0]))
        rel, resl = np.array(rel), np.array(resl)
        out[f"{face}/{bc}"] = {
            "rel_l2_label_vs_steady": {"median": float(np.median(rel)), "p90": float(np.quantile(rel, 0.9)), "max": float(rel.max())},
            "label_residual_rel": {"median": float(np.median(resl)), "max": float(resl.max())},
        }
        print(face, bc, out[f"{face}/{bc}"])

# best-matching discretisation: also check whether label = steady * c (scale mismatch) or steady + smooth offset
s = np.stack([solve(a_i, beta, "arith", "face") for a_i in nu])
c = float((u * s).sum() / (s * s).sum())
out["best_scale_factor_label_over_steady"] = c
out["rel_l2_after_scaling"] = float(np.median(np.linalg.norm((u - c * s).reshape(len(u), -1), axis=1) / np.linalg.norm(s.reshape(len(u), -1), axis=1)))
print("scale factor", c, "rel after scaling", out["rel_l2_after_scaling"])

Path("results").mkdir(exist_ok=True)
Path(f"results/audit_{path.stem}.json").write_text(json.dumps(out, indent=2))
np.savez_compressed(f"results/audit_{path.stem}_samples.npz", idx=idx, nu=nu[:8], label=u[:8], steady=s[:8])
