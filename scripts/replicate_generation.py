"""Replay PDEBench's Darcy generator to test how its labels were produced.

PDEBench (data_gen_NLE/ReactionDiffusionEq) integrates u_t = -A u + beta from an initial condition that is a
constant u0 ~ U(0.1, 10) plus a small random perturbation, and stores a snapshot at a fixed time T. With the
matched operator A (arith faces, zero ghost cells), the snapshot is
    u(T) = beta u1 + exp(-T A) (u0 * 1 + delta - beta u1),        u1 = A^-1 1.
Ignoring delta, label - beta u1 + beta exp(-TA) u1 = u0 * exp(-TA) 1, so u0 can be fitted per sample.
If the mechanism and T are right: the fit explains most of the label's deviation from steady state, the fitted
u0 look like U(0.1, 10), and the same sample gives the same u0 in every beta file (same JAX keys 2020-2069).

Usage: python scripts/replicate_generation.py [n_samples]  -> results/replicate_generation.json
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np
from scipy.stats import kstest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flowguard.solver import assemble, propagate  # noqa: E402

TIMES = [1.75, 2.0]
BETAS = ["0.01", "1.0", "100.0"]


def _work(args):
    nu, u1 = args
    A = assemble(nu, "arith", "ghost")
    n = nu.size
    res = propagate(A, np.stack([np.ones(n), u1.ravel()], 1), TIMES)
    return [(r[:, 0], r[:, 1]) for r in res]  # (exp(-TA) 1, exp(-TA) u1) per T


if __name__ == "__main__":
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    idx = np.sort(np.random.default_rng(7).choice(10000, k, replace=False))
    u1 = np.load("data/cache/2D_DarcyFlow_beta1.0_Train_steady.npy", mmap_mode="r")[idx].astype(np.float64)
    labels = {}
    for b in BETAS:
        with h5py.File(f"data/raw/2D_DarcyFlow_beta{b}_Train.hdf5", "r") as h:
            nu = h["nu"][idx].astype(np.float64)
            labels[b] = h["tensor"][idx, 0].astype(np.float64)
    with Pool(6) as pool:
        prop = pool.map(_work, list(zip(nu, u1)))

    out = {"n": k, "times": TIMES, "per_beta": {}}
    u0_fit = {}
    for b in BETAS:
        beta = float(b)
        for ti, T in enumerate(TIMES):
            u0s, explained, devs = [], [], []
            for s in range(k):
                w1, v = prop[s][ti]
                lab = labels[b][s].ravel()
                steady = beta * u1[s].ravel()
                y = lab - steady + beta * v
                u0 = float(w1 @ y / (w1 @ w1))
                recon = steady + u0 * w1 - beta * v
                dev = np.linalg.norm(lab - steady)
                u0s.append(u0)
                explained.append(1 - np.linalg.norm(lab - recon) / dev)
                devs.append(dev / np.linalg.norm(steady))
            u0s, explained, devs = map(np.array, (u0s, explained, devs))
            ok = devs > 1e-2  # deviation above the float32 / perturbation floor: u0 identifiable
            u0_fit[(b, T)] = (u0s, ok)
            out["per_beta"][f"beta={b} T={T}"] = {
                "n_identifiable": int(ok.sum()),
                "median_fraction_of_deviation_explained": float(np.median(explained[ok])) if ok.any() else None,
                "fitted_u0_quantiles": [float(q) for q in np.quantile(u0s[ok], [0.05, 0.25, 0.5, 0.75, 0.95])] if ok.any() else None,
                "ks_u0_vs_uniform_0.1_10": float(kstest(u0s[ok], "uniform", args=(0.1, 9.9)).statistic) if ok.sum() > 10 else None,
            }
    for T in TIMES:
        a, oka = u0_fit[("0.01", T)]
        c, okc = u0_fit[("1.0", T)]
        both = oka & okc
        out[f"u0_consistency_beta0.01_vs_beta1_T={T}"] = {
            "n": int(both.sum()), "pearson": float(np.corrcoef(a[both], c[both])[0, 1]),
            "median_abs_rel_diff": float(np.median(np.abs(a[both] - c[both]) / np.abs(c[both]))),
        }
    print(json.dumps(out, indent=2))
    Path("results/replicate_generation.json").write_text(json.dumps(out, indent=2))
