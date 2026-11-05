"""Re-solve every PDEBench Darcy sample with the matched discretisation (arith faces, ghost BC).

Usage: python scripts/resolve_dataset.py data/raw/2D_DarcyFlow_beta1.0_Train.hdf5
Writes data/cache/<stem>_steady.npy (float32, N x 128 x 128) and results/labelcheck_<stem>.json,
plus results/labelcheck_<stem>.npz with per-sample deviation of the label from the steady state.
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flowguard.solver import solve  # noqa: E402

FACE, BC = "arith", "ghost"


def _work(args):
    path, lo, hi, beta = args
    with h5py.File(path, "r") as h:
        nu = h["nu"][lo:hi].astype(np.float64)
    return lo, np.stack([solve(a, beta, FACE, BC) for a in nu]).astype(np.float32)


if __name__ == "__main__":
    path = Path(sys.argv[1])
    with h5py.File(path, "r") as h:
        beta = float(h.attrs["beta"])
        n = h["nu"].shape[0]
    chunks = [(str(path), lo, min(lo + 250, n), beta) for lo in range(0, n, 250)]
    steady = np.empty((n, 128, 128), np.float32)
    with Pool(3) as pool:
        for lo, s in pool.imap_unordered(_work, chunks):
            steady[lo:lo + len(s)] = s
            print("done", lo, flush=True)
    Path("data/cache").mkdir(parents=True, exist_ok=True)
    np.save(f"data/cache/{path.stem}_steady.npy", steady)

    with h5py.File(path, "r") as h:
        label = h["tensor"][:, 0]
        frac_low = (h["nu"][:] < 0.5).mean(axis=(1, 2))
    dev = np.linalg.norm((label - steady).reshape(n, -1), axis=1) / np.linalg.norm(steady.reshape(n, -1), axis=1)
    summary = {
        "file": path.name, "beta": beta, "discretisation": [FACE, BC], "n": n,
        "rel_l2_label_vs_steady_quantiles": {q: float(np.quantile(dev, float(q))) for q in ("0.5", "0.75", "0.9", "0.95", "0.99", "1.0")},
        "fraction_over": {t: float((dev > float(t)).mean()) for t in ("0.01", "0.02", "0.05", "0.1")},
        "spearman_dev_vs_low_conductivity_fraction": float(np.corrcoef(np.argsort(np.argsort(dev)), np.argsort(np.argsort(frac_low)))[0, 1]),
        "mean_signed_label_minus_steady_over_mean_steady": float((label - steady).mean() / steady.mean()),
    }
    print(json.dumps(summary, indent=2))
    Path(f"results/labelcheck_{path.stem}.json").write_text(json.dumps(summary, indent=2))
    np.savez_compressed(f"results/labelcheck_{path.stem}.npz", dev=dev, frac_low=frac_low)
