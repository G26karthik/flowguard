"""Test the 'labels are unconverged time-stepping' hypothesis.

If PDEBench ran u_t = div(a grad u) + beta to a fixed time T from a random initial condition,
the label minus the steady state is dominated by the slowest mode: dev ~ C exp(-lambda_1 T),
with lambda_1 the smallest eigenvalue of the (matched) discrete operator. A linear fit of
log(dev) against lambda_1 should then have slope close to -T.

Usage: python scripts/decay_test.py data/raw/2D_DarcyFlow_beta1.0_Train.hdf5 [n]
"""
import json
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.sparse.linalg import eigsh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flowguard.solver import assemble  # noqa: E402

path = Path(sys.argv[1])
n = int(sys.argv[2]) if len(sys.argv) > 2 else 300
dev_all = np.load(f"results/labelcheck_{path.stem}.npz")["dev"]
idx = np.sort(np.random.default_rng(1).choice(len(dev_all), n, replace=False))
with h5py.File(path, "r") as h:
    nu = h["nu"][idx].astype(np.float64)
lam = np.array([eigsh(assemble(a, "arith", "ghost"), k=1, sigma=0, which="LM", return_eigenvectors=False)[0] for a in nu])
dev = dev_all[idx]
keep = dev > 1e-3  # below this the deviation is at the label's float32/round-off floor
slope, icpt = np.polyfit(lam[keep], np.log(dev[keep]), 1)
r = np.corrcoef(lam[keep], np.log(dev[keep]))[0, 1]
out = {"file": path.name, "n": n, "n_fit": int(keep.sum()), "slope": float(slope), "intercept": float(icpt),
       "pearson_r": float(r), "lambda1_range": [float(lam.min()), float(lam.max())]}
print(json.dumps(out, indent=2))
Path(f"results/decay_{path.stem}.json").write_text(json.dumps(out, indent=2))
np.savez_compressed(f"results/decay_{path.stem}.npz", lam=lam, dev=dev, idx=idx)
