"""Steady targets for all 10k PDEBench fields with the flux-consistent discretisation (harmonic faces, u = 0 on
the boundary face): 0.21% from a 4x-refined solve versus 2.6% for PDEBench's scheme (results/discretization_error.json).
Usage: python scripts/resolve_hf.py -> data/cache/2D_DarcyFlow_beta1.0_Train_steady_harm_face.npy
"""
import sys
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flowguard.solver import solve  # noqa: E402

RAW = "data/raw/2D_DarcyFlow_beta1.0_Train.hdf5"


def _work(lo):
    with h5py.File(RAW, "r") as h:
        nu = h["nu"][lo:lo + 250].astype(np.float64)
    return lo, np.stack([solve(a, 1.0, "harm", "face") for a in nu]).astype(np.float32)


if __name__ == "__main__":
    out = np.empty((10000, 128, 128), np.float32)
    with Pool(3) as pool:  # shared laptop: at most 3 workers
        for lo, s in pool.imap_unordered(_work, range(0, 10000, 250)):
            out[lo:lo + len(s)] = s
    np.save("data/cache/2D_DarcyFlow_beta1.0_Train_steady_harm_face.npy", out)
    print("done", out.shape, float(out.mean()))
