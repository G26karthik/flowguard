"""Broader training fields for adaptation, solved with the baseline's discretisation (arith faces, ghost wall).

2000 each: PDEBench bumps with contrast log-uniform in [10, 1000]; thin channels (k 1-5, width 0.01-0.05,
contrast 10-1000); smooth log-normal fields (length 0.05-0.2, contrast 10-1000). Seeds disjoint from evaluation.
'layers' and contrast > 1000 are deliberately excluded (held-out tests).
Usage: python scripts/make_adapt_data.py [harm,face] -> data/cache/adapt_train[_harm_face].npz (a, u; float32)
"""
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flowguard.fields import channels, grf, pdebench  # noqa: E402
from flowguard.solver import solve  # noqa: E402


FACE, BC = (sys.argv[1].split(",") if len(sys.argv) > 1 else ["arith", "ghost"])


def _solve(a):
    return solve(a, 1.0, FACE, BC).astype(np.float32)


def make(rng):
    out = []
    for _ in range(10):  # PDEBench thresholds per batch of 200
        out.append(pdebench(200, rng, a_low=10 ** rng.uniform(-3, -1)))
    for _ in range(2000):
        out.append(channels(1, rng, k=int(rng.integers(1, 6)), width=rng.uniform(0.01, 0.05), a_low=10 ** rng.uniform(-3, -1)))
    for _ in range(2000):
        out.append(grf(1, rng, length=rng.uniform(0.05, 0.2), a_low=10 ** rng.uniform(-3, -1)))
    return np.concatenate(out)


if __name__ == "__main__":
    a = make(np.random.default_rng(777))
    with Pool(3) as pool:  # shared laptop: at most 3 workers
        u = np.stack(pool.map(_solve, list(a), chunksize=50))
    Path("data/cache").mkdir(parents=True, exist_ok=True)
    suffix = "" if [FACE, BC] == ["arith", "ghost"] else f"_{FACE}_{BC}"
    np.savez(f"data/cache/adapt_train{suffix}.npz", a=a.astype(np.float32), u=u)
    print("adapt set", a.shape, "contrast range", float((a.max((1, 2)) / a.min((1, 2))).min()), float((a.max((1, 2)) / a.min((1, 2))).max()))
