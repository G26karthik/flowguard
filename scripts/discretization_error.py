"""How far is the 128x128 finite-volume 'reference' from the continuum PDE solution?

Same continuous coefficient (PDEBench-style Gaussian bumps, threshold fixed at the 128-grid batch mean)
sampled at N = 128, 256, 512; solved with arithmetic and harmonic face means; fine solutions are
block-averaged onto the 128 grid. Single process (shared laptop).

Usage: python scripts/discretization_error.py [n_fields] -> results/discretization_error.json
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flowguard.solver import solve  # noqa: E402

k = int(sys.argv[1]) if len(sys.argv) > 1 else 12
rng = np.random.default_rng(99)
xm, ym = rng.uniform(0, 1, (2, k, 5))
std = 0.5 * rng.uniform(0, 1, (k, 5))


def bumps(n):
    c = (np.arange(n) + 0.5) / n
    s = np.zeros((k, n, n))
    for i in range(5):
        s += np.exp(-((c[None, :, None] - xm[:, None, None, i]) ** 2 + (c[None, None, :] - ym[:, None, None, i]) ** 2) / std[:, None, None, i])
    return s


thr = bumps(128).mean()
restrict = lambda u, n: u.reshape(128, n // 128, 128, n // 128).mean(axis=(1, 3))
rel = lambda x, y: float(np.linalg.norm(x - y) / np.linalg.norm(y))
sol = {}
for n in (128, 256, 512):
    a = np.where(bumps(n) > thr, 1.0, 0.1)
    for face in ("arith", "harm"):
        sol[(n, face)] = [restrict(solve(a[i], 1.0, face, "ghost"), n) for i in range(k)]
    print("done", n, flush=True)

out = {"n_fields": k}
for face in ("arith", "harm"):
    e128 = [rel(sol[(128, face)][i], sol[(512, face)][i]) for i in range(k)]
    e256 = [rel(sol[(256, face)][i], sol[(512, face)][i]) for i in range(k)]
    d1 = [rel(sol[(128, face)][i], sol[(256, face)][i]) for i in range(k)]
    d2 = [rel(sol[(256, face)][i], sol[(512, face)][i]) for i in range(k)]
    rate = float(np.median(np.log2(np.array(d1) / np.array(d2))))
    out[face] = {"rel_128_vs_512_median": float(np.median(e128)), "rel_128_vs_512_max": float(np.max(e128)),
                 "rel_256_vs_512_median": float(np.median(e256)), "observed_order": rate}
# A user-painted 128x128 field defines a pixel-wise constant coefficient: refine by replicating pixels,
# so every interface stays on a cell face at every resolution.
a128 = np.where(bumps(128) > thr, 1.0, 0.1)
for face in ("arith", "harm"):
    for bc in ("ghost", "face"):
        px = {n: [restrict(solve(np.kron(a128[i], np.ones((n // 128, n // 128))), 1.0, face, bc), n) for i in range(k)] for n in (128, 256, 512)}
        d1 = [rel(px[128][i], px[256][i]) for i in range(k)]
        d2 = [rel(px[256][i], px[512][i]) for i in range(k)]
        out[f"pixel_{face}_{bc}"] = {"rel_128_vs_512_median": float(np.median([rel(px[128][i], px[512][i]) for i in range(k)])),
                                     "observed_order": float(np.median(np.log2(np.array(d1) / np.array(d2))))}
        print("pixel", face, bc, out[f"pixel_{face}_{bc}"], flush=True)
out["arith_vs_harm_at_128_median"] = float(np.median([rel(sol[(128, "arith")][i], sol[(128, "harm")][i]) for i in range(k)]))
out["arith_vs_harm_at_512_median"] = float(np.median([rel(sol[(512, "arith")][i], sol[(512, "harm")][i]) for i in range(k)]))
print(json.dumps(out, indent=2))
Path("results/discretization_error.json").write_text(json.dumps(out, indent=2))
