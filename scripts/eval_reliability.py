"""Does each cheap indicator predict the true error of the FNO, in and out of distribution?

For every test set: FNO prediction (member 0 = the served model; all members for ensemble spread),
exact solution (matched solver), true relative L2 error, and the indicators:
  residual_rel   ||f - A u_hat|| / ||f||                    ~0.1 ms
  fft_err_rel    constant-coefficient DST preconditioned      ~2 ms
  galerkin_err   two-level smoothed-aggregation estimate      ~28 ms (exact solve: ~48 ms)
  ens_spread     std over ensemble members / ||mean||         (needs M forward passes)
Calibration (split conformal) of bound = q * fft_err_rel uses the PDEBench validation split only.

Usage: python scripts/eval_reliability.py runs/fno_steady_s0/model.pt [more members ...]
"""
import json
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flowguard.fields import channels, grf, pdebench  # noqa: E402
from flowguard.guard import fft_estimates, indicators  # noqa: E402
from flowguard.infer import Predictor  # noqa: E402
from flowguard.solver import solve  # noqa: E402

pred = Predictor(sys.argv[1:])
split = np.load("results/split.npz")
steady = np.load("data/cache/2D_DarcyFlow_beta1.0_Train_steady.npy", mmap_mode="r")
with h5py.File("data/raw/2D_DarcyFlow_beta1.0_Train.hdf5", "r") as h:
    nu_val, nu_test = h["nu"][split["val"]], h["nu"][split["test"]]

rng = np.random.default_rng(12345)  # disjoint from training (PDEBench used JAX keys 2020-2069)
sets = {
    "pdebench_val": (nu_val, steady[split["val"]]),
    "pdebench_test": (nu_test, steady[split["test"]]),
    "replica": (pdebench(200, rng), None),
    "contrast100": (pdebench(200, rng, a_low=0.01), None),
    "grf": (grf(200, rng), None),
    "channels": (channels(200, rng), None),
    "res64": (pdebench(200, rng, n=64), None),
    "res256": (pdebench(100, rng, n=256), None),
}


def rel(x, y):
    return np.linalg.norm((x - y).reshape(len(x), -1), axis=1) / np.linalg.norm(y.reshape(len(y), -1), axis=1)


per = {}
for name, (a, u) in sets.items():
    a = np.asarray(a, dtype=np.float64)
    u = np.stack([solve(x, 1.0, "arith", "ghost") for x in a]) if u is None else np.asarray(u, dtype=np.float64)
    m = pred.predict(a)  # (M, B, N, N)
    u_hat = m[0]
    ind, e_fft = fft_estimates(a, 1.0, u_hat)
    gal = [indicators(x, 1.0, y) for x, y in zip(a, u_hat)]
    per[name] = {
        "err": rel(u_hat, u),
        "err_ens_mean": rel(m.mean(0), u),
        "residual_rel": ind["residual_rel"],
        "fft_err_rel": ind["fft_err_rel"],
        "galerkin_err_rel": np.array([g[0]["coarse_err_rel"] for g in gal]),
        "ens_spread": np.linalg.norm(m.std(0).reshape(len(a), -1), axis=1) / np.linalg.norm(m.mean(0).reshape(len(a), -1), axis=1) if len(m) > 1 else np.full(len(a), np.nan),
        "err_after_fft_correction": rel(u_hat + e_fft, u),
        "err_after_galerkin_correction": rel(u_hat + np.stack([g[1] for g in gal]), u),
    }
    print(name, "median err %.4f" % np.median(per[name]["err"]), flush=True)

# split-conformal factor on validation: P(err <= q * fft_err_rel) >= 0.9 under exchangeability
cal = per["pdebench_val"]["err"] / per["pdebench_val"]["fft_err_rel"]
nv = len(cal)
q90 = float(np.quantile(cal, min(1.0, np.ceil(0.9 * (nv + 1)) / nv), method="higher"))
qs = lambda e: {"median": float(np.median(e)), "p90": float(np.quantile(e, 0.9)), "max": float(e.max())}
INDS = ["residual_rel", "fft_err_rel", "galerkin_err_rel", "ens_spread"]
summary = {"members": sys.argv[1:], "conformal_q90_fft": q90, "sets": {}}
for name, d in per.items():
    s = {"n": len(d["err"]), "err": qs(d["err"]), "err_ens_mean": qs(d["err_ens_mean"]),
         "err_after_fft_correction": qs(d["err_after_fft_correction"]),
         "err_after_galerkin_correction": qs(d["err_after_galerkin_correction"]),
         "spearman": {k: (None if np.isnan(d[k]).any() else float(spearmanr(d[k], d["err"]).statistic)) for k in INDS},
         "fft_ratio_est_over_true_median": float(np.median(d["fft_err_rel"] / d["err"])),
         "conformal_coverage_fft": float(np.mean(d["err"] <= q90 * d["fft_err_rel"]))}
    summary["sets"][name] = s

# triage: with a budget of exact solves, which indicator finds the bad predictions (err > 5%)?
pool = {k: np.concatenate([per[n][k] for n in per if n != "pdebench_val"]) for k in ["err"] + INDS}
bad = pool["err"] > 0.05
summary["triage"] = {"n": int(len(bad)), "n_bad_over_5pct": int(bad.sum())}
for budget in (0.1, 0.2, 0.3):
    kk = int(budget * len(bad))
    row = {"random_expected": budget}
    for k in INDS + ["err"]:
        if np.isnan(pool[k]).any():
            continue
        top = np.argsort(-pool[k])[:kk]
        row[k if k != "err" else "oracle"] = float(bad[top].sum() / bad.sum())
    summary["triage"][f"recall_at_{int(budget * 100)}pct"] = row

print(json.dumps(summary, indent=2))
tag = Path(sys.argv[1]).parent.name + (f"_ens{len(sys.argv) - 1}" if len(sys.argv) > 2 else "")
Path(f"results/reliability_{tag}.json").write_text(json.dumps(summary, indent=2))
np.savez_compressed(f"results/reliability_{tag}.npz", **{f"{n}__{k}": v for n, d in per.items() for k, v in d.items()})
