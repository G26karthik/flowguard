"""Physical accuracy: error against a 512x512 solve of the same pixel-wise coefficient (harmonic faces, wall on the
boundary face; about 0.03% from the continuum by its observed order, results/discretization_error.json).

Compares, on PDEBench test fields: each 128x128 network and the 128x128 reference of its own discretisation.
Usage: python scripts/eval_physical.py runs/fno_steady_s0/model.pt runs/fno_hf_s0/model.pt [n=40]
"""
import json
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flowguard.infer import Predictor  # noqa: E402
from flowguard.solver import solve  # noqa: E402

models = [a for a in sys.argv[1:] if a.endswith(".pt") or a.endswith(".npz")]
n = int(next((a for a in sys.argv[1:] if a.isdigit()), 40))
idx = np.load("results/split.npz")["test"][:n]
with h5py.File("data/raw/2D_DarcyFlow_beta1.0_Train.hdf5", "r") as h:
    nu = h["nu"][idx].astype(np.float64)

restrict = lambda u: u.reshape(128, 4, 128, 4).mean(axis=(1, 3))
truth = np.stack([restrict(solve(np.kron(a, np.ones((4, 4))), 1.0, "harm", "face")) for a in nu])
rel = lambda x: np.linalg.norm((x - truth).reshape(n, -1), axis=1) / np.linalg.norm(truth.reshape(n, -1), axis=1)
q = lambda e: {"median": float(np.median(e)), "p90": float(np.quantile(e, 0.9)), "max": float(e.max())}

out = {"n": n, "truth": "512x512 harmonic/face solve, block-averaged to 128", "rows": {}}
for m in models:
    p = Predictor([m])
    ref = np.stack([solve(a, 1.0, p.face, p.bc) for a in nu])
    net = p.predict(nu)[0]
    out["rows"][m] = {"discretisation": [p.face, p.bc], "network_vs_truth": q(rel(net)),
                      "reference128_vs_truth": q(rel(ref)), "network_vs_own_reference": q(np.linalg.norm((net - ref).reshape(n, -1), axis=1) / np.linalg.norm(ref.reshape(n, -1), axis=1))}
    print(m, json.dumps(out["rows"][m]), flush=True)
Path("results/physical_accuracy.json").write_text(json.dumps(out, indent=2))
