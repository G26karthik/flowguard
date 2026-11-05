"""Network cost per field vs batch size and grid, for torch on GPU (fp32, TF32 off), torch on CPU, and the NumPy
serving path; compare with results/bench_solver.json. Usage: python scripts/bench_network.py -> results/bench_network.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flowguard.model import FNO2d  # noqa: E402
from flowguard.npfno import NumpyFNO  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
ck = torch.load("runs/fno_steady_s0/model.pt", map_location="cpu", weights_only=False)
npm = NumpyFNO("runs/fno_steady_s0/weights.npz")
rows = []
for dev in (["cuda"] if torch.cuda.is_available() else []) + ["cpu", "numpy"]:
    m = None
    if dev != "numpy":
        m = FNO2d(**ck["cfg"]).to(dev).eval()
        m.load_state_dict(ck["state_dict"])
    for n, batches in ((128, (1, 64, 256)), (256, (1, 16)), (512, (1, 4))):
        for b in batches:
            if dev != "cuda" and b > 64:
                continue
            x = np.random.default_rng(0).uniform(-1, 0, (b, n, n)).astype(np.float32)
            if dev == "numpy":
                f = lambda: npm(x)
            else:
                xt = torch.from_numpy(x).unsqueeze(1).to(dev)
                def f():
                    with torch.no_grad():
                        y = m(xt)
                    if dev == "cuda":
                        torch.cuda.synchronize()
                    return y
            f()  # warm-up
            reps = 5 if dev == "cuda" else 2
            t = time.perf_counter()
            for _ in range(reps):
                f()
            per = (time.perf_counter() - t) / reps / b
            rows.append({"device": dev, "n": n, "batch": b, "ms_per_field": 1e3 * per})
            print(rows[-1], flush=True)
Path("results/bench_network.json").write_text(json.dumps(rows, indent=2))
