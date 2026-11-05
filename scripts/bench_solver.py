"""Wall-clock cost of the exact sparse-direct solve and of the two-level estimator, vs grid size.

Usage: python scripts/bench_solver.py  -> results/bench_solver.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flowguard.fields import pdebench  # noqa: E402
from flowguard.guard import indicators  # noqa: E402
from flowguard.solver import solve  # noqa: E402


def best_of(fn, reps=3):
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return min(ts)


rows = []
for n in (64, 128, 256, 512):
    a = pdebench(1, np.random.default_rng(0), n=n)[0]
    t_solve = best_of(lambda: solve(a, 1.0, "arith", "ghost"))
    u = solve(a, 1.0, "arith", "ghost")
    t_guard = best_of(lambda: indicators(a, 1.0, u * 1.01))
    rows.append({"n": n, "solve_s": t_solve, "two_level_estimate_s": t_guard})
    print(rows[-1], flush=True)
Path("results/bench_solver.json").write_text(json.dumps(rows, indent=2))
