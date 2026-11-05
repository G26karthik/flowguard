# FlowGuard status

Session started 2026-11-05 15:42 (IST). QUEST rule: everything in one session, 72 h max, commit from inside the session.

## Done
- Data: PDEBench 2D Darcy beta = 0.01, 1.0, 100.0 downloaded to `data/raw/` (md5 verified against PDEBench's CSV).
- Solver `flowguard/solver.py`: cell-centred FV; tests incl. 2nd-order convergence on a manufactured solution.
- Label audit (`results/labelcheck_*.json`, `results/decay_*.json`):
  - PDEBench's generator matches arithmetic face means + u = 0 at ghost-cell centres (not on the boundary).
  - Labels are unconverged time-stepping for many samples: beta=0.01 median 17% off steady; beta=1 9.5% of samples >10% off; beta=100 max 3.3%.
  - Deviation tracks the slowest decay mode (log dev vs lambda_1: r = -0.89).
  - `nu` identical across beta files; steady problem is linear, so u_beta = beta * u_1.
- Re-solved steady targets for all 10k fields: `data/cache/*_steady.npy` (regenerate: `scripts/resolve_dataset.py`).
- Generators `flowguard/fields.py` (PDEBench replica KS 0.02; GRF; channels).
- Indicators `flowguard/guard.py`: DST-preconditioned residual estimate (~2 ms); two-level Galerkin estimate (lower bound, but ~55% of exact-solve cost, so not used in the app).
- Benchmark `results/bench_solver.json`: exact solve 9 / 48 / 274 / 2537 ms at N = 64 / 128 / 256 / 512.
- API `flowguard/api.py`, frontend `web/index.html`, `Dockerfile` (not yet deployed).

## Running
- `runs/fno_steady_s0` (100 epochs, ~24 s/epoch alone).
- Queue (serial, after s0): `fno_label_s0`, `fno_steady_s1`, `fno_steady_s2`.

## Next
1. `scripts/eval_reliability.py runs/fno_steady_s0/model.pt` then with all 3 steady members.
2. Compare steady- vs label-trained model on steady truth.
3. Run app locally, screenshot, check UI.
4. Deploy (needs user OK: HF Space + public GitHub repo).
