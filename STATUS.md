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

- `runs/fno_steady_s0` trained (100 epochs, ~31 s/epoch): test vs steady median 0.56%, p90 1.4%, max 7.4%.
- `results/reliability_fno_steady_s0.json`: OOD errors contrast100 76%, GRF 9.4%, channels 11.7%, 64^2 2.9%, 256^2 1.6%.
  DST estimate Spearman 0.74-0.96 (not across resolution); conformal 90% bound: 89.9% coverage on test, fails OOD.
  Triage recall of >5% errors at 20% budget: DST 70%, raw residual 52%, oracle 72%, random 20%.
- App checked locally with the trained model (headless Edge screenshot); flags and batch mode behave.
- Note: two concurrent trainings ran slower than serial; a CUDA illegal-address crash followed force-killing one run.

## Running
- Serial queue (background bash): `fno_label_s0` (started ~17:30), then `fno_steady_s1`, `fno_steady_s2` (~50 min each).

## Next
1. Ensemble eval: `scripts/eval_reliability.py runs/fno_steady_s{0,1,2}/model.pt`.
2. Compare label- vs steady-trained model on steady truth.
3. Adaptation: fine-tune on a broader generated distribution (contrast, GRF, channels) and measure ID cost / OOD gain.
4. Deploy (needs user OK: HF Space + public GitHub repo). README.
