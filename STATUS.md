# FlowGuard status

Session started 2026-11-05 15:42 (IST). QUEST rule: everything in one session, 72 h max, commit from inside the session.
Public repo: https://github.com/G26karthik/flowguard. Live app: https://g26karthik.github.io/flowguard/ (Pages, Actions build).
All measured results: README.md. This file: state and next steps only.

## Done
- Data audit (PDEBench labels = t=1.75 transient snapshots), discretisation study, label-trained baseline.
- Models (runs/): fno_steady_s0/s1/s2 (PDEBench scheme), fno_label_s0, fno_adapt_s0 (broad, PDEBench scheme),
  fno_hf_s0 (harmonic/face targets: physical error 0.56% vs 3.1%).
- Reliability evals: s0 ensemble (results/reliability_fno_steady_s0_ens3.json), adapted (reliability_fno_adapt_s0.json).
- Agent study: ablation rule 85% exact match at 1.2 experiments/case; LLM arm implemented, not run (no credentials).
- Serving: models.json registry (broad default + PDEBench specialist), torch-free NumPy path, Pyodide browser
  runtime, FastAPI + Docker (verified). ui_check.py passes on server, static and live URL.

- Served pair switched to harmonic/face: fno_hf_adapt_s0 (broad, default) and fno_hf_s0 (specialist); live and
  checked with ui_check.py. Grid transfer 2.9%/1.6% -> 0.64%/0.58% (64²/256²) with the consistent scheme.
- Fixed: eval_reliability used PDEBench-scheme truth for val/test regardless of model (flowguard/data.py + test).

## Running
- Nothing.

## Next
1. LLM arm of the agent study once credentials exist (`ANTHROPIC_API_KEY` or `ant auth login`):
   `python scripts/agent_study.py runs/fno_steady_s0/weights.npz --llm claude-opus-5-5`.
2. Known gaps: contrast > 1000x still fails (39%); held-out layers 4.6%; the residual estimate ranks poorly on the
   broad model's held-out sets (Spearman 0.16 / 0.25); served NumPy network is slower than the direct solver
   (GPU torch is 39-148x faster). Agent study was run on the PDEBench-scheme model only.
3. Video walkthrough and the application form are the user's to do.

## Gotchas
- PowerShell: `$env:CUDA_VISIBLE_DEVICES=""` deletes the variable; use "-1" to hide the GPU.
- Two concurrent trainings on the laptop GPU are slower than serial; force-killing one caused a CUDA illegal-address
  crash in the other.
- Shared laptop: multiprocessing pools capped at 3 workers.
