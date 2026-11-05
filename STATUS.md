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

## Running (background)
- data/cache/adapt_train_harm_face.npz generation -> fno_hf_adapt_s0 fine-tune (40 epochs from fno_hf_s0, ~35 min).

## Next
1. Export + eval_reliability for fno_hf_adapt_s0 (and fno_hf_s0); eval_physical for the adapted pair.
2. If it holds: switch models.json to the harmonic/face pair; update Evidence text ("2.6%" reference error -> 0.21%),
   README section 3; rebuild, ui_check, push.
3. LLM arm of the agent study once credentials exist (`ANTHROPIC_API_KEY` or `ant auth login`):
   `python scripts/agent_study.py runs/fno_steady_s0/weights.npz --llm claude-opus-5-5`.
4. Known gaps: no indicator detects grid changes (input flag only); contrast > 1000x still fails; served NumPy network
   is slower than the direct solver (GPU torch is 39-148x faster).

## Gotchas
- PowerShell: `$env:CUDA_VISIBLE_DEVICES=""` deletes the variable; use "-1" to hide the GPU.
- Two concurrent trainings on the laptop GPU are slower than serial; force-killing one caused a CUDA illegal-address
  crash in the other.
- Shared laptop: multiprocessing pools capped at 3 workers.
