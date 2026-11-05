# FlowGuard status

Session started 2026-11-05 15:42 (IST). QUEST rule: everything in one session, 72 h max, commit from inside the session.
Public repo: https://github.com/G26karthik/flowguard. Live app: https://g26karthik.github.io/flowguard/ (Pages, Actions build).

## Done (details and numbers in README.md)
- PDEBench Darcy audit: labels = t=1.75 snapshots of the transient generator (replay explains 99.8-99.98% of deviation).
- Reference discretisation study: PDEBench scheme 2.6% from 4x refined; harmonic/face 0.21%. UI says "reference", not "exact".
- Models: fno_steady_s0 (median 0.56% vs reference), fno_label_s0 (3.3% mean vs steady), fno_steady_s1 done.
- Reliability study for s0 (results/reliability_fno_steady_s0.json).
- Serving is torch-free (NumPy FNO, sgemm convs); browser runtime via Pyodide; scripts/ui_check.py passes on the live URL.
- HF Docker Spaces need PRO (402); gcloud is on a different account: not used.
- Diagnostics (flowguard/diagnose.py) and agent-study harness (scripts/agent_study.py).
- Adaptation data: data/cache/adapt_train.npz (6000 fields; regenerate with scripts/make_adapt_data.py).

## Running (background, serial GPU)
- fno_steady_s2 (ends ~20:17) -> then fno_adapt_s0 (fine-tune from s0, 40 epochs, 8k PDEBench + 5.7k broad fields).
- agent_study P0/P1 baseline on CPU -> results/agent_study_fno_steady_s0.json.

## Next
1. Ensemble: export s1, s2 weights; eval_reliability with s0,s1,s2 (ensemble spread vs DST estimate).
2. Adapted model: eval_reliability (incl. held-out layers, contrast3000); compare with s0.
3. Agent study P2 (LLM) needs Anthropic credentials: ask user (ANTHROPIC_API_KEY or `ant auth login`).
4. Experiment B (optional): retrain on harmonic/face targets, measure vs 512^2 refined solutions.
5. Update README with ensemble/adaptation/agent results; republish.
