# FlowGuard status

Session started 2026-11-05 15:42 (IST). QUEST rule: everything in one session, 72 h max, commit from inside the session.
Public repo: https://github.com/G26karthik/flowguard. Live app: https://g26karthik.github.io/flowguard/ (Pages, Actions build).
All measured results: README.md. This file: state and next steps only.

## Done
- Data audit (PDEBench labels = t=1.75 transient snapshots), discretisation study, label-trained baseline.
- Models (runs/): fno_steady_s0/s1/s2, fno_label_s0, fno_adapt_s0 (PDEBench scheme); served pair fno_hf_s0 and
  fno_hf_adapt_s0 (harmonic/face).
- Reliability evals for every model; eval truth follows each model's discretisation (flowguard/data.py).
- 2026-11-06 performance and product pass:
  - baseline of the live browser runtime (results/perf/before_browser_desktop.json): 10.2 s to first answer, 37.6 MB.
  - NumPy network 178 -> 53 ms/field (truncated-DFT sgemm, blocked GELU); per-field caches; beta rescales locally.
  - Docker image fixed (models.json was never copied: import failed); BLAS threads from the cgroup quota
    (1 vCPU: 200 -> 43 ms/field). 1-vCPU container: 0.31 s first answer, 123 ms per field (results/perf/after_*).
  - UI: examples, explainer, status line, investigation panel; ui_check passes on server and browser runtimes.
  - Investigator integrated (/api/investigate) and evaluated on the served models
    (results/investigator_study.json): rule 85% / 78% vs flags 35% / 41% against causal labels.

- 2026-11-06 server live on Azure Container Apps (Railway trial had expired; user approved Azure, free grant only):
  - subscription 434cb31e... (Pay-As-You-Go, no spending limit); everything in resource group `flowguard`
    (centralindia): environment `flowguard-env` (Consumption profile only, logs destination none, no VNet) and app
    `flowguard` (1 vCPU / 2 GiB, min 0, max 1 replica, FLOWGUARD_THREADS=1, CORS = Pages origin).
  - URL https://flowguard.delightfulbay-9f598c7b.centralindia.azurecontainerapps.io/ ; image
    ghcr.io/g26karthik/flowguard:sha-<commit> (public, built by .github/workflows/image.yml).
  - repo variable FLOWGUARD_API set to that URL; Pages uses the server, browser runtime races it while it wakes.
  - budget `flowguard-any-charge` on the resource group: amount 1 (billing currency), emails on any actual charge.
  - live measurements: results/perf/live_azure_*.json (warm: first answer 0.6 s, 188 ms per field; cold: browser answers
    at 11-13 s, server takes over at 22-24 s); ui_check passes on the live site.

## Running
- Nothing.

## Needs the user
1. LLM arm of the investigator study: ANTHROPIC_API_KEY, then
   `python scripts/investigator_study.py --llm claude-opus-5-5` (~170 evaluation cases).
2. Anything that adds Azure cost (min replicas > 0, more CPU, Log Analytics, a registry, metric alerts) needs the
   user's approval. Free grant per month: 180k vCPU-s, 360k GiB-s, 2M requests = 50 replica-hours at 1 vCPU / 2 GiB.

## Next
- Check usage: `python scripts/azure_usage.py` (replica hours vs free grant, month-to-date cost).
- Deploy a new image: `az containerapp update -n flowguard -g flowguard --image ghcr.io/g26karthik/flowguard:sha-<commit>`.
- Known gaps: contrast > 1000x still fails (39%); held-out layers 4.6%; the residual estimate ranks poorly on the
  broad model's held-out sets (Spearman 0.16 / 0.25); investigator misses the weaker factor in dominated pairs.

## Gotchas
- PowerShell: `$env:CUDA_VISIBLE_DEVICES=""` deletes the variable; use "-1" to hide the GPU.
- Two concurrent trainings on the laptop GPU are slower than serial; force-killing one caused a CUDA illegal-address
  crash in the other.
- Shared laptop: multiprocessing pools capped at 3 workers.
- Perf runs against one server reuse its cache: use a fresh container per run (results/perf/after_* do).
- Bash heredocs containing Python triple quotes get mangled by the shell hook: write patch scripts to files.
