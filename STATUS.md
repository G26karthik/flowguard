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
- 2026-11-06 API abuse limits live (revision flowguard--0000004 or later; flowguard/api.py, README Deployment):
  batch work cap (count x grid area), 1 MiB bodies, one computation at a time + 4 queued, per-client compute budget
  (120 s burst, 0.2 s/s) keyed on the last X-Forwarded-For entry. Verified live, including spoofed XFF.
- 2026-11-06 LLM investigator pilot (scripts/llm_pilot.py, protocol commit 091b94f before the run;
  results/llm_pilot_claude-haiku-5-5_medium.json): 10 fresh cases, rule 10/10, Haiku 5.5 medium 8/10, flags 5/10;
  Haiku 3.1 experiments/case (1.2 unnecessary), 8.4 s/case, $0.0098 total. Not scaled (pre-registered rule).

## Running
- Nothing.

## Needs the user
1. Optional: Sonnet 5.5 (low effort) on the same 10 pilot cases, about $0.2. Only worth it to learn whether Haiku's
   misreading is a model-size issue; at best it ties the rule (10/10).
2. Optional cost cap: a scheduled GitHub Action (OIDC login, role limited to resource group `flowguard`) that
   disables ingress when month-to-date replica time passes ~40 of the 50 free hours; the site then computes in the
   browser. Creates an Entra app registration with write access to that resource group: needs approval.
3. Anything that adds Azure cost (min replicas > 0, more CPU, Log Analytics, a registry, metric alerts) needs the
   user's approval. Free grant per month: 180k vCPU-s, 360k GiB-s, 2M requests = 50 replica-hours at 1 vCPU / 2 GiB.

## Next
- LLM pilot rerun: `uv run --with anthropic python scripts/llm_pilot.py --model M --effort E` (key from .env, never printed).
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
