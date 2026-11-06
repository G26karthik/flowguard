# FlowGuard

Steady Darcy flow from Fourier neural operators, with every answer checked against an independent finite-volume
solver, and cheap indicators that say when a network should not be trusted.

**Live app:** https://g26karthik.github.io/flowguard/ (see [Deployment](#deployment) for the server and the browser runtime).

Built for the QUEST Lab (IISc) Full Stack AI pre-doctoral application, project 3, "PDE solutions as a service",
in a single Claude Code session. The git history and the session log line up; `STATUS.md` is the running log.

## The problem

$$-\nabla\cdot(a(x)\nabla u(x)) = \beta \ \text{ on } (0,1)^2, \qquad u = 0 \text{ on the boundary.}$$

Users give a permeability field `a` (generated, painted, or uploaded) and a source `β`. They get the network's
pressure field, a finite-volume reference solution on the same grid, the difference, and flags when the input is
unlike the network's training data. The problem is linear in `β`, so one model trained at `β = 1` serves every `β`
exactly (`u_β = β u_1`).

## What the evidence shows

Every number comes from a script in `scripts/`; outputs are in `results/` and `runs/`. Errors are relative L2.

### 1. PDEBench's Darcy labels are transient snapshots, not steady states

PDEBench describes its Darcy data as the steady state. Its generator (`data_gen_NLE/ReactionDiffusionEq`) instead
integrates `u_t = div(a grad u) + β` from a near-constant random level `u0 ~ U(0.1, 10)` and stores a fixed-time
snapshot.

- Re-solving all 10,000 fields with the matched discretisation: for `β = 1`, 9.5% of labels are more than 10% from
  the steady solution; for `β = 0.01` the median label is 17% away; `β = 100` is close (max 3.3%).
  (`scripts/resolve_dataset.py`, `results/labelcheck_*.json`)
- Replaying the generator explains 99.8 to 99.98% of every label's deviation: labels are the `t = 1.75` snapshot.
  Fitted `u0` follow `U(0.1, 10)` at `t = 1.75` (KS 0.07) and not at `t = 2.0` (KS 0.70; negative `u0` at `β = 100`);
  the same sample gives the same `u0` in every `β` file (r = 0.99992).
  (`scripts/replicate_generation.py`, `results/replicate_generation.json`, `results/fig_label_audit.png`)
- So the labels are faithful outputs of the documented generator; they differ from the paper's description. For
  slowly-decaying fields they carry `u0`, which is not a model input, so no network can fit them: trained on the
  labels, an FNO has 3.3% mean error against the steady solution; trained on re-solved targets, 0.76%.
- The generator also puts `u = 0` at ghost-cell centres, half a cell outside the domain, not on the boundary.

### 2. The reference solver is an approximation, and the choice of discretisation matters most

On pixel-wise coefficients, difference from a 4x refined solve (`scripts/discretization_error.py`):

| Discretisation (128x128) | vs 4x refined | observed order |
|---|---|---|
| arithmetic faces, ghost-centre wall (PDEBench's) | 2.6% | 1.0 |
| harmonic faces, wall on the boundary face | 0.21% | 1.3 |

The ghost-centre wall alone is an O(h) error. Training the same FNO on targets from each scheme and measuring
against a 512x512 harmonic/face solve on 40 test fields (`scripts/eval_physical.py`, `results/physical_accuracy.json`):

| Targets | network vs 512² truth | 128² reference vs truth | network vs its own reference |
|---|---|---|---|
| PDEBench scheme | 3.1% | 3.1% | 0.47% |
| harmonic/face | **0.56%** | 0.21% | 0.46% |
| PDEBench scheme, adapted (broad) | 3.2% | 3.1% | 0.74% |
| harmonic/face, adapted (broad) | **0.72%** | 0.21% | 0.62% |

(`results/physical_accuracy_adapt.json` for the adapted pair.) Same network, same training cost, same fit to its
targets: the physical error drops 4 to 6x. The PDEBench-scheme network was never limited by learning; it was
faithfully learning a discretisation that is 3% off.

The same choice explains most of the network's poor transfer to other grids. The ghost-centre wall moves with the
grid (h/2 outside the domain), so a network trained at 128² meets a slightly different problem at 64² or 256². With
the consistent scheme, the error at other grids drops from 2.9% / 1.6% to 0.64% / 0.58% (64² / 256²).

**The served models use harmonic/face targets.** The app calls the finite-volume solution the "reference", states
its discretisation error (0.21%), and always solves with the discretisation recorded in the model checkpoint.

### 3. Accuracy in and out of distribution; adaptation

Median error against the model's own reference (`results/reliability_*.json`). The two served models are the
harmonic/face pair; the PDEBench-scheme pair is shown for comparison.

| Inputs | PDEBench-trained (PDEBench scheme) | adapted (PDEBench scheme) | **PDEBench-trained (served)** | **adapted (served, default)** |
|---|---|---|---|---|
| PDEBench test split (1,000) | 0.54% | 0.76% | **0.52%** | 0.73% |
| Fresh fields from a replica of PDEBench's sampler (KS 0.02) | 0.60% | 0.81% | 0.58% | 0.83% |
| Contrast 100x (training: 10x) | 76% | 3.1% | 77% | **3.4%** |
| Smooth log-normal fields | 9.4% | 1.6% | 10.4% | **1.6%** |
| Thin channels | 11.7% | 2.6% | 11.7% | **2.6%** |
| 64x64 / 256x256 grids | 2.9% / 1.6% | 2.5% / 1.6% | **0.64% / 0.58%** | 1.0% / 0.79% |
| Tilted layers (held out from all training) | 6.8% | 4.3% | 6.9% | **4.6%** |
| Contrast 3000x (held out from all training) | 99% | 39% | 99% | 39% |

The adapted models are fine-tuned on PDEBench fields plus 5,700 broader fields (contrast up to 1000x, channels,
smooth fields; `scripts/make_adapt_data.py`). Adaptation costs about 0.2 percentage points on PDEBench-like fields,
removes most of the error on the families it saw, and helps on held-out layers. It does not extrapolate in contrast.
Three seeds of the PDEBench-scheme model give test medians of 0.56, 0.56 and 0.59% in their training-time evaluation
(GPU, TF32); seed 0 is 0.54% in exact fp32.

### 4. Do the cheap checks know when the network is wrong?

Spearman rank correlation between indicator and measured error (`results/reliability_fno_steady_s0_ens3.json`):

| Indicator (cost) | PDEBench-like | smooth | channels | contrast 3000x | layers | 64² grid |
|---|---|---|---|---|---|---|
| DST-preconditioned residual estimate (~2 ms) | 0.78 | 0.78 | 0.96 | 0.58 | 0.61 | -0.21 |
| Raw residual norm | 0.70 | 0.24 | -0.48 | 0.38 | 0.10 | -0.20 |
| Spread of a 3-seed ensemble (3 networks) | 0.84 | -0.11 | -0.05 | 0.09 | 0.41 | 0.03 |
| Symmetry spread, 8 rotations/flips (first 40 fields) | 0.87 | -0.22 | -0.25 | -0.41 | 0.31 | 0.04 |
| Two-level Galerkin estimate (~55% of a solve) | 0.98 | 0.97 | 1.00 | 0.92 | 0.99 | 0.93 |

- The ensemble and the symmetry probe are the best cheap indicators on PDEBench-like fields and fail exactly where
  it matters: the networks agree while all being wrong. The DST estimate is the only cheap indicator that holds
  across input families. With the PDEBench scheme it fails on grid changes (above); with the served harmonic/face
  models it ranks errors there too (Spearman 0.67 at 64², 0.63 at 256²; `results/reliability_fno_hf_s0.json`).
  The app still flags grid changes from the input.
- A split-conformal 90% bound on the DST estimate (calibrated on the validation split) covers 90% of the test split.
  On shifted inputs coverage ranges from 20% (smooth fields) and 32% (64² grid) to 100% (channels, where the bound
  is loose), so the app shows it with a caveat when inputs are flagged.
- The Galerkin estimate is the best indicator but costs about half a reference solve, so solving directly is better.
- Pooled "solve the riskiest 20%" triage mostly measures separation between input families, where every indicator
  except the raw residual does about equally (`results/triage_unflagged.txt`); within a family, the table above is
  what matters.

### 5. Investigating an unreliable prediction, and whether an LLM should choose the experiments

`flowguard/diagnose.py` runs counterfactual experiments on one field. It moves every off-training property back
into the model's training range (grid, contrast, smooth variation; the range comes from `models.json`, so it differs
between the two served models), then adds each property back alone, and swaps the shape for training-family shapes
with the same values and grid. A fixed rule names a property as a cause when adding it back raises the error by
1.5x or more, and the shape when the training-like version is at least 4x worse than training-family shapes (and 4x
the in-distribution median). The app runs this as "Investigate why" (`/api/investigate`): about 0.9 s and 3 to 4
network runs plus solves per field on the 1-vCPU server.

**Served models** (`scripts/investigator_study.py`, `results/investigator_study.json`). Cases with known
construction: contrast 3 to 30x beyond the model's range, 64 or 256 grids, smooth fields (PDEBench model only),
unseen shapes, in-domain variety as a control, singly and in pairs. Thresholds were chosen on a calibration split
(102 cases) and scored on a separate evaluation split (170 cases):

| Exact cause set, evaluation split | broad model | PDEBench specialist | experiments per case |
|---|---|---|---|
| Input flags only, against construction labels | 60% | 73% | 0 |
| Rule, against construction labels | 48% | 63% | 2.8 to 3.6 |
| Input flags only, against causal labels | 35% | 41% | 0 |
| **Rule, against causal labels** | **85%** | **78%** | 2.8 to 3.6 |

The first scoring used construction labels and the rule lost to the input flags. The reason was the labels: the
served (harmonic/face) models are grid-robust (median error on grid-only cases 0.9% and 0.5%, the in-distribution
level), so a grid change is applied by construction but does not cause error. The rule says so; the flags, which
restate the construction, cannot. We then added causal labels from independent data, defined after seeing that first
result: a property counts as a cause for a model when that model's median error on the matching held-out reliability
set is at least 2x its in-distribution median (grid: 1.4x and 1.2x, so not a cause; contrast, smooth fields, unseen
shapes: 2.1x to 149x). Against those labels the rule is right about twice as often as the flags. Its remaining
misses are mostly pairs where one property dominates (contrast with an unseen shape, smooth values with an unseen
shape): once the dominant one is undone, the second changes the error little, so whether it is "a cause" is
ambiguous by construction.

An earlier version of this study, on the PDEBench-scheme model with a fixed geometry threshold
(`scripts/agent_study.py`, `results/agent_study_fno_steady_s0.json`, 72 cases), gave flags 75%, a first rule that
undid each axis alone 56%, and the ablation rule 85%.

**LLM policy: pilot** (`scripts/llm_pilot.py`, `results/llm_pilot_claude-haiku-5-5_medium.json`). The protocol was
committed (091b94f) before the first pilot call: 10 fresh cases (seed 20261106, 5 per served model), the same
experiments and an 8-unit budget for both policies. Truth is per case and by intervention: for every subset of the
factors applied to a case, the error with the others undone (geometry: 16 training-family shapes neither policy
sees); a factor is a cause when its Shapley effect on log error is at least ln 2 (adding it doubles the error on
average). Secondary truth: the population-level labels above. Claude Haiku 5.5, effort medium, adaptive thinking:

| | flags | rule (served) | Haiku 5.5 |
|---|---|---|---|
| exact match, per-case truth | 5/10 | 10/10 | 8/10 |
| exact match, population truth | 5/10 | 9/10 | 7/10 |
| precision / recall of named causes | 0.55 / 0.86 | 1.00 / 1.00 | 0.78 / 1.00 |
| experiments requested per case | 0 | 1.9 | 3.1 |
| unnecessary (not applicable, repeated, unseen) | 0 | 0.3 | 1.2 |
| experiment cost per case (units) | 0 | 3.7 | 4.3 |
| latency per case | 0 | 0.51 s | 8.4 s (7.8 s in the API) |
| tokens, 10 cases (input / output incl. thinking) | | | 44,976 / 10,507 |
| API cost, 10 cases (list price) | | | $0.0098 |

Both Haiku errors name geometry when it is not a cause. On an in-domain smooth field for the broad model, the input's
error (0.067) lay inside the range of the training-shape comparators (0.028 to 0.084), and Haiku's note says
"training fields were two-valued", which the prompt states is false for that model. On a smoothed PDEBench field, the
thresholded baseline was 3.8x the comparators, just under the rule's calibrated 4x; the LLM is given no threshold.
By the pre-registered decision rule (at least the rule's exact matches, or within one at lower cost) the LLM is
not scaled up, and the app keeps the rule. 10 cases cannot separate the policies statistically, and the truth uses
the same edit operators as the tools, which may favour the rule.

### 6. Is the network worth it here?

Cost per field (`results/bench_solver.json`, `results/bench_network.json`; laptop RTX 4060 and Ryzen 9 8945HS):

| Grid | sparse direct solve | FNO, torch GPU (batched) | FNO, torch CPU | FNO, NumPy (served) |
|---|---|---|---|---|
| 128² | 48 ms | 1.2 ms | 26 ms | 53 ms (170 ms before the rewrite) |
| 256² | 274 ms | 5.2 ms | 119 ms | 169 ms |
| 512² | 2.5 s | 17 ms | 480 ms | not measured |

The served NumPy forward pass computes the network's 12 x 12 retained Fourier modes with truncated DFT matrix
products (BLAS) instead of full FFTs, and the GELU in place on cache-sized blocks; it matches the previous pass to
6e-7. On a GPU the network is 39x (128²) to 148x (512²) faster than the direct solve. On the served CPU it is about
as fast as the solve, so there the value is the reliability workflow, and screening many fields with few solves.

## Repository

| Path | What |
|---|---|
| `flowguard/solver.py` | finite-volume solver, matrix-free operator, transient propagator |
| `flowguard/model.py`, `scripts/train.py` | FNO (torch) and training (`--init`, `--extra`, `--disc` for adaptation and experiment B) |
| `flowguard/npfno.py`, `scripts/export_numpy.py` | NumPy forward pass used for serving (max relative difference to torch fp32: 4e-7) |
| `flowguard/guard.py` | residual, DST and Galerkin error indicators |
| `flowguard/diagnose.py` | counterfactual diagnostic experiments and the attribution rule |
| `flowguard/fields.py` | PDEBench sampler replica and shifted families (`layers` is held out from all training) |
| `flowguard/core.py`, `flowguard/api.py`, `models.json` | service logic and caches, model registry, FastAPI wrapper |
| `web/` | front end; `worker.js` runs the same package in the browser via Pyodide |
| `Dockerfile`, `scripts/served_files.py` | server image; `models.json` decides which weights ship |
| `scripts/` | every experiment in this README; `ui_check.py` and `perf_browser.py` drive the app in Microsoft Edge |

## Reproduce

```bash
uv venv .venv --python 3.11
uv pip install --python .venv -r requirements-lock.txt   # CUDA torch; or install torch for your platform first
# data: the three PDEBench 2D Darcy files listed in PDEBench's pdebench_data_urls.csv (md5-checked) into data/raw/
python scripts/resolve_dataset.py data/raw/2D_DarcyFlow_beta1.0_Train.hdf5      # steady targets
python scripts/train.py --target steady --seed 0 --epochs 100 --out runs/fno_steady_s0
python scripts/make_adapt_data.py && python scripts/train.py --target steady --epochs 40 --lr 3e-4 \
    --init runs/fno_steady_s0/model.pt --extra data/cache/adapt_train.npz --out runs/fno_adapt_s0
python scripts/export_numpy.py runs/fno_adapt_s0/model.pt
python scripts/eval_reliability.py runs/fno_adapt_s0/model.pt
# served models (harmonic faces, wall on the boundary face)
python scripts/resolve_hf.py && python scripts/train.py --target steady --disc harm,face --epochs 100 --out runs/fno_hf_s0
python scripts/make_adapt_data.py harm,face && python scripts/train.py --target steady --disc harm,face --epochs 40 \
    --lr 3e-4 --init runs/fno_hf_s0/model.pt --extra data/cache/adapt_train_harm_face.npz --out runs/fno_hf_adapt_s0
python scripts/eval_physical.py runs/fno_steady_s0/model.pt runs/fno_hf_s0/model.pt
python scripts/investigator_study.py                                # investigator on the served models (~4 min)
python -m pytest -q
uvicorn flowguard.api:app --port 8000                               # server mode
python scripts/build_site.py && python -m http.server -d _site      # browser mode
docker build -t flowguard . && docker run -p 8000:8000 flowguard    # production image
uv run --with playwright python scripts/perf_browser.py http://127.0.0.1:8000/ out.json   # timings
```

Seeds: data split `numpy.default_rng(0)`, training `--seed`, evaluation sets `default_rng(12345)`, adaptation fields
`default_rng(777)`, investigator cases `default_rng(2024)` (calibration) and `default_rng(31337)` (evaluation). About
31 s per epoch on the laptop GPU. On Windows PowerShell, hide the GPU with `$env:CUDA_VISIBLE_DEVICES = "-1"`:
assigning an empty string deletes the variable (an evaluation in this project ran on the GPU with cuDNN TF32 because
of that, which moved the test median from 0.54% to 0.56%).

## Deployment

The server is one FastAPI service that also serves the front end (`Dockerfile`). It runs on Azure Container Apps
(Central India, 1 vCPU and 2 GiB, scaling between 0 and 1 replica, so it costs nothing inside the monthly free
grant) from the public image `ghcr.io/g26karthik/flowguard`, which `.github/workflows/image.yml` builds on every push:
https://flowguard.delightfulbay-9f598c7b.centralindia.azurecontainerapps.io/. The GitHub Pages site is built with
that address (repository variable `FLOWGUARD_API`) and calls it, with CORS limited to the Pages origin. The same
`flowguard` package also runs in the browser through Pyodide: when the server does not answer within 2 s (it is
waking from zero), the page starts the in-browser runtime and uses whichever is ready first; "Compute in this
browser instead" (`?local=1`) forces it. `scripts/azure_usage.py` reports month-to-date replica time against the free
grant and the subscription's cost; a resource-group budget emails on any actual charge.

Abuse limits (`flowguard/api.py`). CORS stops other web pages, not direct calls, and the largest request the API
accepted (256 fields at 256 x 256) took 140 s and 1.2 GiB on one core, so two at once could exhaust the 2 GiB
replica. Now: a batch holds at most 256 fields at 128 x 128, scaled down with grid area (64 at 256 x 256, at most
about 27 s); request bodies over 1 MiB are refused; one computation runs at a time, with up to 4 queued for at most
30 s and later ones refused with 503; and each client address has a compute budget of 120 s, refilled at 0.2 s
per second (12 s of compute per minute), after which it gets 429 with `Retry-After`. The address is the last
`X-Forwarded-For` entry, the one Azure's ingress appends. Checked live: 7 simultaneous 64-field batches ran one
after another (4 to 20 s) with 2 refused at once; after about 120 s of compute the address got 429 even when the
caller rotated a fake first `X-Forwarded-For` entry; the health endpoint kept answering. Ordinary use (solves of
0.2 s, batches of 64 in 4 s) stays far below the budget. Users behind one shared address share one budget. These
limits protect memory and responsiveness, not the bill: Azure charges for the time a replica is running, and any
request every few minutes keeps it running. That is capped by max replicas = 1 (about $79 a month if it never
stopped, of which the free grant covers the first 50 hours).

Measured with `scripts/perf_browser.py` in Microsoft Edge (median of 3 runs; `results/perf/`). Before: the live Pages
site, browser runtime. Server, local: the Docker image limited to 1 vCPU and 512 MB, a fresh container per run, on
localhost. Live: the Pages site calling the Azure server from a laptop in India, server already awake (desktop; the phone
viewport gives the same numbers within 30 ms, `live_azure_phone.json`):

| | before (browser runtime) | server, local | live (Pages + Azure) |
|---|---|---|---|
| first answer after opening the page | 10.2 s | 0.31 s | 0.60 s |
| downloaded on first visit | 37.6 MB | 0.26 MB | 0.3 MB |
| new field (network + solver) | 404 ms | 123 ms | 188 ms |
| paint a stroke, re-solve | 511 ms | 179 ms | 308 ms |
| change the source β | 387 ms | 4 ms (local rescale) | 4 ms |
| switch network | 389 ms | 64 ms (solver result reused) | 151 ms |
| 256 x 256 field | 1.9 s | 0.54 s | 0.75 s |
| batch of 32 / 64 fields | 9.8 s / 17.6 s | 2.2 s / 3.5 s | 2.5 s / 4.2 s |
| investigation | not available | 0.86 s | 0.85 s |

Cold start (`scripts/cold_start.py`, `results/perf/live_azure_cold*.json`): after about 5 minutes without requests
the app has no replica, and the first request waits about 25 s for one to start. The page then answers from the
in-browser runtime after 12.7 s (desktop) and 11.0 s (phone viewport), moves to the server at 22 s and 24 s, and the
first solve there takes 1.1 to 1.3 s; later solves are as in the table. Before the hand-off was added, a cold
session stayed on the browser runtime (batch of 32: 8.6 s instead of 2.5 s, `live_azure_cold_desktop.json`).

Three problems were found while measuring: the previous image could not start (it never copied `models.json`);
under a 1-vCPU quota OpenBLAS started one thread per host core and the network took 200 ms per field instead of
43 ms (the API now sizes BLAS threads from the cgroup quota); and the GELU and FFTs dominated the forward pass.
Hugging Face Docker Spaces need a paid plan (`402 Payment Required`), and the Railway trial had expired.
`scripts/ui_check.py` checks both runtimes on desktop and phone widths: examples, painting, beta, model switch, investigation, batch, evidence, a failed request,
and horizontal overflow.

## Data and licences

PDEBench (Takamoto et al., NeurIPS 2022 Datasets and Benchmarks), data from DaRUS, University of Stuttgart.
`results/audit_*_samples.npz` contains 8 PDEBench samples for the audit figure.
