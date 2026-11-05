# FlowGuard

Steady Darcy flow from Fourier neural operators, with every answer checked against an independent finite-volume
solver, and cheap indicators that say when a network should not be trusted.

**Live app:** https://g26karthik.github.io/flowguard/ (runs the Python package in your browser; see [Deployment](#deployment)).

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

Same network, same training cost, same fit to its targets: the physical error drops 5.6x. The PDEBench-scheme network
was never limited by learning; it was faithfully learning a discretisation that is 3% off. The app calls the
finite-volume solution the "reference", states its discretisation error, and always solves with the discretisation
recorded in the model checkpoint.

### 3. Accuracy in and out of distribution; adaptation

Median error against the model's own reference (`results/reliability_*.json`):

| Inputs | PDEBench model | Adapted model |
|---|---|---|
| PDEBench test split (1,000) | **0.54%** | 0.76% |
| Fresh fields from a replica of PDEBench's sampler (KS 0.02) | 0.60% | 0.81% |
| Contrast 100x (training: 10x) | 76% | **3.1%** |
| Smooth log-normal fields | 9.4% | **1.6%** |
| Thin channels | 11.7% | **2.6%** |
| 64x64 / 256x256 grids | 2.9% / 1.6% | 2.5% / 1.6% |
| Tilted layers (held out from both) | 6.8% | **4.3%** |
| Contrast 3000x (held out from both) | 99% | 39% |

The adapted model is fine-tuned from the PDEBench model on PDEBench fields plus 5,700 broader fields (contrast up to
1000x, channels, smooth fields; `scripts/make_adapt_data.py`). It costs 0.2 percentage points on PDEBench-like fields,
less than the PDEBench scheme's own 2.6% discretisation error. It removes most of the error on the families it saw
and helps on held-out layers. It does not extrapolate in contrast. Three seeds of the PDEBench model give test
medians of 0.56, 0.56 and 0.59% in their training-time evaluation (GPU, TF32); seed 0 is 0.54% in exact fp32.

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
  across input families; nothing cheap detects a grid change, so the app flags that from the input.
- A split-conformal 90% bound on the DST estimate (calibrated on the validation split) covers 90% of the test split.
  On shifted inputs coverage ranges from 20% (smooth fields) and 32% (64² grid) to 100% (channels, where the bound
  is loose), so the app shows it with a caveat when inputs are flagged.
- The Galerkin estimate is the best indicator but costs about half a reference solve, so solving directly is better.
- Pooled "solve the riskiest 20%" triage mostly measures separation between input families, where every indicator
  except the raw residual does about equally (`results/triage_unflagged.txt`); within a family, the table above is
  what matters.

### 5. Should an LLM agent investigate unreliable predictions?

`flowguard/diagnose.py` implements counterfactual experiments: undo one off-training axis (contrast, grid,
continuous values) and re-measure; a training-like baseline with everything undone; and "add back one axis". On 72
cases with known causes (`scripts/agent_study.py`, `results/agent_study_fno_steady_s0.json`):

| Policy | exact cause set | experiments per case |
|---|---|---|
| Input flags only | 75% | 0 |
| First fixed rule (undo each axis alone) | 56% | 1.8 |
| Ablation rule (baseline, then add back each axis) | **85%** | 1.2 |

The first rule failed on unchanged fields and missed factors masked by a dominant one. The ablation design fixed
both. The remaining misses are cases where an applied factor does not hurt (blurred channels are easier than sharp
ones), so the construction label is not a causal truth. The rule already uses the minimum number of experiments for
its evidence, and each costs about 0.2 s. An LLM policy is implemented with the same tools (`--llm`), but it was not
run: no Anthropic credentials were available in this session. On this evidence an LLM can only match the rule, at
a cost of seconds and money per case, so the app uses the rule.

### 6. Is the network worth it here?

Cost per field (`results/bench_solver.json`, `results/bench_network.json`; laptop RTX 4060 and Ryzen 9 8945HS):

| Grid | sparse direct solve | FNO, torch GPU (batched) | FNO, torch CPU | FNO, NumPy (served) |
|---|---|---|---|---|
| 128² | 48 ms | 1.2 ms | 26 ms | 170 ms |
| 256² | 274 ms | 5.2 ms | 119 ms | 625 ms |
| 512² | 2.5 s | 17 ms | 480 ms | 2.4 s |

On a GPU the network is 39x (128²) to 148x (512²) faster than the direct solve; at 256² and 512² it runs off its
training grid. The free deployment runs a NumPy forward pass in the browser, which is slower than the direct solve:
there, the value is the reliability workflow, not speed.

## Repository

| Path | What |
|---|---|
| `flowguard/solver.py` | finite-volume solver, matrix-free operator, transient propagator |
| `flowguard/model.py`, `scripts/train.py` | FNO (torch) and training (`--init`, `--extra`, `--disc` for adaptation and experiment B) |
| `flowguard/npfno.py`, `scripts/export_numpy.py` | NumPy forward pass used for serving (max relative difference to torch fp32: 4e-7) |
| `flowguard/guard.py` | residual, DST and Galerkin error indicators |
| `flowguard/diagnose.py` | counterfactual diagnostic experiments and the ablation rule |
| `flowguard/fields.py` | PDEBench sampler replica and shifted families (`layers` is held out from all training) |
| `flowguard/core.py`, `flowguard/api.py`, `models.json` | service logic, model registry, FastAPI wrapper |
| `web/` | front end; `worker.js` runs the same package in the browser via Pyodide |
| `scripts/` | every experiment in this README; `ui_check.py` drives the app in Microsoft Edge |

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
python -m pytest -q
uvicorn flowguard.api:app --port 8000                               # server mode
python scripts/build_site.py && python -m http.server -d _site      # browser mode
```

Seeds: data split `numpy.default_rng(0)`, training `--seed`, evaluation sets `default_rng(12345)`, adaptation fields
`default_rng(777)`. About 31 s per epoch on the laptop GPU. On Windows PowerShell, hide the GPU with
`$env:CUDA_VISIBLE_DEVICES = "-1"`: assigning an empty string deletes the variable (an evaluation in this project ran
on the GPU with cuDNN TF32 because of that, which moved the test median from 0.54% to 0.56%).

## Deployment

Hugging Face requires a paid plan for Docker Spaces (`402 Payment Required` from the API), and free container hosts
offer about 0.1 vCPU. The live app is a static GitHub Pages site that runs the same `flowguard` package (solver,
indicators, NumPy network, service logic) in the browser through Pyodide. The FastAPI server and the `Dockerfile`
(591 MB image, verified) run the identical code server-side; the UI uses a server when one is present.
`scripts/ui_check.py` checks both modes and the live URL: field generation, painting, the error-map toggle, batch
study, evidence, and phone width.

## Data and licences

PDEBench (Takamoto et al., NeurIPS 2022 Datasets and Benchmarks), data from DaRUS, University of Stuttgart.
`results/audit_*_samples.npz` contains 8 PDEBench samples for the audit figure.
