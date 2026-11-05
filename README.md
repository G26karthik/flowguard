# FlowGuard

Steady Darcy flow from a Fourier neural operator, with every answer checked against an independent
finite-volume solver, and cheap indicators that say when the network should not be trusted.

**Live app:** https://g26karthik.github.io/flowguard/ (runs the Python package in your browser; see [Deployment](#deployment)).

Built for the QUEST Lab (IISc) Full Stack AI pre-doctoral application, project 3, "PDE solutions as a service",
in a single Claude Code session. The git history and the session log line up.

## The problem

$$-\nabla\cdot(a(x)\nabla u(x)) = \beta \ \text{ on } (0,1)^2, \qquad u = 0 \text{ on the boundary.}$$

Users give a permeability field `a` (generated, painted, or uploaded) and a source `β`. They get the
network's pressure field, a reference finite-volume solution on the same grid, the difference, and flags when
the input is unlike the training data. Because the problem is linear in `β`, one model trained at `β = 1`
serves every `β` exactly (`u_β = β u_1`).

## What the evidence shows

All numbers come from scripts in `scripts/`, with outputs in `results/` and `runs/`. Relative L2 errors.

### 1. PDEBench's Darcy labels are transient snapshots, not steady states

PDEBench describes its Darcy data as the steady state of the equation. The generator code
(`data_gen_NLE/ReactionDiffusionEq`) instead integrates `u_t = div(a grad u) + β` from a near-constant random
initial level `u0 ~ U(0.1, 10)` and stores a snapshot at a fixed time.

- Re-solving all 10,000 fields with the matched discretisation: for `β = 1`, 9.5% of labels are more than 10% away
  from the steady solution; for `β = 0.01`, the median label is 17% away. `β = 100` is close (max 3.3%).
  (`results/labelcheck_*.json`, `scripts/resolve_dataset.py`)
- Replaying the generator (`scripts/replicate_generation.py`) explains 99.8 to 99.98% of every label's deviation:
  the labels are the `t = 1.75` snapshot. Fitted `u0` follow `U(0.1, 10)` at `t = 1.75` (KS 0.07) but not at
  `t = 2.0` (KS 0.70, negative `u0` at `β = 100`), and the same sample gives the same `u0` in every `β` file
  (r = 0.99992). (`results/replicate_generation.json`, `results/fig_label_audit.png`)
- So the labels are faithful outputs of the documented generator. They are only "wrong" as steady states, and for
  slowly-decaying fields they carry the random `u0`, which is not a model input.
- Consequence, measured: the same FNO trained on PDEBench's labels has 3.3% mean error against the steady solution;
  trained on re-solved steady targets, 0.76%. (`runs/fno_label_s0`, `runs/fno_steady_s0`)
- Also found: the generator puts `u = 0` at ghost-cell centres, half a cell outside the domain, not on the boundary
  as the paper states.

### 2. The reference solver is an approximation too

The finite-volume reference is not the exact PDE solution (`scripts/discretization_error.py`):

| Discretisation on 128x128 (pixel-wise coefficient) | difference from 4x refined | observed order |
|---|---|---|
| arithmetic faces, ghost-centre wall (PDEBench's, and the network's target) | 2.6% | 1.0 |
| harmonic faces, wall on the boundary face | 0.21% | 1.3 |

The network reproduces PDEBench's discrete model (0.56% median) more closely than that model reproduces the PDE.
The app therefore calls it the "reference", states its discretisation error, and the solver always uses the
discretisation recorded in the model checkpoint.

### 3. Accuracy in and out of distribution (seed 0)

| Inputs | median error vs reference | 90th pct |
|---|---|---|
| PDEBench test split (1,000) | 0.56% | 1.4% |
| Fresh fields from a replica of PDEBench's sampler (KS 0.02 vs dataset) | 0.60% | 1.5% |
| Contrast 100x instead of 10x | 76% | 85% |
| Smooth log-normal fields | 9.4% | 15% |
| Thin channels | 11.7% | 20% |
| 64x64 / 256x256 grids | 2.9% / 1.6% | |

(`results/reliability_fno_steady_s0.json`)

### 4. Do the cheap checks know when the network is wrong?

- A DST-preconditioned residual estimate (about 2 ms per field) ranks errors with Spearman 0.74 to 0.96 on every
  input family, but not across resolution changes (-0.21 and 0.08).
- Raw residual norm is a poor indicator (Spearman -0.27 on channels).
- A split-conformal 90% bound calibrated on the validation split covers 89.9% of the test split and fails out of
  distribution (20% to 53% coverage), so the app shows it only with an explicit caveat when inputs are flagged.
- Budget triage over all sets pooled: solving the top 20% ranked by the estimate finds 70% of predictions with error
  above 5% (random: 20%, perfect ranking: 72%). The pool is dominated by out-of-distribution sets where bad
  predictions are easy to find.
- A two-level Galerkin estimate ranks errors almost perfectly (Spearman 0.92 to 0.998) and can correct most of the
  error, but it costs about 55% of a full sparse direct solve, so solving directly is better. Not used in the app.

### 5. Is the network worth it here?

At 128x128 a sparse direct solve takes about 50 ms. The NumPy network (used for serving) takes about 150 ms per field
on a laptop CPU and about 200 ms in the browser. In this setting the network is not a faster solver; the value of
the app is the reliability workflow around it, and the measurements above. The torch model on GPU is faster in
batch, and the direct solve grows to 2.5 s at 512x512 (`results/bench_solver.json`).

## Repository

| Path | What |
|---|---|
| `flowguard/solver.py` | finite-volume solver, matrix-free operator, transient propagator |
| `flowguard/model.py`, `scripts/train.py` | FNO (torch) and training |
| `flowguard/npfno.py`, `scripts/export_numpy.py` | NumPy forward pass used for serving (matches torch to 4e-7) |
| `flowguard/guard.py` | residual, DST and Galerkin error indicators |
| `flowguard/fields.py` | PDEBench sampler replica and shifted families |
| `flowguard/core.py`, `flowguard/api.py` | service logic and FastAPI wrapper |
| `web/` | front end; `worker.js` runs the same package in the browser via Pyodide |
| `scripts/` | every experiment in this README |
| `STATUS.md` | running log of state and next steps |

## Reproduce

```bash
uv venv .venv --python 3.11
uv pip install --python .venv -r requirements-lock.txt   # CUDA torch; or install torch for your platform first
# data: the three PDEBench 2D Darcy files listed in PDEBench's pdebench_data_urls.csv (md5-checked), into data/raw/
python scripts/resolve_dataset.py data/raw/2D_DarcyFlow_beta1.0_Train.hdf5   # steady targets, ~5 min
python scripts/train.py --target steady --seed 0 --epochs 100 --out runs/fno_steady_s0
python scripts/export_numpy.py runs/fno_steady_s0/model.pt
python scripts/eval_reliability.py runs/fno_steady_s0/model.pt
python -m pytest -q
uvicorn flowguard.api:app --port 8000          # server mode
python scripts/build_site.py && python -m http.server -d _site   # browser mode
```

Seeds: data split `numpy.default_rng(0)`, training `--seed`, evaluation sets `default_rng(12345)`.
Hardware used: RTX 4060 Laptop GPU (8 GB), Ryzen 9 8945HS, about 31 s per epoch.

## Deployment

Hugging Face now requires a paid plan for Docker Spaces (the API returned `402 Payment Required`), and free
container hosts give about 0.1 vCPU. The live app is therefore a static GitHub Pages site that runs the same
`flowguard` package (solver, indicators, NumPy network, service logic) in the browser through Pyodide. The FastAPI
server and `Dockerfile` run the identical code server-side; the UI uses the server when one is present.
`scripts/ui_check.py` drives both modes in Microsoft Edge (field generation, painting, batch study, evidence,
phone width).

## Data and licences

PDEBench (Takamoto et al., NeurIPS 2022 Datasets and Benchmarks), data from DaRUS, University of Stuttgart.
`results/audit_*_samples.npz` contains 8 PDEBench samples for the audit figure.
