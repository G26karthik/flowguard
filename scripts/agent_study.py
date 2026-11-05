"""Can an investigator choose diagnostic experiments better than fixed rules?

Cases have known causes: PDEBench-style fields with controlled changes applied
  contrast   (contrast 100-1000x)      grid        (64 or 256 cells)
  continuous (log-permeability blurred) geometry    (channels or layers instead of bumps)
singly, in pairs, and unchanged controls. Policies:
  P0 flags   no experiments: input flags, plus 'geometry' if no flag fires but the residual estimate is high
  P1 rule    every applicable counterfactual experiment, then flowguard.diagnose.attribute
  P2 llm     an LLM picks experiments one at a time (tool use) and names the causes; needs ANTHROPIC_API_KEY
Score: exact match of the cause set; cost in experiments (1 = one network run + one reference solve).

Usage: python scripts/agent_study.py runs/<run>/weights.npz [--llm MODEL] [--n PER_CATEGORY]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flowguard.core import input_flags  # noqa: E402
from flowguard.diagnose import COST, attribute, investigate, network_error, run_experiment  # noqa: E402
from flowguard.fields import channels, layers, pdebench  # noqa: E402
from flowguard.guard import fft_estimates  # noqa: E402
from flowguard.infer import Predictor  # noqa: E402

FLAG_TO_CAUSE = {"contrast": "contrast", "resolution": "grid", "continuous": "continuous"}


def make_case(rng, factors):
    n = int(rng.choice([64, 256])) if "grid" in factors else 128
    a_low = 10 ** rng.uniform(-3, -2) if "contrast" in factors else 0.1
    if "geometry" in factors:
        a = channels(1, rng, n=n, a_low=a_low)[0] if rng.random() < 0.5 else layers(1, rng, n=n, a_low=a_low)[0]
    else:
        a = pdebench(200, rng, n=n, a_low=a_low)[int(rng.integers(200))]
    if "continuous" in factors:
        a = np.exp(gaussian_filter(np.log(a), sigma=3 * n / 128))
    return a


def policy_flags(pred, a, q90):
    u_hat = pred.predict(a)[0]
    ind, _ = fft_estimates(a[None], 1.0, u_hat[None], pred.face, pred.bc)
    causes = {FLAG_TO_CAUSE[f["code"]] for f in input_flags(a, None) if f["code"] in FLAG_TO_CAUSE}
    if not causes and q90 * float(ind["fft_err_rel"][0]) > 3 * 0.006:
        causes.add("geometry")
    return {"causes": sorted(causes), "cost": 0}


LLM_SYSTEM = """You investigate why a neural operator's prediction of steady Darcy flow is inaccurate.
The network was trained only on 128x128 fields with two permeability values and contrast 10 (PDEBench).
Possible causes: contrast, grid, continuous, geometry (shapes unlike PDEBench's blobs). Several can apply.
Tools run counterfactual experiments: each edits the input back to the training setting along one axis and
reports the network's error afterwards (cost 1 each); 'all' undoes every applicable axis at once (cost 1).
An axis is a cause if undoing it alone cuts the error substantially; geometry is a cause if the error stays
high (> ~2%) after undoing every applicable axis. Spend as few experiments as you can, then call `conclude`."""


def policy_llm(pred, a, model, q90):
    import anthropic

    client = anthropic.Anthropic()
    u_hat = pred.predict(a)[0]
    ind, _ = fft_estimates(a[None], 1.0, u_hat[None], pred.face, pred.bc)
    obs = {"grid": a.shape[0], "contrast": round(float(a.max() / a.min()), 1),
           "distinct_values": int(min(np.unique(np.round(a, 6)).size, 99)),
           "flags": [f["msg"] for f in input_flags(a, None)],
           "estimated_error_upper_90": round(q90 * float(ind["fft_err_rel"][0]), 4),
           "measured_error_vs_reference": round(network_error(pred, a), 4)}
    tools = [{"name": "experiment", "description": "Run one counterfactual experiment.", "strict": True,
              "input_schema": {"type": "object", "additionalProperties": False, "required": ["name"],
                               "properties": {"name": {"type": "string", "enum": ["contrast", "grid", "continuous", "all"]}}}},
             {"name": "conclude", "description": "Name the causes and end the investigation.", "strict": True,
              "input_schema": {"type": "object", "additionalProperties": False, "required": ["causes"],
                               "properties": {"causes": {"type": "array", "items": {"type": "string", "enum": ["contrast", "grid", "continuous", "geometry"]}}}}}]
    # forced tool_choice (any/tool) is a 400 on current models: use auto and say so in the prompt
    msgs = [{"role": "user", "content": json.dumps(obs) + "\nAnswer only through the tools; finish with `conclude`."}]
    cost, usage = 0, {"input_tokens": 0, "output_tokens": 0}
    for _ in range(8):
        r = client.beta.messages.create(
            model=model, max_tokens=16000, system=LLM_SYSTEM, tools=tools, tool_choice={"type": "auto"},
            output_config={"effort": "medium"}, betas=["server-side-fallback-2026-07-01"], fallbacks="default",
            messages=msgs)
        usage["input_tokens"] += r.usage.input_tokens
        usage["output_tokens"] += r.usage.output_tokens
        if r.stop_reason == "refusal":
            return {"causes": [], "cost": cost, "usage": usage, "note": "refused"}
        msgs.append({"role": "assistant", "content": r.content})  # full content: thinking blocks must be replayed
        results = []
        for block in r.content:
            if block.type != "tool_use":
                continue
            if block.name == "conclude":
                return {"causes": sorted(set(block.input.get("causes", []))), "cost": cost, "usage": usage}
            ev = run_experiment(pred, a, block.input["name"])
            cost += COST[block.input["name"]] if ev["applicable"] else 0
            results.append({"type": "tool_result", "tool_use_id": block.id, "content": json.dumps(ev)})
        msgs.append({"role": "user", "content": results or "Use the tools: run an experiment or call `conclude`."})
    return {"causes": [], "cost": cost, "usage": usage, "note": "no conclusion within 8 turns"}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("weights", nargs="+")
    ap.add_argument("--llm")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--reliability", default="results/reliability_fno_steady_s0.json")
    args = ap.parse_args()
    pred = Predictor(args.weights)
    q90 = json.loads(Path(args.reliability).read_text())["conformal_q90_fft"]
    cats = [(), ("contrast",), ("grid",), ("continuous",), ("geometry",), ("contrast", "grid"),
            ("contrast", "geometry"), ("grid", "continuous"), ("continuous", "geometry")]
    rng = np.random.default_rng(31337)
    rows = []
    for cat in cats:
        for _ in range(args.n):
            a = make_case(rng, cat)
            p1 = investigate(pred, a)
            row = {"truth": sorted(cat), "err": p1["err"], "rule": {"causes": p1["causes"], "cost": p1["cost"]},
                   "flags": policy_flags(pred, a, q90),
                   "effects": {e["experiment"]: round(p1["err"] / max(e["err_after"], 1e-12), 2) for e in p1["evidence"] if e["applicable"]}}
            if args.llm:
                row["llm"] = policy_llm(pred, a, args.llm, q90)
            rows.append(row)
        print(cat, "median err %.3f" % np.median([r["err"] for r in rows[-args.n:]]), flush=True)

    out = {"weights": args.weights, "n_cases": len(rows), "policies": {}, "per_category": {}}
    for pol in ["flags", "rule"] + (["llm"] if args.llm else []):
        acc = [set(r[pol]["causes"]) == set(r["truth"]) for r in rows]
        out["policies"][pol] = {"exact_match": float(np.mean(acc)), "mean_cost": float(np.mean([r[pol]["cost"] for r in rows]))}
    for cat in cats:
        sel = [r for r in rows if r["truth"] == sorted(cat)]
        out["per_category"]["+".join(cat) or "none"] = {
            "median_err": float(np.median([r["err"] for r in sel])),
            **{pol: float(np.mean([set(r[pol]["causes"]) == set(cat) for r in sel])) for pol in out["policies"]},
            "median_effect_of_undo": {k: float(np.median([r["effects"][k] for r in sel if k in r["effects"]]))
                                      for k in ("contrast", "grid", "continuous", "all") if any(k in r["effects"] for r in sel)}}
    print(json.dumps(out, indent=2))
    tag = Path(args.weights[0]).parent.name + (f"_{args.llm}" if args.llm else "")
    Path(f"results/agent_study_{tag}.json").write_text(json.dumps({**out, "rows": rows}, indent=1))
