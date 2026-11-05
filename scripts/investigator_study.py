"""Does the served investigator find why a served model is wrong, and would an LLM choose experiments better?

Cases have known causes, defined against each model's own training domain (models.json):
  contrast    contrast 3-30x beyond the model's maximum         grid        64 or 256 cells (training: 128)
  continuous  smoothly varying values (only for a two-valued model)
  geometry    shapes the model never saw: channels or layers (PDEBench model), layers (broad model)
  variety     (control, truth = no cause) in-domain inputs other than PDEBench blobs: channels, smooth
              fields, contrast up to the maximum (broad model only)
singly, in pairs, and unchanged controls. Every experiment is run once per case (training-like baseline,
add-back-one-axis, geometry comparison) and stored, so rule variants are scored from the same evidence.

Policies
  flags   no experiments: the app's input flags, plus geometry when nothing is flagged but the 90% bound
          on the residual estimate exceeds 3x the in-distribution median error
  rule    flowguard.diagnose.investigate as served (ablation + geometry experiment); its two thresholds are
          chosen on the calibration split (both models pooled) and scored on the evaluation split
  llm     (--llm MODEL, needs ANTHROPIC_API_KEY) a language model picks experiments one at a time with the same
          tools and names the causes; evaluation split only
Score: exact match of the cause set; cost in units of one network run + one reference solve.

Usage: python scripts/investigator_study.py [--n-cal 6] [--n-eval 10] [--llm claude-opus-5-5]
Output: results/investigator_study.json (and results/investigator_study_<llm>.json with --llm)
"""
import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flowguard.core import input_flags  # noqa: E402
from flowguard.diagnose import COST, attribute, network_error, run_experiment  # noqa: E402
from flowguard.fields import channels, grf, layers, pdebench  # noqa: E402
from flowguard.guard import fft_estimates  # noqa: E402
from flowguard.infer import Predictor  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FLAG_TO_CAUSE = {"contrast": "contrast", "resolution": "grid", "continuous": "continuous"}
RATIOS, GEO_RATIOS = (1.25, 1.5, 2.0, 3.0), (1.5, 2.0, 3.0, 4.0)


def categories(dom):
    if dom["continuous"]:
        return [(), ("variety",), ("contrast",), ("grid",), ("geometry",), ("contrast", "grid"),
                ("contrast", "geometry"), ("grid", "geometry")]
    return [(), ("contrast",), ("grid",), ("continuous",), ("geometry",), ("contrast", "grid"),
            ("contrast", "geometry"), ("grid", "continuous"), ("continuous", "geometry")]


def make_case(rng, factors, dom):
    n = int(rng.choice([64, 256])) if "grid" in factors else 128
    cmax = dom["contrast_max"]
    if "contrast" in factors:
        c = 10 ** rng.uniform(np.log10(3 * cmax), np.log10(30 * cmax))
    elif "variety" in factors:
        c = 10 ** rng.uniform(1, np.log10(cmax))
    else:
        c = 10.0
    lo = 1 / c
    if "geometry" in factors:
        shape = "layers" if dom["continuous"] or rng.random() < 0.5 else "channels"
    elif "variety" in factors:
        shape = str(rng.choice(["channels", "grf", "pdebench"]))
    else:
        shape = "pdebench"
    if shape == "layers":
        a = layers(1, rng, n=n, a_low=lo)[0]
    elif shape == "channels":
        a = channels(1, rng, n=n, a_low=lo)[0]
    elif shape == "grf":
        a = grf(1, rng, n=n, a_low=lo)[0]
    else:
        a = pdebench(200, rng, n=n, a_low=lo)[int(rng.integers(200))]
    if "continuous" in factors:
        a = np.exp(gaussian_filter(np.log(a), sigma=3 * n / 128))
    return a, shape


def collect(pred, a, dom):
    """Every experiment the rule can use, run once."""
    err0 = network_error(pred, a)
    ev = [run_experiment(pred, a, "all", dom)]
    if ev[0]["applicable"] and len(ev[0]["undone"]) > 1:
        ev += [run_experiment(pred, a, f"keep_{k}", dom) for k in ev[0]["undone"]]
    ev.append(run_experiment(pred, a, "geometry", dom))
    return err0, ev


def rule_from_evidence(err0, ev, id_err, ratio, geo_ratio):
    """The served policy replayed on stored evidence: the geometry experiment runs only when the baseline is at
    least geo_ratio x id_err (diagnose.investigate), so it is dropped (and not charged) otherwise."""
    base = ev[0]["err_after"] if ev[0]["applicable"] else err0
    used = [e for e in ev if e["experiment"] != "geometry" or base >= geo_ratio * id_err]
    causes = attribute(err0, used, id_err, ratio, geo_ratio)
    return sorted(causes), sum(COST[e["experiment"]] for e in used if e["applicable"])


def policy_flags(a, est90, dom, id_err):
    causes = {FLAG_TO_CAUSE[f["code"]] for f in input_flags(a, None, dom) if f["code"] in FLAG_TO_CAUSE}
    if not causes and est90 > 3 * id_err:
        causes.add("geometry")
    return sorted(causes)


def llm_system(dom, id_err):
    vals = "two-valued and smoothly varying fields" if dom["continuous"] else "two-valued fields only"
    axes = ["contrast", "grid"] + ([] if dom["continuous"] else ["continuous"])
    return f"""You investigate why a neural operator's prediction of steady Darcy flow is inaccurate.
The network was trained on {dom['grid']}x{dom['grid']} fields with contrast (max/min permeability) up to {dom['contrast_max']:g}, {vals}.
On inputs like its training data its median relative error is {id_err:.2%}.
Possible causes: {', '.join(axes)} (the input is outside the training range on that axis) and geometry (shapes unlike
its training data). Several can apply, or none. Tools run experiments and report the network's error on an edited input:
  {' / '.join(axes)}   move that one axis back into the training range
  all                  move every off-training axis back at once (a training-like baseline, same shape)
  {' / '.join('keep_' + k for k in axes)}   move every off-training axis back except that one
  geometry             replace the shape of the training-like baseline by training-family shapes with the same values
                       and grid (median error over 4); costs 4
Each other experiment costs 1. Factors can interact: one large factor can hide another.
Spend as few experiments as you can, then call `conclude`."""


def policy_llm(client, pred, a, dom, id_err, est90, model):
    axes = ["contrast", "grid"] + ([] if dom["continuous"] else ["continuous"])
    names = axes + ["all"] + [f"keep_{k}" for k in axes] + ["geometry"]
    obs = {"grid": a.shape[0], "contrast": round(float(a.max() / a.min()), 1),
           "distinct_values": int(min(np.unique(np.round(a, 6)).size, 99)),
           "flags": [f["msg"] for f in input_flags(a, None, dom)],
           "estimated_error_upper_90": round(est90, 4), "measured_error_vs_reference": round(network_error(pred, a), 4)}
    tools = [{"name": "experiment", "description": "Run one experiment on the input.", "strict": True,
              "input_schema": {"type": "object", "additionalProperties": False, "required": ["name"],
                               "properties": {"name": {"type": "string", "enum": names}}}},
             {"name": "conclude", "description": "Name the causes (possibly none) and end the investigation.", "strict": True,
              "input_schema": {"type": "object", "additionalProperties": False, "required": ["causes"],
                               "properties": {"causes": {"type": "array", "items": {"type": "string", "enum": axes + ["geometry"]}}}}}]
    msgs = [{"role": "user", "content": json.dumps(obs) + "\nAnswer only through the tools; finish with `conclude`."}]
    cost, usage = 0, {"input_tokens": 0, "output_tokens": 0}
    for _ in range(10):
        r = client.messages.create(model=model, max_tokens=16000, system=llm_system(dom, id_err), tools=tools,
                                   tool_choice={"type": "auto"}, thinking={"type": "adaptive"}, messages=msgs)
        usage["input_tokens"] += r.usage.input_tokens
        usage["output_tokens"] += r.usage.output_tokens
        if r.stop_reason == "refusal":
            return {"causes": [], "cost": cost, "usage": usage, "note": "refused"}
        msgs.append({"role": "assistant", "content": r.content})
        results = []
        for block in r.content:
            if block.type != "tool_use":
                continue
            if block.name == "conclude":
                return {"causes": sorted(set(block.input.get("causes", []))), "cost": cost, "usage": usage}
            ev = run_experiment(pred, a, block.input["name"], dom)
            cost += COST[block.input["name"]] if ev["applicable"] else 0
            results.append({"type": "tool_result", "tool_use_id": block.id, "content": json.dumps(ev)})
        msgs.append({"role": "user", "content": results or "Use the tools: run an experiment or call `conclude`."})
    return {"causes": [], "cost": cost, "usage": usage, "note": "no conclusion within 10 turns"}


def score(rows, key):
    return {"exact_match": float(np.mean([set(r[key]["causes"]) == set(r["truth"]) for r in rows])),
            "mean_cost": float(np.mean([r[key]["cost"] for r in rows])), "n": len(rows)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-cal", type=int, default=6)
    ap.add_argument("--n-eval", type=int, default=10)
    ap.add_argument("--llm")
    args = ap.parse_args()
    registry = json.loads((ROOT / "models.json").read_text())
    client = None
    if args.llm:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            sys.exit("--llm needs ANTHROPIC_API_KEY")
        import anthropic
        client = anthropic.Anthropic()

    cases = {}
    t0 = time.time()
    for name, entry in registry.items():
        pred = Predictor([str(ROOT / w) for w in entry["weights"]])
        rel = json.loads((ROOT / entry["reliability"]).read_text())
        dom, id_err, q90 = entry["domain"], rel["sets"]["pdebench_test"]["err"]["median"], rel["conformal_q90_fft"]
        for split, n_per, seed in (("cal", args.n_cal, 2024), ("eval", args.n_eval, 31337)):
            rng = np.random.default_rng(seed)
            for cat in categories(dom):
                for _ in range(n_per):
                    a, shape = make_case(rng, cat, dom)
                    u_hat = pred.predict(a)[0]
                    est90 = q90 * float(fft_estimates(a[None], 1.0, u_hat[None], pred.face, pred.bc)[0]["fft_err_rel"][0])
                    err0, ev = collect(pred, a, dom)
                    row = {"model": name, "split": split, "category": "+".join(cat) or "none", "shape": shape,
                           "truth": sorted(set(cat) - {"variety"}), "err": err0, "id_err": id_err, "evidence": ev,
                           "flags": {"causes": policy_flags(a, est90, dom, id_err), "cost": 0}}
                    if client and split == "eval":
                        row["llm"] = policy_llm(client, pred, a, dom, id_err, est90, args.llm)
                    cases.setdefault(split, []).append(row)
            print(name, split, f"{len(cases[split])} cases so far, {time.time() - t0:.0f} s", flush=True)

    # choose the rule's thresholds on the calibration split (both models pooled): exact match, then lower cost
    grid = []
    for ratio, geo in itertools.product(RATIOS, GEO_RATIOS):
        res = [rule_from_evidence(r["err"], r["evidence"], r["id_err"], ratio, geo) for r in cases["cal"]]
        acc = float(np.mean([set(c) == set(r["truth"]) for (c, _), r in zip(res, cases["cal"])]))
        grid.append({"ratio": ratio, "geo_ratio": geo, "cal_exact_match": acc, "cal_mean_cost": float(np.mean([k for _, k in res]))})
    best = max(grid, key=lambda g: (g["cal_exact_match"], -g["cal_mean_cost"]))
    for split in cases:
        for r in cases[split]:
            for key, (ra, ge) in (("rule", (best["ratio"], best["geo_ratio"])), ("rule_default", (1.5, 2.0))):
                c, k = rule_from_evidence(r["err"], r["evidence"], r["id_err"], ra, ge)
                r[key] = {"causes": c, "cost": k}

    pols = ["flags", "rule", "rule_default"] + (["llm"] if client else [])
    out = {"thresholds": best, "threshold_grid": grid, "results": {}}
    for split in ("cal", "eval"):
        for name in registry:
            rows = [r for r in cases[split] if r["model"] == name]
            out["results"][f"{split}/{name}"] = {
                "policies": {p: score(rows, p) for p in pols if all(p in r for r in rows)},
                "per_category": {c: {"n": len(sel), "median_err": float(np.median([r["err"] for r in sel])),
                                     **{p: float(np.mean([set(r[p]["causes"]) == set(r["truth"]) for r in sel])) for p in pols if all(p in r for r in sel)}}
                                 for c in dict.fromkeys(r["category"] for r in rows) for sel in [[r for r in rows if r["category"] == c]]}}
    print(json.dumps({"thresholds": best, **{k: v["policies"] for k, v in out["results"].items()}}, indent=1))
    tag = f"_{args.llm}" if args.llm else ""
    Path(ROOT / f"results/investigator_study{tag}.json").write_text(json.dumps({**out, "rows": cases}, indent=1, default=float))
