"""LLM investigator pilot: an LLM choosing experiments vs the served deterministic investigator, on fresh cases.

PROTOCOL (fixed before any LLM call; the commit that adds this file precedes every results file it writes)

Cases. 10 fresh cases, 5 per served model (models.json), seed 20261106 (never used by earlier studies: those used
2024 and 31337). For each model, 5 categories drawn without replacement from investigator_study.categories(), one
case each, built by investigator_study.make_case (contrast 3-30x beyond the model's maximum, 64 or 256 grid,
smoothed values, unseen shapes; singly and in pairs; unchanged and in-domain-variety controls).

Ground truth: which factors genuinely caused this case's error. Primary, per case, by intervention:
  F = the factors applied when the case was built (controls: none).
  For every subset S of F (factors kept), E(S) = the network's relative L2 error vs the reference solve on the case
  with the factors outside S undone: contrast, grid and continuous by the same edit operators the tools use
  (flowguard.diagnose.undo_axis); geometry by the median error over 16 training-family shapes (PDEBench bumps,
  seed 9001, never the tools' comparators) with the same values and grid, smoothed as make_case does when
  continuous is kept.
  phi_f = Shapley value of f on ln E(S) (adding f, averaged over all orders: neither "remove from the full input"
  nor "add to the clean baseline" is privileged, so a dominant factor neither hides nor inflates a weak one).
  f is a genuine cause iff phi_f >= ln 2: adding it at least doubles the error, on average over contexts.
  The oracle has no budget and uses comparators neither policy sees; neither policy's thresholds (rule: add-back
  ratio 1.5, geometry ratio 4 over 4 comparators) are used.
Secondary: investigator_study.causal_factors: an applied factor counts if the model's median error on the
  independent held-out set for that factor is >= 2x its in-distribution median (population level, not per case).

Policies, same evidence and budget. Both start from the input and its measured error against the reference, and
may use the same experiments (flowguard.diagnose.run_experiment): undo one axis, undo all, undo all but one axis,
geometry comparison. Costs: diagnose.COST (1 per network run + reference solve; geometry 4; an experiment that is
not applicable costs 0). Budget: 8 units per case (the rule needs at most 7).
  rule   flowguard.diagnose.investigate exactly as served by flowguard.core (geometry=True, ratio=1.5, geo_ratio=4).
  llm    the model picks experiments one at a time through tools and names the causes with `conclude`. It also sees
         the input's grid, contrast, number of distinct values, the app's flags and the 90% error-bound estimate.
         An experiment that would exceed the budget is refused (not run). At most 12 tool calls and 10 turns.
  flags  no experiments (the app's input flags): a zero-cost reference point.

Metrics (per policy):
  correctness       exact match of the cause set with the primary truth (headline); also with the secondary
                    truth; per-factor precision and recall pooled over cases.
  tool calls        experiments requested (the rule's experiments run; the LLM's experiment calls).
  unnecessary       requested experiments that were not applicable, repeated an earlier one on the same case, were
                    refused for budget, or were requested in the same turn as `conclude` (their results could not
                    inform it: they are not run or charged).
  experiment cost   units actually charged.
  latency           wall time per case from start to conclusion, including the error measurement and every
                    experiment; for the LLM also the time spent in API calls.
  tokens, cost      input/output/cache tokens from the API's usage fields; cost at list price for prompts under
                    100K tokens (Haiku 5.5 $0.10/$0.50, Sonnet 5.5 $2/$10 per MTok; cache reads 0.1x, writes 1.25x
                    input).
Decision rule for a larger run: the LLM is worth scaling only if its exact-match count is at least the rule's,
or within one case of it at lower experiment cost, with no more than one case lost to a protocol failure (no
conclusion, refused budget). 10 cases cannot separate policies statistically; this pilot looks for failure modes.

Usage (key from the environment or .env; never printed or stored):
  uv run --with anthropic python scripts/llm_pilot.py --model claude-haiku-5-5 --effort medium
Output: results/llm_pilot_<model>_<effort>.json
"""
import argparse
import itertools
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from investigator_study import categories, causal_factors, llm_system, make_case, policy_flags  # noqa: E402

from flowguard.diagnose import COST, _undo, investigate, network_error, network_errors, run_experiment  # noqa: E402
from flowguard.fields import pdebench  # noqa: E402
from flowguard.guard import fft_estimates  # noqa: E402
from flowguard.infer import Predictor  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SEED, PER_MODEL, BUDGET, MAX_CALLS, MAX_TURNS = 20261106, 5, 8, 12, 10
ORACLE_COMPARATORS, ORACLE_SEED = 16, 9001
PRICE = {"claude-haiku-5-5": (0.10, 0.50), "claude-sonnet-5-5": (2.0, 10.0)}  # $ per MTok in, out


def smooth(a):  # how make_case builds a continuous field
    return np.exp(gaussian_filter(np.log(a), sigma=3 * a.shape[0] / 128))


def oracle(pred, a, factors, dom):
    """Primary truth: Shapley value of each applied factor on ln(error) over the full factorial of kept factors."""
    F = sorted(factors)
    assert not {"contrast", "continuous"} <= set(F)  # never built together, so a kept smoothing has contrast 10
    E = {}
    for r in range(len(F) + 1):
        for S in itertools.combinations(F, r):
            b, _ = _undo(a, [k for k in F if k not in S and k != "geometry"], dom)
            if "geometry" in F and "geometry" not in S:
                rng = np.random.default_rng(ORACLE_SEED)
                if "continuous" in S:
                    comps = np.stack([smooth(x) for x in pdebench(ORACLE_COMPARATORS, rng, n=b.shape[0], a_low=0.1)])
                else:
                    comps = pdebench(ORACLE_COMPARATORS, rng, n=b.shape[0], a_low=float(b.min()), a_high=float(b.max()))
                E[S] = float(np.median(network_errors(pred, comps)))
            else:
                E[S] = network_error(pred, b)
    L = {S: math.log(e) for S, e in E.items()}
    phi = {}
    for f in F:
        rest = [k for k in F if k != f]
        phi[f] = sum(math.factorial(len(S)) * math.factorial(len(F) - len(S) - 1) / math.factorial(len(F))
                     * (L[tuple(sorted(S + (f,)))] - L[S])
                     for r in range(len(rest) + 1) for S in itertools.combinations(rest, r))
    return sorted(f for f in F if phi[f] >= math.log(2)), {"+".join(S) or "none": e for S, e in E.items()}, phi


def policy_rule(pred, a, dom, id_err):
    t = time.perf_counter()
    r = investigate(pred, a, domain=dom, id_err=id_err, geometry=True, ratio=1.5, geo_ratio=4.0)
    ev = r["evidence"]
    return {"causes": r["causes"], "cost": r["cost"], "tool_calls": len(ev),
            "unnecessary": sum(not e["applicable"] for e in ev), "experiments": [e["experiment"] for e in ev],
            "latency_s": time.perf_counter() - t}


def policy_llm(client, model, effort, pred, a, dom, id_err, q90):
    t0 = time.perf_counter()
    err0 = network_error(pred, a)
    u_hat = pred.predict(a)[0]
    est90 = q90 * float(fft_estimates(a[None], 1.0, u_hat[None], pred.face, pred.bc)[0]["fft_err_rel"][0])
    axes = ["contrast", "grid"] + ([] if dom["continuous"] else ["continuous"])
    names = axes + ["all"] + [f"keep_{k}" for k in axes] + ["geometry"]
    from flowguard.core import input_flags
    obs = {"grid": a.shape[0], "contrast": round(float(a.max() / a.min()), 1),
           "distinct_values": int(min(np.unique(np.round(a, 6)).size, 99)),
           "flags": [f["msg"] for f in input_flags(a, None, dom)],
           "estimated_error_upper_90": round(est90, 4), "measured_error_vs_reference": round(err0, 4),
           "experiment_budget": BUDGET}
    system = llm_system(dom, id_err) + (
        f"\nBudget: {BUDGET} cost units for this input. An experiment that would exceed the remaining budget is refused."
        "\nAn experiment on an axis that is already inside the training range returns applicable=false and costs 0.")
    tools = [{"name": "experiment", "description": "Run one experiment on the input. Returns the evidence and the budget left.",
              "strict": True, "input_schema": {"type": "object", "additionalProperties": False, "required": ["name"],
                                               "properties": {"name": {"type": "string", "enum": names}}}},
             {"name": "conclude", "description": "Name the causes (possibly none) and end the investigation.", "strict": True,
              "input_schema": {"type": "object", "additionalProperties": False, "required": ["causes"],
                               "properties": {"causes": {"type": "array", "items": {"type": "string", "enum": axes + ["geometry"]}}}}}]
    msgs = [{"role": "user", "content": json.dumps(obs) + "\nAnswer only through the tools; finish with `conclude`."}]
    usage = {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}
    out = {"causes": None, "cost": 0, "tool_calls": 0, "unnecessary": 0, "experiments": [], "api_calls": 0,
           "api_s": 0.0, "stop_reasons": [], "turns": [], "text": []}
    done = set()
    for _ in range(MAX_TURNS):
        t = time.perf_counter()
        r = client.messages.create(model=model, max_tokens=16000, system=system, tools=tools, messages=msgs,
                                   thinking={"type": "adaptive"}, output_config={"effort": effort})
        out["api_s"] += time.perf_counter() - t
        out["api_calls"] += 1
        out["stop_reasons"].append(r.stop_reason)
        for k in usage:
            usage[k] += getattr(r.usage, k, 0) or 0
        if r.stop_reason == "refusal":
            out["note"] = "refused"
            break
        msgs.append({"role": "assistant", "content": r.content})
        results = []
        calls = [b for b in r.content if b.type == "tool_use"]
        out["turns"].append([b.input.get("name", "conclude") for b in calls])
        out["text"] += [b.text.strip() for b in r.content if b.type == "text" and b.text.strip()]
        final = next((b for b in calls if b.name == "conclude"), None)
        if final:
            out["causes"] = sorted(set(final.input.get("causes", [])))
            blind = sum(b.name == "experiment" for b in calls)
            out["tool_calls"] += blind
            out["unnecessary"] += blind
            break
        for block in calls:
            name = block.input["name"]
            out["tool_calls"] += 1
            if out["tool_calls"] > MAX_CALLS:
                ev = {"error": "tool-call limit reached; call conclude"}
                out["unnecessary"] += 1
            elif out["cost"] + COST[name] > BUDGET:
                ev = {"error": f"refused: costs {COST[name]}, {BUDGET - out['cost']} units left"}
                out["unnecessary"] += 1
                out["experiments"].append(name + " (refused)")
            else:
                ev = run_experiment(pred, a, name, dom)
                out["unnecessary"] += (not ev["applicable"]) or name in done
                out["cost"] += COST[name] if ev["applicable"] else 0
                out["experiments"].append(name)
                done.add(name)
                ev = {**ev, "budget_left": BUDGET - out["cost"]}
            results.append({"type": "tool_result", "tool_use_id": block.id, "content": json.dumps(ev)})
        msgs.append({"role": "user", "content": results or "Use the tools: run an experiment or call `conclude`."})
    if out["causes"] is None:
        out.setdefault("note", f"no conclusion within {MAX_TURNS} turns")
        out["causes"] = []
    pin, pout = PRICE[model]
    out["usage"] = usage
    out["cost_usd"] = (pin * (usage["input_tokens"] + 0.1 * usage["cache_read_input_tokens"]
                              + 1.25 * usage["cache_creation_input_tokens"]) + pout * usage["output_tokens"]) / 1e6
    out["latency_s"] = time.perf_counter() - t0
    return out


def load_key():
    """ANTHROPIC_API_KEY from the environment, else from the gitignored .env; the value is never printed."""
    if not os.environ.get("ANTHROPIC_API_KEY") and (ROOT / ".env").exists():
        for line in (ROOT / ".env").read_text().splitlines():
            k, _, v = line.partition("=")
            if k.strip() == "ANTHROPIC_API_KEY" and v.strip():
                os.environ["ANTHROPIC_API_KEY"] = v.strip().strip('"').strip("'")
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def summarise(rows, policies):
    s = {}
    for p in policies:
        rs = [r for r in rows if p in r]
        tp = fp = fn = 0
        for r in rs:
            got, want = set(r[p]["causes"]), set(r["truth"])
            tp, fp, fn = tp + len(got & want), fp + len(got - want), fn + len(want - got)
        s[p] = {"n": len(rs), "exact": sum(set(r[p]["causes"]) == set(r["truth"]) for r in rs),
                "exact_secondary": sum(set(r[p]["causes"]) == set(r["truth_population"]) for r in rs),
                "precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn)}
        for k in ("tool_calls", "unnecessary", "cost", "latency_s", "api_s", "cost_usd"):
            v = [r[p][k] for r in rs if k in r[p]]
            if v:
                s[p][f"mean_{k}"] = float(np.mean(v))
                s[p][f"total_{k}"] = float(np.sum(v))
        if any("usage" in r[p] for r in rs):
            s[p]["tokens"] = {k: int(sum(r[p]["usage"][k] for r in rs)) for k in rs[0][p]["usage"]}
            s[p]["failures"] = sum("note" in r[p] for r in rs)
    return s


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-haiku-5-5")
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--dry-run", type=int, metavar="SEED", help="debug on another seed without the LLM; writes nothing")
    args = ap.parse_args()
    if args.dry_run is None and not load_key():
        sys.exit("needs ANTHROPIC_API_KEY (environment or .env)")
    if args.dry_run is None:
        import anthropic
        client = anthropic.Anthropic()
    commit = subprocess.run(["git", "log", "-1", "--format=%H %cI", "--", "scripts/llm_pilot.py"], cwd=ROOT,
                            capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "scripts/llm_pilot.py"], cwd=ROOT).returncode != 0
    registry = json.loads((ROOT / "models.json").read_text())
    rng = np.random.default_rng(SEED if args.dry_run is None else args.dry_run)
    rows = []
    for name, entry in registry.items():
        pred = Predictor([str(ROOT / w) for w in entry["weights"]])
        rel = json.loads((ROOT / entry["reliability"]).read_text())
        dom, id_err, q90 = entry["domain"], rel["sets"]["pdebench_test"]["err"]["median"], rel["conformal_q90_fft"]
        eff, _, ratio = causal_factors(entry)
        cats = categories(dom)
        for ci in rng.permutation(len(cats))[:PER_MODEL]:
            cat = cats[ci]
            a, shape = make_case(rng, cat, dom)
            factors = sorted(set(cat) - {"variety"})
            truth, E, phi = oracle(pred, a, factors, dom)
            err0 = network_error(pred, a)
            u_hat = pred.predict(a)[0]
            est90 = q90 * float(fft_estimates(a[None], 1.0, u_hat[None], pred.face, pred.bc)[0]["fft_err_rel"][0])
            row = {"model": name, "category": "+".join(cat) or "none", "shape": shape, "grid": a.shape[0],
                   "contrast": float(a.max() / a.min()), "err": err0, "id_err": id_err,
                   "truth": truth, "truth_population": sorted(k for k in factors if eff[k] >= ratio),
                   "oracle_errors": E, "oracle_phi": phi,
                   "flags": {"causes": policy_flags(a, est90, dom, id_err), "cost": 0, "tool_calls": 0, "unnecessary": 0},
                   "rule": policy_rule(pred, a, dom, id_err)}
            if args.dry_run is None:
                row["llm"] = policy_llm(client, args.model, args.effort, pred, a, dom, id_err, q90)
            rows.append(row)
            print(f"{name:9s} {row['category']:22s} truth={truth} phi={ {k: round(v, 2) for k, v in phi.items()} } "
                  f"rule={row['rule']['causes']} llm={row.get('llm', {}).get('causes')}", flush=True)
    if args.dry_run is not None:
        print(json.dumps(summarise(rows, ["flags", "rule"]), indent=1))
        sys.exit()
    out = {"protocol_commit": commit, "protocol_modified_after_commit": dirty, "model": args.model, "effort": args.effort,
           "budget": BUDGET, "summary": summarise(rows, ["flags", "rule", "llm"]), "rows": rows}
    print(json.dumps(out["summary"], indent=1))
    path = ROOT / f"results/llm_pilot_{args.model}_{args.effort}.json"
    path.write_text(json.dumps(out, indent=1, default=float))
