"""Month-to-date Azure Container Apps usage of the FlowGuard app against the monthly free grant, plus actual cost.

Usage: python scripts/azure_usage.py [resource_group=flowguard] [app=flowguard]
Needs `az login`. Reads platform metrics and Cost Management only (both free). Usage is replica time x allocation
(1 vCPU, 2 GiB): billed per second while a replica runs, whether or not it is busy, so it is an upper bound on what
counts against the grant. The free grant is per subscription; nothing else in this subscription uses Container Apps.
"""
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

FREE = {"vcpu_s": 180_000, "gib_s": 360_000, "requests": 2_000_000}  # learn.microsoft.com/azure/container-apps/billing


def az(*args, fatal=True):
    out = subprocess.run(["az", *args, "-o", "json"], capture_output=True, text=True, shell=sys.platform == "win32")
    if out.returncode:
        if not fatal:
            return out.stderr.strip()
        sys.exit(out.stderr.strip())
    return json.loads(out.stdout or "null")


rg = sys.argv[1] if len(sys.argv) > 1 else "flowguard"
name = sys.argv[2] if len(sys.argv) > 2 else "flowguard"
app = az("containerapp", "show", "-g", rg, "-n", name)
res = app["properties"]["template"]["containers"][0]["resources"]
cpu, mem = float(res["cpu"]), float(res["memory"].rstrip("Gi"))
now = datetime.now(timezone.utc)
start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def series(metric, agg, interval):
    m = az("monitor", "metrics", "list", "--resource", app["id"], "--metric", metric, "--aggregation", agg,
           "--interval", interval,
           "--start-time", f"{start:%Y-%m-%dT%H:%M:%SZ}", "--end-time", f"{now:%Y-%m-%dT%H:%M:%SZ}")
    return [p.get(agg.lower()) or 0 for ts in m["value"][0]["timeseries"] for p in ts["data"]]


replica_s = sum(series("Replicas", "Average", "PT5M")) * 300
requests = sum(series("Requests", "Total", "PT1H"))
use = {"vcpu_s": replica_s * cpu, "gib_s": replica_s * mem, "requests": requests}
print(f"{name}: {cpu} vCPU, {mem} GiB, replicas {app['properties']['template']['scale']['minReplicas']}"
      f"-{app['properties']['template']['scale']['maxReplicas']}; since {start:%Y-%m-%d}")
print(f"  replica time {replica_s / 3600:.2f} h")
for k, v in use.items():
    print(f"  {k:<9} {v:>12,.0f} of {FREE[k]:>9,} free ({100 * v / FREE[k]:.1f}%)")

sub = az("account", "show")["id"]
q = {"type": "ActualCost", "timeframe": "MonthToDate",
     "dataset": {"granularity": "None", "aggregation": {"c": {"name": "Cost", "function": "Sum"}}}}
with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:  # a file: cmd.exe mangles inline JSON
    json.dump(q, f)
cost = az("rest", "--method", "post", "--url", f"https://management.azure.com/subscriptions/{sub}/providers/"
          "Microsoft.CostManagement/query?api-version=2023-11-01", "--body", "@" + f.name, fatal=False)
if isinstance(cost, str):  # Cost Management throttles (429) often; usage above does not depend on it
    sys.exit(f"  subscription cost: query failed ({cost[:80]}); rerun later")
rows = cost["properties"]["rows"]
print("  subscription cost month to date:", f"{rows[0][0]:.4f} {rows[0][1]}" if rows else "none recorded",
      "(Cost Management lags by up to a day)")
