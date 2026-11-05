"""Measure where the time goes in the FlowGuard web app (either runtime), in Microsoft Edge via Playwright.

Usage: uv run --with playwright python scripts/perf_browser.py URL OUT.json [--phone] [--runs 3]

Cold load (fresh browser context, empty cache): navigation, network bytes and time per resource (worker
requests included), status-text phases, time to the first verdict. Then warm interactions, each timed from
the user action to the updated verdict / batch table:
  generate (new seed), paint stroke, beta change, model switch, 256 grid, batch (32 and 64 fields), evidence.
Python-side compute comes from the app's own response timings (ms.nn, ms.reference, batch ms.*).
Rendering is timed in-page by calling render(state.last) / renderBatch(last batch) directly.
--phone: 390x844 viewport. CPU throttling is not applied: DevTools throttling does not reach Web Workers,
so a throttled number would be misleading for the browser runtime.
"""
import argparse
import json
import statistics
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ap = argparse.ArgumentParser()
ap.add_argument("url")
ap.add_argument("out")
ap.add_argument("--phone", action="store_true")
ap.add_argument("--runs", type=int, default=3)
ap.add_argument("--timeout", type=float, default=300)
args = ap.parse_args()
T = args.timeout * 1000

INIT = """
window.__ev = [];
const __t0 = performance.now();
document.addEventListener('DOMContentLoaded', () => {
  const watch = (id) => { const el = document.getElementById(id); if (!el) return;
    new MutationObserver(() => window.__ev.push([performance.now(), id, el.innerText.slice(0, 90)]))
      .observe(el, {childList: true, characterData: true, subtree: true}); };
  ['verdict', 'bsummary', 'runtime', 'ev', 'status'].forEach(watch);
});
"""


def verdict(page):
    return page.locator("#verdict").inner_text()


def mark(page):
    page.evaluate("window.__prev = state.last")


def wait_new_result(page):
    page.wait_for_function("state.last !== window.__prev && !state.busy", timeout=T)


def timed(page, action):
    mark(page)
    t = time.perf_counter()
    action()
    wait_new_result(page)
    wall = 1e3 * (time.perf_counter() - t)
    ms = page.evaluate("state.last && state.last.ms")
    return {"wall_ms": wall, "server_ms": ms}


def render_ms(page, reps=5):
    return page.evaluate("""(reps) => { const out = [];
      for (let k = 0; k < reps; k++) { const t = performance.now(); render(state.last); out.push(performance.now() - t); }
      return out; }""", reps)


def stroke(page):
    box = page.locator("#ca").bounding_box()
    y = box["y"] + box["height"] * 0.4
    page.mouse.move(box["x"] + 8, y)
    page.mouse.down()
    for k in range(1, 16):
        page.mouse.move(box["x"] + 8 + k * (box["width"] - 16) / 15, y)
    page.mouse.up()


res = {"url": args.url, "viewport": "phone" if args.phone else "desktop", "runs": []}
vp = {"width": 390, "height": 844} if args.phone else {"width": 1366, "height": 900}
with sync_playwright() as p:
    browser = p.chromium.launch(channel="msedge", headless=True)
    for run in range(args.runs):
        ctx = browser.new_context(viewport=vp)  # fresh context: empty HTTP cache
        page = ctx.new_page()
        page.add_init_script(INIT)
        reqs, console = [], []
        page.on("requestfinished", lambda r: reqs.append(r))
        page.on("console", lambda m: console.append(f"{m.type}: {m.text}") if m.type in ("error", "warning") else None)
        page.on("pageerror", lambda e: console.append(f"pageerror: {e}"))
        t_nav = time.time() * 1e3
        page.goto(args.url, wait_until="load", timeout=T)
        page.wait_for_function("typeof state !== 'undefined' && state.last && !state.busy", timeout=T)
        first_wall = time.time() * 1e3 - t_nav
        origin = page.evaluate("performance.timeOrigin")
        nav = page.evaluate("(() => { const n = performance.getEntriesByType('navigation')[0];"
                            " return {dcl: n.domContentLoadedEventEnd, load: n.loadEventEnd}; })()")
        ev = page.evaluate("window.__ev")
        net = []
        for r in reqs:
            try:
                tm, sz = r.timing, r.sizes()
            except Exception:  # noqa: BLE001 - some requests have no timing (data: URLs)
                continue
            start = tm["startTime"] - origin
            net.append({"url": r.url.split("?")[0][-70:], "start_ms": start, "end_ms": start + tm["responseEnd"],
                        "bytes": sz["responseBodySize"] + sz["responseHeadersSize"]})
        rec = {"cold": {"first_verdict_ms": first_wall, "dom_content_loaded_ms": nav["dcl"], "load_ms": nav["load"],
                        "phases": [[round(t), k, s] for t, k, s in ev if k in ("verdict", "runtime")],
                        "first_solve_server_ms": page.evaluate("state.last.ms"),
                        "bytes_total": sum(x["bytes"] for x in net), "network": sorted(net, key=lambda x: x["start_ms"])}}

        warm = {}
        warm["render_single_ms"] = render_ms(page)
        warm["generate"] = []
        base = int(time.time()) % 100000 * 10  # fresh seeds: a server's cache must not serve earlier runs' fields
        for s in (base + 1, base + 2, base + 3):
            page.fill("#seed", str(s))
            warm["generate"].append(timed(page, lambda: page.click("#gen")))
        warm["paint"] = [timed(page, lambda: stroke(page)) for _ in range(2)] if not args.phone else []
        if page.evaluate("typeof state.unit !== 'undefined'"):  # current UI: beta rescales locally, no request
            warm["beta"] = {"wall_ms": page.evaluate("() => { const t = performance.now(); const b = document.getElementById('beta');"
                                                     " b.value = '2'; b.oninput(); b.value = '1'; b.oninput(); return (performance.now() - t) / 2; }"),
                            "server_ms": None}
        else:
            page.fill("#beta", "2")
            warm["beta"] = timed(page, lambda: page.dispatch_event("#beta", "change"))
            page.fill("#beta", "1")
            timed(page, lambda: page.dispatch_event("#beta", "change"))
        warm["model_switch"] = timed(page, lambda: page.select_option("#model", "pdebench"))
        page.select_option("#n", "256")
        warm["grid256"] = timed(page, lambda: page.click("#gen"))
        warm["render_single_256_ms"] = render_ms(page)
        page.select_option("#n", "128")
        timed(page, lambda: page.select_option("#model", "broad"))

        if page.locator("[data-ex=contrast]").count():  # investigation (current UI only)
            warm["contrast_example"] = timed(page, lambda: page.click("[data-ex=contrast]"))
            t = time.perf_counter()
            page.click("#invbtn")
            page.wait_for_selector("#inv .concl, #inv .err", timeout=T)
            warm["investigate"] = {"wall_ms": 1e3 * (time.perf_counter() - t), "text": page.locator("#inv").inner_text()[-300:]}
            page.click("[data-ex=typical]")
            page.wait_for_function("!state.busy", timeout=T)
        page.click("nav button[data-view=batch]")
        warm["batch"] = {}
        for count in (32, 64):
            page.fill("#bcount", str(count))
            page.evaluate("() => { window.__bt = null; window.__rb = window.__rb || renderBatch;"
                          " window.renderBatch = (j) => { const t = performance.now();"
                          " window.__rb(j); window.__bt = {ms: performance.now() - t, server: j.ms}; }; }")
            t = time.perf_counter()
            page.click("#runbatch")
            page.wait_for_function("window.__bt || document.querySelector('#bsummary .err')", timeout=T)
            info = page.evaluate("window.__bt")
            warm["batch"][count] = {"wall_ms": 1e3 * (time.perf_counter() - t), **(info or {"error": verdict(page)})}
        t = time.perf_counter()
        page.click("nav button[data-view=evidence]")
        page.wait_for_selector("#ev table", timeout=T)
        warm["evidence_ms"] = 1e3 * (time.perf_counter() - t)
        rec["warm"], rec["console"] = warm, console
        res["runs"].append(rec)
        ctx.close()
        print(f"run {run}: first verdict {first_wall:.0f} ms, {rec['cold']['bytes_total'] / 1e6:.1f} MB", flush=True)
    browser.close()


def med(xs):
    return round(statistics.median(xs)) if xs else None


R = res["runs"]
res["summary"] = {
    "first_verdict_ms": med([r["cold"]["first_verdict_ms"] for r in R]),
    "bytes_total_MB": round(statistics.median([r["cold"]["bytes_total"] for r in R]) / 1e6, 1),
    "generate_wall_ms": med([g["wall_ms"] for r in R for g in r["warm"]["generate"]]),
    "generate_server_nn_ms": med([g["server_ms"]["nn"] for r in R for g in r["warm"]["generate"]]),
    "generate_server_ref_ms": med([g["server_ms"]["reference"] for r in R for g in r["warm"]["generate"]]),
    "paint_wall_ms": med([g["wall_ms"] for r in R for g in r["warm"]["paint"]]),
    "beta_wall_ms": med([r["warm"]["beta"]["wall_ms"] for r in R]),
    "model_switch_wall_ms": med([r["warm"]["model_switch"]["wall_ms"] for r in R]),
    "grid256_wall_ms": med([r["warm"]["grid256"]["wall_ms"] for r in R]),
    "render_128_ms": med([x for r in R for x in r["warm"]["render_single_ms"]]),
    "render_256_ms": med([x for r in R for x in r["warm"]["render_single_256_ms"]]),
    "batch32_wall_ms": med([r["warm"]["batch"][32]["wall_ms"] for r in R]),
    "batch64_wall_ms": med([r["warm"]["batch"][64]["wall_ms"] for r in R]),
    "evidence_ms": med([r["warm"]["evidence_ms"] for r in R]),
    "investigate_wall_ms": med([r["warm"]["investigate"]["wall_ms"] for r in R if "investigate" in r["warm"]]),
}
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
Path(args.out).write_text(json.dumps(res, indent=1))
print(json.dumps(res["summary"], indent=1))
