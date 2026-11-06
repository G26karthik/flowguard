"""Open the live site while the server is scaled to zero; time the first answer, the move to the server, and a solve
after it. Run only after the app has had no traffic for ~6 minutes (az containerapp replica list ... is empty).

Usage: uv run --with playwright python scripts/cold_start.py URL OUT.json [--phone]
"""
import json
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

url, out = sys.argv[1], Path(sys.argv[2])
phone = "--phone" in sys.argv
r: dict = {"phone": phone}
with sync_playwright() as p:
    browser = p.chromium.launch(channel="msedge", headless=True)
    ctx = browser.new_context(viewport={"width": 390, "height": 844} if phone else {"width": 1366, "height": 900},
                              has_touch=phone)
    page = ctx.new_page()
    api = []
    page.on("request", lambda q: api.append((time.perf_counter(), q.url)) if "/api/" in q.url else None)
    t0 = time.perf_counter()
    page.goto(url)
    page.wait_for_function("typeof state !== 'undefined' && state.last && !state.busy", timeout=120_000)
    r["first_answer_s"] = time.perf_counter() - t0
    r["first_answer_runtime"] = page.evaluate("backendReady.then(b => b.kind)")
    page.wait_for_function("document.querySelector('#runtime').innerText.includes('server')", timeout=120_000)
    r["server_in_use_s"] = time.perf_counter() - t0
    page.evaluate("window.__prev = state.last")
    n = len(api)
    t = time.perf_counter()
    page.fill("#seed", "11")
    page.click("#gen")
    page.wait_for_function("state.last !== window.__prev && !state.busy", timeout=60_000)
    r["solve_after_switch_ms"] = 1000 * (time.perf_counter() - t)
    r["solve_after_switch_went_to_server"] = any("/api/solve" in u for _, u in api[n:])
    browser.close()
print(json.dumps(r, indent=1))
out.write_text(json.dumps(r, indent=1))
