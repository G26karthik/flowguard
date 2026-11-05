"""Drive the real app in Microsoft Edge (Playwright) and screenshot each step.

Usage: uv run --with playwright python scripts/ui_check.py URL out_dir [timeout_s]
Checks (desktop 1366x900 and phone 390x844): first solve, the examples, painting re-solves, beta rescales
without a request, model switch, the error-map toggle, the investigation, batch screening, the evidence tab,
a network failure (requests aborted) shown as an error and recovered from, no horizontal scroll, and no
console errors.
"""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

url, out = sys.argv[1], Path(sys.argv[2])
timeout = float(sys.argv[3]) * 1000 if len(sys.argv) > 3 else 180_000
out.mkdir(parents=True, exist_ok=True)
problems, console = [], []


def settled(page):
    page.wait_for_function("typeof state !== 'undefined' && state.last && !state.busy", timeout=timeout)


def new_result(page, action):
    page.evaluate("window.__prev = state.last")
    action()
    page.wait_for_function("state.last !== window.__prev && !state.busy", timeout=timeout)


def check(cond, msg):
    if not cond:
        problems.append(msg)


with sync_playwright() as p:
    browser = p.chromium.launch(channel="msedge", headless=True)
    for name, vp in (("desktop", {"width": 1366, "height": 900}), ("phone", {"width": 390, "height": 844})):
        ctx = browser.new_context(viewport=vp, device_scale_factor=1, has_touch=name == "phone")
        page = ctx.new_page()
        page.on("console", lambda m, name=name: console.append(f"{name} {m.type}: {m.text}") if m.type in ("error", "warning") else None)
        page.on("pageerror", lambda e, name=name: console.append(f"{name} pageerror: {e}"))
        requests = []
        page.on("request", lambda r: requests.append(r.url) if "/api/" in r.url else None)
        page.goto(url)
        settled(page)
        page.screenshot(path=out / f"{name}_1_first.png", full_page=True)
        check("agrees" in page.locator("#verdict").inner_text(), f"{name}: typical example does not agree: {page.locator('#verdict').inner_text()[:120]}")

        # beta: local rescale, no request
        n_req = len(requests)
        page.fill("#beta", "3")
        page.dispatch_event("#beta", "input")
        check(len(requests) == n_req, f"{name}: beta change sent a request")
        page.fill("#beta", "1")
        page.dispatch_event("#beta", "input")

        # extreme-contrast example, then the investigation
        new_result(page, lambda: page.click("[data-ex=contrast]"))
        check("Do not use" in page.locator("#verdict").inner_text(), f"{name}: extreme contrast not flagged")
        page.click("#invbtn")
        page.wait_for_selector("#inv .concl, #inv .err", timeout=timeout)
        inv = page.locator("#inv").inner_text()
        check("contrast" in inv.split("Measured cause")[-1] if "Measured cause" in inv else False, f"{name}: investigation did not name contrast: {inv[-200:]}")
        page.screenshot(path=out / f"{name}_2_investigation.png", full_page=True)

        if name == "desktop":
            new_result(page, lambda: page.click("[data-ex=typical]"))
            page.locator("[data-brush=low]").click()
            box = page.locator("#ca").bounding_box()
            y = box["y"] + box["height"] * 0.5

            def stroke():
                page.mouse.move(box["x"] + 10, y)
                page.mouse.down()
                for k in range(1, 21):
                    page.mouse.move(box["x"] + 10 + k * (box["width"] - 20) / 20, y)
                page.mouse.up()
            new_result(page, stroke)
            page.screenshot(path=out / f"{name}_3_painted.png", full_page=True)
            new_result(page, lambda: page.select_option("#model", "pdebench"))
            page.select_option("#model", "broad")
            settled(page)
            page.locator("[data-err=est]").click()
            page.screenshot(path=out / f"{name}_4_estimated_error.png", clip={"x": 0, "y": 0, "width": 1366, "height": 900})

            # failure: the next solve's request is aborted; the app must say so and keep working afterwards
            n_console = len(console)
            page.route("**/api/solve", lambda route: route.abort())
            page.click("#gen")
            page.wait_for_function("!state.busy && /Could not solve/.test(document.querySelector('#verdict').innerText)", timeout=timeout)
            page.screenshot(path=out / f"{name}_5_failure.png", full_page=True)
            page.unroute("**/api/solve")
            console[n_console:] = [c for c in console[n_console:] if "ERR_FAILED" not in c]  # the aborted request itself
            new_result(page, lambda: page.click("#gen"))
            check("Could not" not in page.locator("#verdict").inner_text(), "did not recover after a failed request")

            page.locator("nav button[data-view=batch]").click()
            page.fill("#bcount", "64")
            page.locator("#runbatch").click()
            page.wait_for_function("document.querySelector('#btable table') || document.querySelector('#bsummary .err')", timeout=timeout)
            page.screenshot(path=out / f"{name}_6_batch.png", full_page=True)
            if page.locator("#bsummary .err").count():
                problems.append("batch error: " + page.locator("#bsummary").inner_text())
            rows = page.locator("#btable tbody tr").count()
            check(rows == 16, f"batch table has {rows} checked rows, expected 16")

            page.locator("nav button[data-view=evidence]").click()
            page.wait_for_selector("#ev table", timeout=timeout)
            page.screenshot(path=out / f"{name}_7_evidence.png", full_page=True)

        overflow = page.evaluate("document.documentElement.scrollWidth - window.innerWidth")
        check(overflow <= 0, f"{name}: horizontal overflow of {overflow}px")
        ctx.close()
    browser.close()

print("console:", *console, sep="\n  ") if console else print("console: clean")
print("problems:", *problems, sep="\n  ") if problems else print("problems: none")
sys.exit(1 if problems or any("error" in c for c in console) else 0)
