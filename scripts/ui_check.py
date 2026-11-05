"""Drive the real app in Microsoft Edge (Playwright) and screenshot each step.

Usage: uv run --with playwright python scripts/ui_check.py http://127.0.0.1:8765/ out_dir [timeout_s]
Checks: field generation, painting a stroke re-solves, error-map toggle, batch study, evidence tab,
phone-width layout without horizontal scroll, and no console errors.
"""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

url, out = sys.argv[1], Path(sys.argv[2])
timeout = float(sys.argv[3]) * 1000 if len(sys.argv) > 3 else 120_000
out.mkdir(parents=True, exist_ok=True)
problems, console = [], []


def verdict_text(page):
    return page.locator("#verdict").inner_text()


with sync_playwright() as p:
    browser = p.chromium.launch(channel="msedge", headless=True)
    for name, vp in (("desktop", {"width": 1366, "height": 900}), ("phone", {"width": 390, "height": 844})):
        page = browser.new_page(viewport=vp, device_scale_factor=1)
        page.on("console", lambda m: console.append(f"{name} {m.type}: {m.text}") if m.type in ("error", "warning") else None)
        page.on("pageerror", lambda e: console.append(f"{name} pageerror: {e}"))
        page.goto(url)
        page.wait_for_function("document.querySelector('#verdict').innerText.includes('reference')", timeout=timeout)
        page.screenshot(path=out / f"{name}_1_generated.png", full_page=True)
        first = verdict_text(page)

        if name == "desktop":
            # paint a horizontal low-permeability stroke across the field
            page.locator("[data-brush=low]").click()
            box = page.locator("#ca").bounding_box()
            y = box["y"] + box["height"] * 0.5
            page.mouse.move(box["x"] + 10, y)
            page.mouse.down()
            for k in range(1, 21):
                page.mouse.move(box["x"] + 10 + k * (box["width"] - 20) / 20, y)
            page.mouse.up()
            page.wait_for_function(f"document.querySelector('#verdict').innerText !== {first!r}", timeout=timeout)
            page.screenshot(path=out / f"{name}_2_painted.png", full_page=True)
            if verdict_text(page) == first:
                problems.append("painting did not trigger a new solve")
            page.locator("[data-err=est]").click()
            page.screenshot(path=out / f"{name}_3_estimated_error.png", clip={"x": 0, "y": 0, "width": 1366, "height": 900})

            page.locator("nav button[data-view=batch]").click()
            page.fill("#bcount", "64")
            page.locator("#runbatch").click()
            page.wait_for_selector("#btable table", timeout=timeout)
            page.screenshot(path=out / f"{name}_4_batch.png", full_page=True)
            rows = page.locator("#btable tbody tr").count()
            if rows != 16:
                problems.append(f"batch table has {rows} verified rows, expected 16")

            page.locator("nav button[data-view=evidence]").click()
            page.wait_for_selector("#ev table", timeout=timeout)
            page.screenshot(path=out / f"{name}_5_evidence.png", full_page=True)

        overflow = page.evaluate("document.documentElement.scrollWidth - window.innerWidth")
        if overflow > 0:
            problems.append(f"{name}: horizontal overflow of {overflow}px")
        page.close()
    browser.close()

print("console:", *console, sep="\n  ") if console else print("console: clean")
print("problems:", *problems, sep="\n  ") if problems else print("problems: none")
sys.exit(1 if problems or any("error" in c for c in console) else 0)
