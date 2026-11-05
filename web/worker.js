// Runs the flowguard Python package in the browser (Pyodide), off the main thread.
// The static site ships the same core.py / solver.py / npfno.py the FastAPI server uses (see manifest.json).
importScripts("https://cdn.jsdelivr.net/pyodide/v0.26.4/full/pyodide.js");

let ready = null;

async function boot() {
  const manifest = await (await fetch("manifest.json", { cache: "no-store" })).json();
  postMessage({ progress: "Loading Python runtime" });
  const py = await loadPyodide({ indexURL: "https://cdn.jsdelivr.net/pyodide/v0.26.4/full/" });
  postMessage({ progress: "Loading NumPy and SciPy" });
  await py.loadPackage(["numpy", "scipy"]);
  postMessage({ progress: "Loading model weights" });
  py.FS.mkdir("/app");
  for (const f of manifest.files) {
    const r = await fetch(f);
    if (!r.ok) throw new Error(`could not fetch ${f} (HTTP ${r.status})`);
    const buf = new Uint8Array(await r.arrayBuffer());
    const dir = f.split("/").slice(0, -1);
    for (let i = 1; i <= dir.length; i++) { try { py.FS.mkdir("/app/" + dir.slice(0, i).join("/")); } catch (e) { /* exists */ } }
    py.FS.writeFile("/app/" + f, buf);
  }
  py.runPython(`
import sys, json
sys.path.insert(0, "/app")
from pathlib import Path
from flowguard.core import Registry
_m = json.loads(${JSON.stringify(JSON.stringify(manifest))})
_load = lambda p: json.loads(Path("/app/" + p).read_text()) if p else None
svc = Registry(_m["registry"], _load, _load(_m.get("discretization")), path=lambda w: "/app/" + w)
def call(method, payload):
    try:
        return json.dumps(getattr(svc, method)(json.loads(payload)))
    except ValueError as exc:
        return json.dumps({"error": str(exc)})
`);
  return py;
}

onmessage = async (ev) => {
  const { id, method, payload } = ev.data;
  try {
    ready = ready || boot();
    const py = await ready;
    const out = py.globals.get("call")(method, JSON.stringify(payload || {}));
    postMessage({ id, result: JSON.parse(out) });
  } catch (e) {
    postMessage({ id, error: String(e && e.message || e) });
  }
};
