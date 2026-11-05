import base64
import importlib

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from flowguard.model import FNO2d
from flowguard.solver import solve


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    # untrained checkpoints: these tests check plumbing and validation, not accuracy
    d = tmp_path_factory.mktemp("models")
    for s in (0, 1):
        torch.manual_seed(s)
        m = FNO2d(modes=12, width=8)
        torch.save({"state_dict": m.state_dict(), "cfg": m.cfg, "y_scale": 0.05}, d / f"m{s}.pt")
    mp = pytest.MonkeyPatch()
    mp.setenv("FLOWGUARD_MODELS", str(d / "m*.pt"))
    import flowguard.api as api
    api = importlib.reload(api)
    yield TestClient(api.app)
    mp.undo()


def dec(s, n):
    return np.frombuffer(base64.b64decode(s), dtype="<f4").reshape(n, n)


def test_solve_returns_exact_solution_and_consistent_error(client):
    r = client.post("/api/solve", json={"field": {"family": "pdebench", "seed": 3}, "beta": 2.0})
    assert r.status_code == 200, r.text
    j = r.json()
    a, u, up = dec(j["a"], 128), dec(j["u_exact"], 128), dec(j["u_pred"], 128)
    np.testing.assert_allclose(u, solve(a.astype(np.float64), 2.0, "arith", "ghost"), rtol=1e-5, atol=1e-7)
    assert abs(j["true_rel_err"] - np.linalg.norm(up - u) / np.linalg.norm(u)) < 1e-4
    assert j["ensemble_spread"] is not None


def test_user_field_and_flags(client):
    n = 64
    a = np.full((n, n), 0.01, dtype="<f4")
    a[20:40] = 1.0
    r = client.post("/api/solve", json={"field": {"n": n, "values": base64.b64encode(a.tobytes()).decode()}})
    assert r.status_code == 200, r.text
    codes = {f["code"] for f in r.json()["flags"]}
    assert {"resolution", "contrast"} <= codes


@pytest.mark.parametrize("payload", [
    {"field": {"n": 64, "values": base64.b64encode(np.ones(10, "<f4").tobytes()).decode()}},
    {"field": {"n": 64, "values": base64.b64encode((-np.ones(64 * 64, "<f4")).tobytes()).decode()}},
    {"field": {"family": "nope"}},
    {"field": {"family": "pdebench", "n": 1000}},
    {"field": {"family": "pdebench"}, "beta": 0},
])
def test_rejects_bad_input(client, payload):
    assert client.post("/api/solve", json=payload).status_code == 400


def test_batch_verifies_budget(client):
    r = client.post("/api/batch", json={"field": {"family": "grf", "seed": 1}, "count": 20, "verify_risky": 3, "verify_random": 4})
    assert r.status_code == 200, r.text
    rows = r.json()["rows"]
    assert sum(x["role"] == "risky" for x in rows) == 3 and sum(x["role"] == "audit" for x in rows) == 4
    assert sum(x["true_rel_err"] is not None for x in rows) == 7
    risky_est = min(x["fft_err_rel"] for x in rows if x["role"] == "risky")
    assert all(x["fft_err_rel"] <= risky_est for x in rows if x["role"] != "risky")
    assert client.post("/api/batch", json={"field": {"family": "grf"}, "count": 5, "verify_risky": 5, "verify_random": 1}).status_code == 400
