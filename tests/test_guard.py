import numpy as np

from flowguard.fields import pdebench
from flowguard.guard import indicators
from flowguard.solver import assemble, solve


def _case(seed=0, n=64):
    a = pdebench(1, np.random.default_rng(seed), n=n)[0]
    return a, solve(a, 1.0, "arith", "ghost")


def test_exact_solution_gives_zero_indicators():
    a, u = _case()
    ind, e, _ = indicators(a, 1.0, u)
    assert ind["residual_rel"] < 1e-10 and ind["coarse_err_rel"] < 1e-8


def test_coarse_estimate_is_energy_lower_bound_and_tracks_smooth_error():
    a, u = _case(1)
    n = a.shape[0]
    c = (np.arange(n) + 0.5) / n
    A = assemble(a, "arith", "ghost")
    rng = np.random.default_rng(3)
    for amp in (0.01, 0.05):
        # smooth error plus a little high-frequency noise, the kind of error a network makes
        delta = amp * u.max() * (np.outer(np.sin(np.pi * c), np.sin(2 * np.pi * c)) + 0.1 * rng.standard_normal((n, n)))
        ind, e_est, _ = indicators(a, 1.0, u + delta)
        true_e = -delta
        true_energy = np.sqrt(true_e.ravel() @ (A @ true_e.ravel()))
        assert ind["coarse_energy_err"] <= true_energy * (1 + 1e-9)
        # most of the (mostly smooth) error is recovered in L2 and in shape
        ratio = np.linalg.norm(e_est) / np.linalg.norm(true_e)
        assert 0.75 < ratio <= 1.0 + 1e-9, ratio
        assert np.corrcoef(e_est.ravel(), true_e.ravel())[0, 1] > 0.95
