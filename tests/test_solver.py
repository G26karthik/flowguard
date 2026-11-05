import numpy as np
import pytest

from flowguard.solver import apply_operator, assemble, residual, solve


def _grid(n):
    c = (np.arange(n) + 0.5) / n
    return np.meshgrid(c, c, indexing="ij")


@pytest.mark.parametrize("face", ["arith", "harm"])
@pytest.mark.parametrize("bc", ["face", "ghost"])
def test_matrix_free_matches_sparse_and_is_symmetric(face, bc):
    rng = np.random.default_rng(0)
    n = 17
    a = np.where(rng.random((n, n)) > 0.5, 1.0, 0.1)
    u = rng.standard_normal((n, n))
    A = assemble(a, face, bc)
    np.testing.assert_allclose(A @ u.ravel(), apply_operator(a, u, face, bc).ravel(), rtol=1e-12, atol=1e-9)
    assert abs(A - A.T).max() < 1e-9
    # positive definite: smallest eigenvalue of a small SPD matrix > 0
    assert np.linalg.eigvalsh(A.toarray()).min() > 0


def test_batched_operator_matches_per_sample():
    rng = np.random.default_rng(1)
    a = rng.uniform(0.1, 1, (3, 8, 8))
    u = rng.standard_normal((3, 8, 8))
    batch = apply_operator(a, u)
    for k in range(3):
        np.testing.assert_allclose(batch[k], apply_operator(a[k], u[k]))


def _manufactured(n):
    x, y = _grid(n)
    pi = np.pi
    u = np.sin(pi * x) * np.sin(pi * y)
    a = 1 + 0.5 * np.sin(2 * pi * x) * np.cos(2 * pi * y)
    ax = pi * np.cos(2 * pi * x) * np.cos(2 * pi * y)
    ay = -pi * np.sin(2 * pi * x) * np.sin(2 * pi * y)
    ux = pi * np.cos(pi * x) * np.sin(pi * y)
    uy = pi * np.sin(pi * x) * np.cos(pi * y)
    f = -(ax * ux + ay * uy) + 2 * pi**2 * a * u
    return a, f, u


@pytest.mark.parametrize("face", ["arith", "harm"])
def test_second_order_convergence_manufactured_solution(face):
    errs = []
    for n in (32, 64, 128):
        a, f, u = _manufactured(n)
        errs.append(np.sqrt(np.mean((solve(a, f, face, "face") - u) ** 2)))
    rates = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
    assert errs[-1] < 1e-3
    assert np.all(rates > 1.8), rates


def test_solution_has_zero_residual_and_scales_linearly_with_source():
    rng = np.random.default_rng(2)
    a = np.where(rng.random((32, 32)) > 0.5, 1.0, 0.1)
    u1 = solve(a, 1.0)
    assert np.abs(residual(a, u1, 1.0)).max() < 1e-8
    np.testing.assert_allclose(solve(a, 7.5), 7.5 * u1, rtol=1e-10)


def test_propagate_matches_matrix_exponential():
    from scipy.linalg import expm

    from flowguard.solver import propagate

    rng = np.random.default_rng(4)
    a = np.where(rng.random((12, 12)) > 0.5, 1.0, 0.1)
    A = assemble(a, "arith", "ghost")
    v = np.ones(144) + 0.3 * rng.standard_normal(144)
    got = propagate(A, v, [0.05, 0.3])
    for t, g in zip([0.05, 0.3], got):
        ref = expm(-t * A.toarray()) @ v
        assert np.linalg.norm(g - ref) / np.linalg.norm(ref) < 2e-3, t


def test_constant_coefficient_means_agree():
    a = np.full((16, 16), 0.3)
    np.testing.assert_allclose(solve(a, 1.0, "arith"), solve(a, 1.0, "harm"))
