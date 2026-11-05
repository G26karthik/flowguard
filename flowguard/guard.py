"""Cheap reliability indicators for a predicted Darcy solution, without a fine-grid solve.

Residual r = f - A u_hat satisfies A e = r for the true error e = u - u_hat.
Two-level estimate: solve (P^T A P) e_c = P^T r with P a smoothed-aggregation prolongation onto a
k-times coarser grid. P e_c is the A-orthogonal projection of e onto range(P), so
||P e_c||_A <= ||e||_A always (a guaranteed lower bound on the energy-norm error).
Plain piecewise-constant aggregates recover only ~29% of a smooth error's L2 norm (their jumps cost
energy); s=4 damped-Jacobi smoothing steps of P recover ~91% (measured, k=4, N=128).
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from flowguard.solver import assemble, residual


def aggregation(n, k):
    """Piecewise-constant prolongation (n*n, (n//k)**2), row-major like the solver."""
    if n % k:
        raise ValueError(f"grid {n} not divisible by coarsening {k}")
    i = np.arange(n)
    coarse = (i[:, None] // k) * (n // k) + (i[None, :] // k)
    return sp.csr_matrix((np.ones(n * n), (np.arange(n * n), coarse.ravel())), shape=(n * n, (n // k) ** 2))


def prolongation(A, n, k=4, s=4):
    """Smoothed aggregation: P = (I - 2/3 D^-1 A)^s P0."""
    P = aggregation(n, k)
    Dinv = sp.diags(1 / A.diagonal())
    for _ in range(s):
        P = P - (2 / 3) * (Dinv @ (A @ P))
    return sp.csc_matrix(P)


def indicators(a, f, u_hat, face="arith", bc="ghost", k=4, s=4):
    """Return cheap indicators and the coarse error field. All norms relative to ||u_hat||."""
    a = np.asarray(a, dtype=np.float64)
    u_hat = np.asarray(u_hat, dtype=np.float64)
    n = a.shape[0]
    A = assemble(a, face, bc)
    r = residual(a, u_hat, f, face, bc)
    P = prolongation(A, n, k, s)
    e_c = splu((P.T @ A @ P).tocsc()).solve(P.T @ r.ravel())
    e_est = (P @ e_c).reshape(n, n)
    un = np.linalg.norm(u_hat)
    energy = lambda v: float(np.sqrt(v.ravel() @ (A @ v.ravel())))
    fn = np.linalg.norm(np.broadcast_to(f, (n, n)))
    return {
        "residual_rel": float(np.linalg.norm(r) / fn),
        "coarse_err_rel": float(np.linalg.norm(e_est) / un),
        "coarse_energy_err": energy(e_est),
        "energy_norm_u_hat": energy(u_hat),
    }, e_est, r
