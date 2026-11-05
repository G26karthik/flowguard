"""Independent steady-state solver for 2D Darcy flow on the unit square.

    -div(a(x) grad u(x)) = f(x)  in (0,1)^2,   u = 0 on the boundary.

Cell-centred finite volumes on an N x N grid (centres at (i+0.5)/N, the PDEBench grid).
Arrays are indexed [i, j] with i along x and j along y, as in PDEBench's `nu`.

face: how the coefficient on a cell face is formed from its two cells.
  "arith" - arithmetic mean (what PDEBench's generator uses)
  "harm"  - harmonic mean (flux-continuous; the physically consistent choice for jumps)
bc: where the Dirichlet condition is imposed.
  "face"  - u = 0 on the boundary face (ghost = -u_edge, second order)
  "ghost" - u = 0 at the ghost-cell centre h/2 outside the domain (ghost = 0)
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu


def _face_coeffs(a, face):
    """Coefficients on x-faces (..., N+1, N) and y-faces (..., N, N+1); boundary faces copy the edge cell."""
    p = np.pad(a, [(0, 0)] * (a.ndim - 2) + [(1, 1), (1, 1)], mode="edge")
    if face == "arith":
        mean = lambda l, r: 0.5 * (l + r)
    elif face == "harm":
        mean = lambda l, r: 2 * l * r / (l + r)
    else:
        raise ValueError(face)
    ax = mean(p[..., :-1, 1:-1], p[..., 1:, 1:-1])
    ay = mean(p[..., 1:-1, :-1], p[..., 1:-1, 1:])
    return ax, ay


def _ghost_weight(bc):
    if bc not in ("face", "ghost"):
        raise ValueError(bc)
    return 2.0 if bc == "face" else 1.0


def apply_operator(a, u, face="arith", bc="face"):
    """Matrix-free A u = -div(a grad u) for batches of fields (..., N, N)."""
    a = np.asarray(a, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    n = u.shape[-1]
    ax, ay = _face_coeffs(a, face)
    g = -1.0 if bc == "face" else 0.0
    _ghost_weight(bc)
    pad = [(0, 0)] * (u.ndim - 2) + [(1, 1), (1, 1)]
    up = np.pad(u, pad)
    up[..., 0, :], up[..., -1, :] = g * up[..., 1, :], g * up[..., -2, :]
    up[..., :, 0], up[..., :, -1] = g * up[..., :, 1], g * up[..., :, -2]
    fx = ax * (up[..., 1:, 1:-1] - up[..., :-1, 1:-1])  # flux * h on x-faces
    fy = ay * (up[..., 1:-1, 1:] - up[..., 1:-1, :-1])
    return -(fx[..., 1:, :] - fx[..., :-1, :] + fy[..., :, 1:] - fy[..., :, :-1]) * n * n


def assemble(a, face="arith", bc="face"):
    """Sparse SPD matrix of the operator for one coefficient field (N, N); unknowns flattened row-major."""
    a = np.asarray(a, dtype=np.float64)
    n = a.shape[0]
    ax, ay = _face_coeffs(a, face)
    w = _ghost_weight(bc)
    axw, ayw = ax.copy(), ay.copy()
    axw[0], axw[-1] = w * ax[0], w * ax[-1]
    ayw[:, 0], ayw[:, -1] = w * ay[:, 0], w * ay[:, -1]
    diag = (axw[1:] + axw[:-1] + ayw[:, 1:] + ayw[:, :-1]).ravel()
    off_x = -ax[1:-1].ravel()                 # couples (i, j) with (i+1, j): offset n
    off_y = -ay[:, 1:-1]                      # couples (i, j) with (i, j+1): offset 1
    off_y = np.concatenate([off_y, np.zeros((n, 1))], axis=1).ravel()[:-1]  # zero across row ends
    A = sp.diags([diag, off_x, off_x, off_y, off_y], [0, n, -n, 1, -1], format="csc")
    return A * (n * n)


def solve(a, f, face="arith", bc="face"):
    """Steady solution for one field. f may be a scalar (PDEBench's constant beta) or an (N, N) array."""
    a = np.asarray(a, dtype=np.float64)
    n = a.shape[0]
    rhs = np.broadcast_to(np.asarray(f, dtype=np.float64), (n, n)).ravel()
    return splu(assemble(a, face, bc)).solve(rhs).reshape(n, n)


def residual(a, u, f, face="arith", bc="face"):
    """r = f - A u. Zero (to round-off) for the solver's own solution."""
    return np.asarray(f, dtype=np.float64) - apply_operator(a, u, face, bc)
