"""NumPy forward pass of flowguard.model.FNO2d, so serving needs no torch (and runs under Pyodide).

Weights come from scripts/export_numpy.py: <run>/weights.npz plus metadata in the same file.
"""
import json

import numpy as np
import scipy.fft as sfft

_A = [np.float32(c) for c in (0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429)]


def _erf(x):
    """Abramowitz & Stegun 7.1.26, |error| < 1.5e-7; ~3x faster than scipy.special.erf on float32 arrays."""
    s = np.sign(x)
    x = np.abs(x)
    t = 1 / (1 + np.float32(0.3275911) * x)
    poly = t * (_A[0] + t * (_A[1] + t * (_A[2] + t * (_A[3] + t * _A[4]))))
    return s * (1 - poly * np.exp(-x * x))


def _gelu(x):
    return 0.5 * x * (1 + _erf(x * np.float32(1 / np.sqrt(2))))  # torch's default (exact) GELU


def _conv1x1(w, b, x):
    bs, c, n, m = x.shape
    return (w[:, :, 0, 0] @ x.reshape(bs, c, n * m)).reshape(bs, -1, n, m) + b[None, :, None, None]


def _modes_mul(xf, w):
    """(B, I, a, c) x (I, O, a, c) -> (B, O, a, c), one small matmul per retained Fourier mode."""
    bs, i, a, c = xf.shape
    lhs = xf.transpose(2, 3, 0, 1).reshape(a * c, bs, i)
    rhs = w.transpose(2, 3, 0, 1).reshape(a * c, i, -1)
    return (lhs @ rhs).reshape(a, c, bs, -1).transpose(2, 3, 0, 1)


class NumpyFNO:
    def __init__(self, path):
        z = np.load(path)
        self.meta = json.loads(str(z["meta"]))
        self.p = {k: z[k] for k in z.files if k != "meta"}
        self.layers = self.meta["cfg"]["layers"]
        self.modes = self.meta["cfg"]["modes"]

    def __call__(self, loga):
        """loga: (B, N, N) log10 coefficient. Returns (B, N, N) solution for unit source, float64.
        Computes in float32 like the torch model."""
        p, m = self.p, self.modes
        b, n, w = loga.shape
        gx = np.broadcast_to(((np.arange(n) + 0.5) / n)[None, :, None], (b, n, w))
        gy = np.broadcast_to(((np.arange(w) + 0.5) / w)[None, None, :], (b, n, w))
        x = _conv1x1(p["lift.weight"], p["lift.bias"], np.stack([loga, gx, gy], 1).astype(np.float32))
        for k in range(self.layers):
            xf = sfft.rfft2(x, workers=-1)
            out = np.zeros(xf.shape, dtype=xf.dtype)
            W = p[f"spectral.{k}.w"]
            out[:, :, :m, :m] = _modes_mul(xf[:, :, :m, :m], W[0])
            out[:, :, -m:, :m] = _modes_mul(xf[:, :, -m:, :m], W[1])
            x = sfft.irfft2(out, s=(n, w), workers=-1).astype(np.float32) + _conv1x1(p[f"local.{k}.weight"], p[f"local.{k}.bias"], x)
            if k < self.layers - 1:
                x = _gelu(x)
        x = _gelu(_conv1x1(p["proj.0.weight"], p["proj.0.bias"], x))
        return _conv1x1(p["proj.2.weight"], p["proj.2.bias"], x)[:, 0].astype(np.float64) * self.meta["y_scale"]
