"""NumPy forward pass of flowguard.model.FNO2d, so serving needs no torch (and runs under Pyodide).

Weights come from scripts/export_numpy.py: <run>/weights.npz plus metadata in the same file.
"""
import json

import numpy as np
from scipy.linalg.blas import sgemm

_A = [np.float32(c) for c in (0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429)]
_R, _P, _HALF = np.float32(1 / np.sqrt(2)), np.float32(0.3275911), np.float32(0.5)


def _gelu(x, blk=1 << 15):
    """torch's default (exact, erf) GELU; erf by Abramowitz & Stegun 7.1.26 (|error| < 1.5e-7).
    In place on cache-sized blocks: 2.6x faster than whole-array temporaries, and the GELU was two thirds of
    the forward pass. With c = 1 - erf(|z|), z = x/sqrt2: gelu = x - x c/2 for x >= 0, x c/2 for x < 0."""
    xf = np.ascontiguousarray(x, dtype=np.float32).reshape(-1)
    y = np.empty_like(xf)
    t, e = np.empty(blk, np.float32), np.empty(blk, np.float32)
    for s in range(0, xf.size, blk):
        v, o = xf[s:s + blk], y[s:s + blk]
        tt, ee = t[:v.size], e[:v.size]
        np.multiply(v, _R, out=o)
        np.abs(o, out=ee)
        np.multiply(ee, _P, out=tt)
        tt += 1
        np.reciprocal(tt, out=tt)
        np.multiply(ee, ee, out=ee)
        np.negative(ee, out=ee)
        np.exp(ee, out=ee)
        np.multiply(tt, _A[4], out=o)
        for c in (_A[3], _A[2], _A[1], _A[0]):
            o += c
            o *= tt
        o *= ee          # c = 1 - erf(|z|)
        o *= v
        o *= _HALF       # x c / 2
        np.subtract(v, o, out=ee)
        np.copyto(o, ee, where=v >= 0)
    return y.reshape(x.shape)


def _mm(a, b):
    """a @ b for C-contiguous float32 matrices through BLAS sgemm. Not numpy matmul: Pyodide's numpy matmul has
    no BLAS (67x slower there). (A B)^T = B^T A^T, and transposed C-ordered arrays are Fortran-ordered views,
    so sgemm copies nothing and the result's transpose is C-ordered again."""
    return sgemm(1.0, b.T, a.T).T


def _dft_bases(n, w, m):
    """Truncated DFT matrices for the retained modes (kx in [0, m) and [n-m, n), ky in [0, m)), float32.
    Equivalent to torch.fft.rfft2 / irfft2 (norm "backward") restricted to those modes: the network zeroes all
    other modes, so transforming only these is exact, and as matrix products it is BLAS work instead of FFTs."""
    ky, kx = np.arange(m), np.r_[0:m, n - m:n]
    aw = 2 * np.pi * np.outer(np.arange(w), ky) / w                 # (w, m)
    fwd_w = np.concatenate([np.cos(aw), -np.sin(aw)], 1)             # x @ fwd_w = [Re, Im] of the w-transform
    ah = 2 * np.pi * np.outer(kx, np.arange(n)) / n                  # (2m, n)
    er, ei = np.cos(ah), -np.sin(ah)
    fwd_h = np.block([[er, -ei], [ei, er]])                          # (4m, 2n): complex product in real form
    ir, ii = np.cos(ah.T) / n, np.sin(ah.T) / n                      # (n, 2m) inverse along h, 1/n
    inv_h = np.block([[ir, -ii], [ii, ir]])                          # (2n, 4m)
    c = np.where(ky == 0, 1.0, 2.0)[:, None] / w                     # irfft: Hermitian doubling, imag of DC dropped
    inv_w = np.concatenate([c * np.cos(aw.T), -c * np.sin(aw.T)], 0)  # (2m, w)
    return [np.ascontiguousarray(x, dtype=np.float32) for x in (fwd_w, fwd_h, inv_h, inv_w)]


class NumpyFNO:
    """Channel-first (C, B, N, N) internally: every 1x1 convolution is one sgemm without transposes."""

    def __init__(self, path):
        z = np.load(path)
        self.meta = json.loads(str(z["meta"]))
        p = {k: z[k] for k in z.files if k != "meta"}
        self.layers = self.meta["cfg"]["layers"]
        self.modes = m = self.meta["cfg"]["modes"]
        conv = lambda k: (np.ascontiguousarray(p[k + ".weight"][:, :, 0, 0], dtype=np.float32), p[k + ".bias"].astype(np.float32)[:, None])
        self.lift, self.p0, self.p2 = conv("lift"), conv("proj.0"), conv("proj.2")
        self.local = [conv(f"local.{k}") for k in range(self.layers)]
        # spectral weights (2, I, O, m, m) -> (2m * m, I, O): one (B, I) x (I, O) product per retained mode
        self.spec = [np.ascontiguousarray(np.concatenate([W[0], W[1]], 2).transpose(2, 3, 0, 1).reshape(2 * m * m, W.shape[1], W.shape[2]))
                     for W in (p[f"spectral.{k}.w"] for k in range(self.layers))]
        self._bases = {}

    def _spectral(self, x, k, bases):
        c, b, n, w = x.shape
        m = self.modes
        fwd_w, fwd_h, inv_h, inv_w = bases
        y = _mm(x.reshape(c * b * n, w), fwd_w).reshape(c * b, n, 2, m)            # along w
        y = np.ascontiguousarray(y.transpose(2, 1, 0, 3)).reshape(2 * n, c * b * m)  # [Re; Im] rows over h
        z = _mm(fwd_h, y).reshape(2, 2 * m, c, b, m)                                 # along h: (Re/Im, kx, C, B, ky)
        z = (z[0] + 1j * z[1]).transpose(0, 3, 2, 1).reshape(2 * m * m, b, c)       # (mode, B, C)
        v = (z @ self.spec[k]).reshape(2 * m, m, b, -1)                              # (kx, ky, B, O)
        o = v.shape[-1]
        v = v.transpose(0, 3, 2, 1).reshape(2 * m, o * b * m)
        g = _mm(inv_h, np.ascontiguousarray(np.concatenate([v.real, v.imag]), dtype=np.float32))  # (2n, O B m)
        g = np.ascontiguousarray(g.reshape(2, n, o, b, m).transpose(2, 3, 1, 0, 4)).reshape(o * b * n, 2 * m)
        return _mm(g, inv_w).reshape(o, b, n, w)

    def __call__(self, loga):
        """loga: (B, N, N) log10 coefficient. Returns (B, N, N) solution for unit source, float64.
        Computes in float32 like the torch model."""
        b, n, w = loga.shape
        key = (n, w)
        if key not in self._bases:
            self._bases[key] = _dft_bases(n, w, self.modes)
        gx = np.broadcast_to(((np.arange(n) + 0.5) / n)[None, :, None], (b, n, w))
        gy = np.broadcast_to(((np.arange(w) + 0.5) / w)[None, None, :], (b, n, w))
        conv = lambda wb, x: _mm(wb[0], x.reshape(x.shape[0], -1)) + wb[1]
        x = conv(self.lift, np.stack([loga, gx, gy]).astype(np.float32)).reshape(-1, b, n, w)
        for k in range(self.layers):
            x = self._spectral(x, k, self._bases[key]) + conv(self.local[k], x).reshape(x.shape)
            if k < self.layers - 1:
                x = _gelu(x)
        x = _gelu(conv(self.p0, x))
        return conv(self.p2, x).reshape(b, n, w).astype(np.float64) * self.meta["y_scale"]
