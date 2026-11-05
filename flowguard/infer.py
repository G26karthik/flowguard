"""Predict steady Darcy solutions with exported FNO weights (NumPy only, no torch at serving time).

Models are trained for beta = 1. The steady problem is linear in the source, so u_beta = beta * u_1 exactly.
"""
from pathlib import Path

import numpy as np

from flowguard.npfno import NumpyFNO


class Predictor:
    def __init__(self, paths):
        # a training checkpoint path (model.pt) resolves to its exported weights.npz
        self.models = [NumpyFNO(Path(p).with_name("weights.npz") if str(p).endswith(".pt") else p) for p in paths]
        discs = {tuple(m.meta.get("discretisation", ("arith", "ghost"))) for m in self.models}
        if len(discs) != 1:
            raise ValueError(f"ensemble members disagree on discretisation: {discs}")
        self.face, self.bc = discs.pop()

    def predict(self, a, beta=1.0, batch=32):
        """a: (B, N, N) or (N, N) positive coefficients. Returns members (M, B, N, N) or (M, N, N), float64."""
        a = np.asarray(a, dtype=np.float64)
        single = a.ndim == 2
        a = a[None] if single else a
        if not np.all(a > 0):
            raise ValueError("coefficient field must be strictly positive")
        loga = np.log10(a)
        out = np.stack([np.concatenate([m(loga[i:i + batch]) for i in range(0, len(a), batch)]) for m in self.models]) * beta
        return out[:, 0] if single else out
