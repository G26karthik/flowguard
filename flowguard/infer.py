"""Load trained FNO checkpoints and predict steady Darcy solutions for a constant source beta.

Models are trained for beta = 1. The steady problem is linear in the source, so u_beta = beta * u_1 exactly.
"""
from pathlib import Path

import numpy as np
import torch

from flowguard.model import FNO2d


class Predictor:
    def __init__(self, paths, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.models, self.scales = [], []
        for p in paths:
            ck = torch.load(Path(p), map_location=self.device, weights_only=False)
            m = FNO2d(**ck["cfg"]).to(self.device).eval()
            m.load_state_dict(ck["state_dict"])
            self.models.append(m)
            self.scales.append(ck["y_scale"])

    @torch.no_grad()
    def predict(self, a, beta=1.0, batch=64):
        """a: (B, N, N) or (N, N) positive coefficients. Returns members (M, B, N, N) float64."""
        a = np.asarray(a, dtype=np.float32)
        single = a.ndim == 2
        a = a[None] if single else a
        if not np.all(a > 0):
            raise ValueError("coefficient field must be strictly positive")
        x = torch.from_numpy(np.log10(a)).unsqueeze(1)
        out = []
        for m, s in zip(self.models, self.scales):
            parts = [m(x[i:i + batch].to(self.device)).squeeze(1).cpu() * s for i in range(0, len(x), batch)]
            out.append(torch.cat(parts).double().numpy() * beta)
        out = np.stack(out)
        return out[:, 0] if single else out
