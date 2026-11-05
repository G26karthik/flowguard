import json

import numpy as np
import torch

from flowguard.model import FNO2d
from flowguard.npfno import NumpyFNO


def test_numpy_forward_matches_torch(tmp_path):
    torch.manual_seed(0)
    m = FNO2d(modes=6, width=8, layers=3).eval()
    arrays = {k: v.numpy() for k, v in m.state_dict().items()}
    meta = {"cfg": m.cfg, "y_scale": 0.7}
    np.savez(tmp_path / "w.npz", meta=json.dumps(meta), **arrays)
    loga = np.random.default_rng(0).uniform(-1, 0, (3, 32, 32))
    with torch.no_grad():
        ref = m(torch.from_numpy(loga).float().unsqueeze(1)).squeeze(1).numpy() * 0.7
    got = NumpyFNO(tmp_path / "w.npz")(loga)
    np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-5)
