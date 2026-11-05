"""Export a torch checkpoint to NumPy weights for serving: python scripts/export_numpy.py runs/<run>/model.pt"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

src = Path(sys.argv[1])
ck = torch.load(src, map_location="cpu", weights_only=False)
arrays = {k: v.numpy() for k, v in ck["state_dict"].items()}
meta = {k: ck[k] for k in ("cfg", "y_scale", "input", "target", "seed", "discretisation") if k in ck}
np.savez_compressed(src.with_name("weights.npz"), meta=json.dumps(meta), **arrays)
print("wrote", src.with_name("weights.npz"), meta)
