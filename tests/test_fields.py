from pathlib import Path

import numpy as np
import pytest
from scipy.stats import ks_2samp

from flowguard.fields import FAMILIES, pdebench

DATA = Path("data/raw/2D_DarcyFlow_beta1.0_Train.hdf5")


@pytest.mark.skipif(not DATA.exists(), reason="PDEBench file not downloaded")
def test_pdebench_replica_matches_dataset_high_fraction_distribution():
    import h5py

    with h5py.File(DATA, "r") as h:
        real = (h["nu"][:2000] > 0.5).mean(axis=(1, 2))
    rng = np.random.default_rng(0)
    ours = np.concatenate([(pdebench(200, rng) > 0.5).mean(axis=(1, 2)) for _ in range(10)])
    stat = ks_2samp(real, ours).statistic
    print("KS high-fraction (batch 200):", stat, "means", real.mean(), ours.mean())
    assert stat < 0.06


@pytest.mark.parametrize("name", list(FAMILIES))
def test_families_shape_and_range(name):
    a = FAMILIES[name](4, np.random.default_rng(0), n=64)
    assert a.shape == (4, 64, 64)
    assert a.min() >= 0.1 - 1e-12 and a.max() <= 1.0 + 1e-12
