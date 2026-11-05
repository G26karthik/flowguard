"""Coefficient-field generators: the PDEBench distribution and deliberate shifts away from it."""
import numpy as np


def _centres(n):
    return (np.arange(n) + 0.5) / n


def pdebench(count, rng, n=128, bumps=5, a_low=0.1, a_high=1.0):
    """PDEBench Darcy sampler (data_gen_NLE ReactionDiffusionEq): 5 Gaussian bumps, thresholded at the mean.

    PDEBench thresholds at the mean of the whole generated batch; we do the same, so count=1 thresholds
    per sample (a slightly different distribution - measured in tests/test_fields.py).
    """
    c = _centres(n)
    xm, ym = rng.uniform(0, 1, (2, count, bumps))
    std = 0.5 * rng.uniform(0, 1, (count, bumps))
    s = np.zeros((count, n, n))
    for i in range(bumps):
        s += np.exp(-((c[None, :, None] - xm[:, None, None, i]) ** 2 + (c[None, None, :] - ym[:, None, None, i]) ** 2) / std[:, None, None, i])
    return np.where(s > s.mean(), a_high, a_low)


def grf(count, rng, n=128, length=0.1, a_low=0.1, a_high=1.0):
    """Smooth log-normal field: Gaussian random field with squared-exponential spectrum, mapped to [a_low, a_high] in log space."""
    k = np.fft.fftfreq(n, d=1 / n)
    k2 = k[:, None] ** 2 + k[None, :] ** 2
    amp = np.exp(-0.5 * (2 * np.pi * length) ** 2 * k2)
    g = np.fft.ifft2(amp * np.fft.fft2(rng.standard_normal((count, n, n)))).real
    g = (g - g.mean(axis=(1, 2), keepdims=True)) / g.std(axis=(1, 2), keepdims=True)
    t = 1 / (1 + np.exp(-2 * g))  # squash to (0, 1)
    return np.exp(np.log(a_low) + t * (np.log(a_high) - np.log(a_low)))


def channels(count, rng, n=128, k=3, width=0.03, a_low=0.1, a_high=1.0):
    """k straight high-permeability channels at random angles through a low-permeability background."""
    c = _centres(n)
    x, y = np.meshgrid(c, c, indexing="ij")
    out = np.full((count, n, n), a_low)
    for b in range(count):
        for _ in range(k):
            th = rng.uniform(0, np.pi)
            px, py = rng.uniform(0.2, 0.8, 2)
            d = np.abs((x - px) * np.sin(th) - (y - py) * np.cos(th))
            out[b][d < width] = a_high
    return out


def layers(count, rng, n=128, k=6, a_low=0.1, a_high=1.0):
    """Tilted strata with random thicknesses, alternating permeability (sedimentary layering).
    Held out from every training set: used only to test generalisation of adapted models."""
    c = _centres(n)
    x, y = np.meshgrid(c, c, indexing="ij")
    out = np.empty((count, n, n))
    for b in range(count):
        th = rng.uniform(-0.4, 0.4)
        s = y * np.cos(th) + x * np.sin(th)
        cuts = np.sort(rng.uniform(s.min(), s.max(), k - 1))
        out[b] = np.where(np.searchsorted(cuts, s) % 2 == 0, a_high, a_low)
    return out


FAMILIES = {"pdebench": pdebench, "grf": grf, "channels": channels, "layers": layers}
