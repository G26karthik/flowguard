"""Fourier Neural Operator (Li et al., 2021) for 2D fields on a uniform grid."""
import torch
import torch.nn as nn


class SpectralConv2d(nn.Module):
    def __init__(self, width, modes):
        super().__init__()
        self.modes = modes
        scale = 1 / (width * width)
        # two weight blocks: positive and negative frequencies along the first axis
        self.w = nn.Parameter(scale * torch.randn(2, width, width, modes, modes, dtype=torch.cfloat))

    def forward(self, x):
        b, c, h, w = x.shape
        m = self.modes
        xf = torch.fft.rfft2(x)
        out = torch.zeros(b, c, h, w // 2 + 1, dtype=torch.cfloat, device=x.device)
        out[:, :, :m, :m] = torch.einsum("bixy,ioxy->boxy", xf[:, :, :m, :m], self.w[0])
        out[:, :, -m:, :m] = torch.einsum("bixy,ioxy->boxy", xf[:, :, -m:, :m], self.w[1])
        return torch.fft.irfft2(out, s=(h, w))


class FNO2d(nn.Module):
    """Maps a coefficient field (B, 1, N, N) to a solution field (B, 1, N, N).

    Grid coordinates are appended as input channels. Works at any N >= 2*modes,
    but accuracy away from the training resolution must be measured, not assumed.
    """

    def __init__(self, modes=12, width=32, layers=4, in_ch=1):  # defaults = the trained configuration (scripts/train.py)
        super().__init__()
        self.cfg = dict(modes=modes, width=width, layers=layers, in_ch=in_ch)
        self.lift = nn.Conv2d(in_ch + 2, width, 1)
        self.spectral = nn.ModuleList(SpectralConv2d(width, modes) for _ in range(layers))
        self.local = nn.ModuleList(nn.Conv2d(width, width, 1) for _ in range(layers))
        self.proj = nn.Sequential(nn.Conv2d(width, 128, 1), nn.GELU(), nn.Conv2d(128, 1, 1))

    def forward(self, a):
        b, _, n, m = a.shape
        gx = ((torch.arange(n, device=a.device) + 0.5) / n).view(1, 1, n, 1).expand(b, 1, n, m)
        gy = ((torch.arange(m, device=a.device) + 0.5) / m).view(1, 1, 1, m).expand(b, 1, n, m)
        x = self.lift(torch.cat([a, gx, gy], 1))
        for k, (s, l) in enumerate(zip(self.spectral, self.local)):
            x = s(x) + l(x)
            if k < len(self.spectral) - 1:
                x = nn.functional.gelu(x)
        return self.proj(x)
