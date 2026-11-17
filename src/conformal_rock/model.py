"""The 3D residual network (Section 3.4, Table S28) and its loss.

Two-channel 64^3 input; three strided stages with 32, 64 and 128 channels; global average pooling;
a 128-unit layer; and three heads: mean, log-variance and ten quantiles for each of six properties.
With the default settings the network has 1,479,560 trainable parameters.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, c: int, p: float):
        super().__init__()
        self.c1 = nn.Conv3d(c, c, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm3d(c)
        self.c2 = nn.Conv3d(c, c, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm3d(c)
        self.drop = nn.Dropout3d(p)

    def forward(self, x):
        h = F.silu(self.b1(self.c1(x)))
        h = self.b2(self.c2(self.drop(h)))
        return F.silu(x + h)


def down(ci: int, co: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv3d(ci, co, 3, stride=2, padding=1, bias=False), nn.BatchNorm3d(co), nn.SiLU())


class RockNet(nn.Module):
    """Returns (mean, log-variance, quantiles, penultimate features) in standardised target units."""

    def __init__(self, in_ch: int = 2, n_out: int = 6, n_q: int = 10, base: int = 16, p: float = 0.1):
        super().__init__()
        b = base
        self.stem = nn.Sequential(nn.Conv3d(in_ch, b, 3, padding=1, bias=False), nn.BatchNorm3d(b), nn.SiLU())
        self.s1 = nn.Sequential(down(b, 2 * b), ResBlock(2 * b, p))
        self.s2 = nn.Sequential(down(2 * b, 4 * b), ResBlock(4 * b, p))
        self.s3 = nn.Sequential(down(4 * b, 8 * b), ResBlock(8 * b, p))
        self.fc = nn.Sequential(nn.Linear(8 * b, 8 * b), nn.SiLU())
        self.drop = nn.Dropout(p)
        self.h_mu = nn.Linear(8 * b, n_out)
        self.h_lv = nn.Linear(8 * b, n_out)
        self.h_q = nn.Linear(8 * b, n_out * n_q)
        self.n_out, self.n_q = n_out, n_q

    def forward(self, x):
        z = self.fc(self.s3(self.s2(self.s1(self.stem(x)))).mean((2, 3, 4)))
        zd = self.drop(z)
        return self.h_mu(zd), self.h_lv(zd).clamp(-10, 6), self.h_q(zd).view(-1, self.n_out, self.n_q), z


def n_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_mc_dropout(model: nn.Module, on: bool) -> None:
    """Batch norm always uses running statistics; dropout is active only for MC-dropout passes."""
    model.eval()
    if on:
        for m in model.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout3d)):
                m.train()


def loss_fn(out, y, mode: str, taus: torch.Tensor, w_nll: float = 0.5, w_q: float = 1.0, warm: bool = False):
    """Training loss.

    ``joint``     MSE + w_nll * Gaussian NLL + w_q * pinball loss (main runs)
    ``mean_only`` MSE only (ablation)
    ``gaussian``  Gaussian NLL only, with a short MSE warm-up (dedicated Gaussian and ensemble members)
    """
    mu, lv, q, _ = out
    mu, lv, q = mu.float(), lv.float(), q.float()
    mse = F.mse_loss(mu, y)
    if mode == "mean_only":
        return mse
    nll = 0.5 * (torch.exp(-lv) * (y - mu) ** 2 + lv).mean()
    if mode == "gaussian":
        return nll + (mse if warm else 0.0)
    e = y.unsqueeze(-1) - q
    return mse + w_nll * nll + w_q * torch.maximum(taus * e, (taus - 1) * e).mean()


def load_checkpoint(path, device="cpu"):
    """Load a released checkpoint. Returns (model, ym, ys, run_info); ym/ys de-standardise outputs."""
    ck = torch.load(path, map_location=device, weights_only=False)
    run = ck.get("run", {})
    in_ch = 1 if run.get("variant") in ("frac", "bin") else 2
    model = RockNet(in_ch=in_ch)
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    return model, ck["ym"].to(device), ck["ys"].to(device), run
