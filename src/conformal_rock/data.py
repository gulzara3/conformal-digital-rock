"""Rebuild the working dataset from the public DeePore archive (Sections 3.1 to 3.3).

Source: ``DeePore_Dataset.h5`` (Zenodo record 4297035, Rabbani et al., 2020) with datasets ``X``
(17,700 binary volumes of 256^3, stored 1 = solid) and ``Y`` (1,515 features per volume; the first 15 are
single-value properties such as permeability in px^2 and formation factor).

Pipeline
--------
1. Apply DeePore's own outlier rules and count the removals per rule (Table S1).
2. Check the phase convention: permeability must rise with pore fraction.
3. Coarse-grain each 256^3 volume to two channels at 64^3 from 4^3 blocks, without cropping or
   re-binarising: the block pore fraction (conserves porosity exactly) and the block maximum of a capped
   distance-to-solid map computed at full resolution.
4. Compute label-free image descriptors (porosity, surface density, pore-size statistics, heterogeneity).
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass

import numpy as np

from .config import CFG, EXCLUSION_RULES, PROPS

DESC_NAMES = ["phi", "sv", "r_mean", "r_p50", "r_p90", "r_max", "het"]


# ----------------------------------------------------------------------------------------------
# Archive access
# ----------------------------------------------------------------------------------------------
def md5sum(path: str, chunk: int = 1 << 24) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def read_labels(h5_path: str) -> np.ndarray:
    """All label rows as an (N, 1515) float64 array."""
    import h5py

    with h5py.File(h5_path, "r") as f:
        return np.squeeze(np.asarray(f["Y"][:, :, ...], dtype=np.float64))


def read_volume(h5_file, i: int) -> np.ndarray:
    """Boolean volume, True where the stored value is 1."""
    v = np.squeeze(np.asarray(h5_file["X"][int(i)]))
    hi = float(v.max())
    return v > (0.5 * hi if hi > 0 else 0.5)


# ----------------------------------------------------------------------------------------------
# Exclusions and targets
# ----------------------------------------------------------------------------------------------
def apply_exclusions(y_all: np.ndarray):
    """DeePore's outlier rules applied in order. Returns (valid_indices, table_rows)."""
    y15 = y_all[:, :15]
    rules = [("NaN or Inf in any feature", ~np.isfinite(y_all).all(1))]
    for name, col, op, val in EXCLUSION_RULES:
        with np.errstate(invalid="ignore"):
            rules.append((name, y15[:, col] > val if op == ">" else y15[:, col] < val))
    extra = np.zeros(len(y_all), bool)
    for _, col, tr, _ in PROPS:
        if tr == "log10":
            with np.errstate(invalid="ignore"):
                extra |= ~(y15[:, col] > 0)
    rules.append(("non-positive value in a log-transformed target", extra))
    bad = np.zeros(len(y_all), bool)
    rows = []
    for name, m in rules:
        m = np.asarray(m, bool)
        rows.append(dict(rule=name, n_flagged=int(m.sum()), n_new=int((m & ~bad).sum())))
        bad |= m
    rows.append(dict(rule="TOTAL excluded", n_flagged=int(bad.sum()), n_new=int(bad.sum())))
    rows.append(dict(rule="valid volumes", n_flagged=int((~bad).sum()), n_new=int((~bad).sum())))
    return np.where(~bad)[0], rows


def transform_targets(y_all: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """The six modelled targets (log10 where stated) for the selected rows, float32 (N, 6)."""
    y15 = y_all[rows, :15]
    cols = [np.log10(y15[:, c]) if tr == "log10" else y15[:, c] for _, c, tr, _ in PROPS]
    return np.stack(cols, 1).astype(np.float32)


# ----------------------------------------------------------------------------------------------
# Coarse-graining and descriptors (torch; runs on GPU if available)
# ----------------------------------------------------------------------------------------------
def featurise(pore, ks, cap: int):
    """Coarse inputs and descriptors for a batch of full-resolution volumes.

    pore : torch bool tensor [B, D, D, D], True = pore.
    ks   : coarse-graining factors (e.g. [4] for 256^3 -> 64^3).
    cap  : cap of the distance map in voxels.

    Returns ({k: uint8 array [B, 2, D/k, D/k, D/k]}, descriptors float array [B, 7 + cap + 1]).
    Channel 0 holds pore-voxel counts per block (scaled to 0..255 if k^3 > 255); channel 1 holds the block
    maximum of the distance map.
    """
    import torch
    import torch.nn.functional as F

    B, Dv = pore.shape[0], pore.shape[1]
    dt = torch.float16 if pore.is_cuda else torch.float32
    with torch.no_grad():
        m = pore[:, None].to(dt)
        phi = pore.float().mean((1, 2, 3))
        faces = ((pore[:, 1:] != pore[:, :-1]).sum((1, 2, 3)) + (pore[:, :, 1:] != pore[:, :, :-1]).sum((1, 2, 3))
                 + (pore[:, :, :, 1:] != pore[:, :, :, :-1]).sum((1, 2, 3))).float()
        sv = faces / Dv ** 3
        dist, cur = m.clone(), m
        for it in range(cap):          # alternate 6- and 26-neighbour erosions (octagonal metric)
            if it % 2 == 0:
                cur = torch.minimum(torch.minimum(-F.max_pool3d(-cur, (3, 1, 1), 1, (1, 0, 0)),
                                                  -F.max_pool3d(-cur, (1, 3, 1), 1, (0, 1, 0))),
                                    -F.max_pool3d(-cur, (1, 1, 3), 1, (0, 0, 1)))
            else:
                cur = -F.max_pool3d(-cur, 3, 1, 1)
            dist += cur
        nb = cap + 2
        off = torch.arange(B, device=pore.device).view(B, 1, 1, 1, 1) * nb
        hist = torch.bincount((dist.round().long() + off).flatten(), minlength=B * nb).view(B, nb).float()
        psd = hist[:, 1:] / hist[:, 1:].sum(1, keepdim=True).clamp(min=1)
        r = torch.arange(1, nb, device=pore.device).float()
        cdf = psd.cumsum(1)
        r_mean = (psd * r).sum(1)
        r_p50 = r[(cdf >= 0.5).float().argmax(1)]
        r_p90 = r[(cdf >= 0.9).float().argmax(1)]
        r_max = (psd > 0).float().mul(r).max(1).values
        het = F.avg_pool3d(pore[:, None].float(), max(Dv // 8, 1)).flatten(1).std(1)
        desc = torch.cat([torch.stack([phi, sv, r_mean, r_p50, r_p90, r_max, het], 1), psd], 1)
        out = {}
        for k in ks:
            cnt = F.avg_pool3d(pore[:, None].float(), k) * (k ** 3)
            scale = k ** 3 if k ** 3 <= 255 else 255
            fr = (cnt if scale == k ** 3 else cnt / (k ** 3) * 255).round().to(torch.uint8)
            pm = F.max_pool3d(dist.float(), k).round().to(torch.uint8)
            out[k] = torch.cat([fr, pm], 1).cpu().numpy()
    return out, desc.cpu().numpy()


def input_scales(k: int, cap: int) -> tuple[float, float]:
    """Divisors that map the stored uint8 channels to [0, 1]."""
    return float(k ** 3 if k ** 3 <= 255 else 255), float(cap + 1)


# ----------------------------------------------------------------------------------------------
# Full build
# ----------------------------------------------------------------------------------------------
@dataclass
class BuildResult:
    n: int
    out_dir: str
    ones_is_solid: bool
    phase_rho: float


def build_dataset(h5_path: str, out_dir: str, work_res: int | None = None, cap: int | None = None,
                  n_max: int | None = None, batch: int = 8, device: str | None = None,
                  check_md5: str | None = None, progress: bool = True) -> BuildResult:
    """Build ``X_work.npy`` (uint8, N x 2 x W^3), ``arrays.npz`` (indices, targets, descriptors),
    ``exclusions.csv`` and ``proc_meta.json`` in ``out_dir``.
    """
    import h5py
    import torch
    from scipy import stats

    work_res = work_res or CFG.work_res
    cap = cap or CFG.poremap_cap
    n_max = CFG.n_max if n_max is None else n_max
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(out_dir, exist_ok=True)

    md5 = None
    if check_md5:
        md5 = md5sum(h5_path)
        if md5 != check_md5:
            raise RuntimeError(f"MD5 {md5} does not match {check_md5}: the archive is incomplete or changed.")

    y_all = read_labels(h5_path)
    valid, rows = apply_exclusions(y_all)
    sel = valid
    if n_max and n_max < len(valid):
        sel = np.sort(np.random.default_rng(CFG.data_seed).choice(valid, n_max, replace=False))
    yt = transform_targets(y_all, sel)
    n = len(sel)

    with h5py.File(h5_path, "r") as f:
        full_res = int([s for s in f["X"].shape[1:] if s > 1][0])
        k = full_res // work_res
        assert full_res % work_res == 0, "full resolution must be a multiple of the working resolution"
        # phase convention: permeability must rise with pore fraction
        probe = np.sort(np.random.default_rng(1).choice(n, min(48, n), replace=False))
        ones_frac = [read_volume(f, sel[i]).mean() for i in probe]
        rho = float(stats.spearmanr(ones_frac, yt[probe, 0])[0])
        ones_is_solid = bool(rho < 0) if abs(rho) > 0.2 else True

        xw = np.lib.format.open_memmap(os.path.join(out_dir, "X_work.npy"), mode="w+", dtype=np.uint8,
                                       shape=(n, 2, work_res, work_res, work_res))
        desc = np.zeros((n, len(DESC_NAMES) + cap + 1), np.float32)
        it = range(0, n, batch)
        if progress:
            try:
                from tqdm import tqdm
                it = tqdm(it, desc="featurise", total=(n + batch - 1) // batch)
            except ImportError:
                pass
        for s in it:
            ids = range(s, min(s + batch, n))
            ones = np.stack([read_volume(f, sel[i]) for i in ids])
            pore = torch.from_numpy(~ones if ones_is_solid else ones).to(device)
            out, d = featurise(pore, [k], cap)
            xw[s:s + len(ids)] = out[k]
            desc[s:s + len(ids)] = d
        xw.flush()
        del xw

    np.savez_compressed(os.path.join(out_dir, "arrays.npz"), sel=sel, Yt=yt, DESC=desc)
    import csv
    with open(os.path.join(out_dir, "exclusions.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["rule", "n_flagged", "n_new"])
        w.writeheader()
        w.writerows(rows)
    meta = dict(N=n, N_ALL=len(y_all), FULL_RES=full_res, K_W=k, ONES_IS_SOLID=ones_is_solid, phase_rho=rho,
                source=os.path.basename(h5_path), md5=md5, poremap_cap=cap, work_res=work_res, n_max=n_max)
    with open(os.path.join(out_dir, "proc_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    return BuildResult(n=n, out_dir=out_dir, ones_is_solid=ones_is_solid, phase_rho=rho)


def make_synthetic_h5(path: str, n: int = 120, dv: int = 32, seed: int = 0) -> None:
    """A tiny file in DeePore's layout for tests ONLY (thresholded Gaussian fields, made-up labels)."""
    import h5py

    rng = np.random.default_rng(seed)
    fr = np.fft.fftfreq(dv)
    k2 = fr[:, None, None] ** 2 + fr[None, :, None] ** 2 + fr[None, None, :] ** 2
    with h5py.File(path, "w") as f:
        X = f.create_dataset("X", (n, dv, dv, dv), dtype="uint8")
        Y = f.create_dataset("Y", (n, 1515, 1), dtype="float64")
        for i in range(n):
            L = rng.uniform(1.5, 4.0)
            phi = rng.uniform(0.12, 0.45)
            fld = np.fft.ifftn(np.fft.fftn(rng.standard_normal((dv, dv, dv))) * np.exp(-0.5 * k2 * (2 * np.pi * L) ** 2)).real
            pore = fld < np.quantile(fld, phi)
            y = np.full(1515, 0.5)
            y[0] = max(0.02 * phi ** 3 * L ** 2 * 10 ** rng.normal(0, 0.1), 2e-4)
            y[1] = min(phi ** -1.8, 119.0); y[2] = 1.8; y[3] = 1e-3
            y[4] = min(1 + 0.5 * (1 - phi), 1.79); y[7] = L * (0.8 + phi); y[10] = 0.1; y[14] = min(0.6 * (1 - phi) ** 2, 0.69)
            if i % 37 == 5:
                y[0] = 1e-6                      # exercises the permeability rule
            X[i] = (~pore).astype("uint8")
            Y[i, :, 0] = y
