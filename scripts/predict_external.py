#!/usr/bin/env python
"""Apply a released network, its conformal calibration, the guard and the triage rule to new volumes.

Input: an ``.npz`` file with ``volumes`` (uint8, N x 256 x 256 x 256, 1 = solid, as in DeePore) and,
optionally, ``log_k`` labels (log10 permeability in px^2 of the same grid).

Output: a CSV with the predicted log10 permeability, its 90% CQR interval, the guard p-value, and the
accept-or-simulate decision at a 25% tolerance and a 5% bad-accept bound. With labels, coverage is
reported too.

Usage
-----
    python scripts/predict_external.py --volumes my_rocks.npz --ckpt checkpoints/main_s0.pt \
        --preds release/preds/main_s0.npz --out my_rocks_predictions.csv

The guarantees hold only for rocks exchangeable with DeePore's calibration volumes. Flagged rocks should be
simulated; calibration on rocks from the target formation is needed for a formal guarantee on new rocks.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch  # noqa: E402

from conformal_rock import conformal as C  # noqa: E402
from conformal_rock import triage as T  # noqa: E402
from conformal_rock.config import CFG, J_K  # noqa: E402
from conformal_rock.data import featurise, input_scales  # noqa: E402
from conformal_rock.io import load_preds  # noqa: E402
from conformal_rock.model import load_checkpoint  # noqa: E402
from conformal_rock.shift import conformal_pvalue, novelty_scores  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--volumes", required=True)
    ap.add_argument("--ckpt", default="checkpoints/main_s0.pt")
    ap.add_argument("--preds", default="release/preds/main_s0.npz", help="released predictions of the same run")
    ap.add_argument("--tol-factor", type=float, default=1.25)
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--work-res", type=int, default=CFG.work_res, help="must match the network (64 for released runs)")
    ap.add_argument("--cap", type=int, default=CFG.poremap_cap)
    ap.add_argument("--out", default="external_predictions.csv")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ext = np.load(a.volumes)
    V = ext["volumes"]
    full = V.shape[1]
    k = full // a.work_res
    frac_scale, pore_scale = input_scales(k, a.cap)

    model, ym, ys, _ = load_checkpoint(a.ckpt, dev)
    outs = []
    with torch.no_grad():
        for s in range(0, len(V), 4):
            pore = torch.from_numpy(V[s:s + 4] == 0).to(dev)
            o, _ = featurise(pore, [k], a.cap)
            x = torch.from_numpy(o[k]).to(dev).float()
            x = torch.cat([x[:, 0:1] / frac_scale, x[:, 1:2] / pore_scale], 1)
            mu, lv, q, z = model(x)
            outs.append((mu * ys + ym, torch.exp(0.5 * lv) * ys, q * ys[None, :, None] + ym[None, :, None], z))
    mu, sig, q, feat = (torch.cat([o[i] for o in outs]).cpu().numpy() for i in range(4))
    new = dict(mu=mu, sig=sig, q=q, feat=feat, y=np.full_like(mu, np.nan))

    Pm = load_preds(os.path.splitext(os.path.basename(a.preds))[0], os.path.dirname(a.preds) or ".")
    lo, hi, _ = C.interval("cqr", Pm["cal"], new, J_K, CFG.alpha, tr=Pm["tr"])
    nov_cal, nov_new = novelty_scores(Pm["tr"]["feat"], Pm["cal"]["feat"], feat)
    pv = conformal_pvalue(nov_cal, nov_new)
    bad_cal = T.bad_accept_labels(Pm["cal"]["mu"][:, J_K], Pm["cal"]["y"][:, J_K], a.tol_factor)
    t = T.ltt_threshold(Pm["cal"]["sig"][:, J_K], bad_cal, a.eps, CFG.ltt_delta, T.threshold_grid(Pm["val"]["sig"][:, J_K]))
    flagged = pv <= CFG.guard_delta
    df = pd.DataFrame(dict(pred_log10_k=mu[:, J_K], lo_90=lo, hi_90=hi, guard_pvalue=pv, flagged_unfamiliar=flagged,
                           decision=np.where((sig[:, J_K] <= t) & ~flagged, "accept", "simulate")))
    if "log_k" in ext.files:
        df["log10_k"] = ext["log_k"]
        df["covered"] = (df.log10_k >= lo) & (df.log10_k <= hi)
        print(f"coverage of the 90% interval on these volumes: {df.covered.mean():.3f}")
    df.to_csv(a.out, index=False)
    print(f"{len(df)} volumes: {int(flagged.sum())} flagged by the guard, {(df.decision == 'accept').sum()} accepted; wrote {a.out}")


if __name__ == "__main__":
    main()
