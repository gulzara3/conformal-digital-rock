#!/usr/bin/env python
"""Train one network and save its predictions in the release format.

The Colab notebook in ``notebooks/`` runs the full plan of 30 networks. This script trains a single run with
the same settings, for readers who prefer a command line or want to retrain one seed.

Examples
--------
    # retrain main seed 0 on the released split (needs data/processed from prepare_deepore.py)
    python scripts/train.py --data data/processed --run main_s0 --out runs/

    # a fresh random split with another seed
    python scripts/train.py --data data/processed --run my_run --seed 3 --out runs/

    # quick code test on the synthetic build
    python scripts/train.py --data /tmp/cdr_smoke --run smoke --seed 0 --epochs 2 --batch 8 --mc-T 3 --out /tmp/cdr_runs

GPU results match the paper statistically, not bit for bit: cuDNN kernels and mixed precision are not
deterministic across hardware.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from conformal_rock.config import CFG  # noqa: E402
from conformal_rock.data import input_scales  # noqa: E402
from conformal_rock.model import RockNet, loss_fn, n_parameters, set_mc_dropout  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")


def split_pool(n: int, seed: int) -> dict:
    """Random split into train / monitor / calibration / test, identical to the notebook's rule."""
    rng = np.random.default_rng(1000 + seed)
    p = rng.permutation(np.arange(n))
    fr = (CFG.frac_train, CFG.frac_val, CFG.frac_cal, CFG.frac_test)
    a = int(round(fr[0] * n)); b = a + int(round(fr[1] * n)); c = b + int(round(fr[2] * n))
    parts = [p[:a], p[a:b], p[b:c], p[c:]]
    return dict(tr=np.sort(parts[0]), val=np.sort(parts[1]), cal=np.sort(parts[2]), te=np.sort(parts[3]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--run", default="main_s0", help="run name; a released split with this name is reused")
    ap.add_argument("--seed", type=int, default=None, help="split and initialisation seed (default: from run name)")
    ap.add_argument("--mode", default="joint", choices=["joint", "mean_only", "gaussian"])
    ap.add_argument("--epochs", type=int, default=CFG.epochs)
    ap.add_argument("--batch", type=int, default=CFG.batch)
    ap.add_argument("--mc-T", type=int, default=CFG.mc_T)
    ap.add_argument("--out", default="runs")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    meta = json.load(open(os.path.join(a.data, "proc_meta.json")))
    arr = np.load(os.path.join(a.data, "arrays.npz"))
    yt = arr["Yt"].astype(np.float32)
    xw = np.load(os.path.join(a.data, "X_work.npy"), mmap_mode="r")
    n, P = yt.shape
    frac_scale, pore_scale = input_scales(meta["K_W"], meta["poremap_cap"])

    seed = a.seed if a.seed is not None else (int(a.run.split("_s")[-1]) if "_s" in a.run else 0)
    released = os.path.join(RESULTS, "splits", f"{a.run}.npz")
    if os.path.exists(released) and len(np.load(released)["h5_index"]) == n:
        z = np.load(released)
        split = {k: z[k] for k in ("tr", "val", "cal", "te")}
        print(f"using the released split of {a.run}")
    else:
        split = split_pool(n, seed)
        print(f"random split with seed {seed}")

    xt = torch.from_numpy(np.ascontiguousarray(xw))
    y_t = torch.from_numpy(yt).to(dev)
    taus = torch.tensor(CFG.q_levels, device=dev).view(1, 1, -1)

    def batch_x(idx, train=False):
        x = xt[idx].to(dev).float()
        x = torch.cat([x[:, 0:1] / frac_scale, x[:, 1:2] / pore_scale], 1)
        if train:
            flips = [d for d in (2, 3, 4) if torch.rand(()) < 0.5]
            if flips:
                x = torch.flip(x, flips)
        return x

    torch.manual_seed(10_000 + seed); np.random.seed(10_000 + seed)
    tr = torch.from_numpy(split["tr"]); va = torch.from_numpy(split["val"])
    ym, ys = y_t[tr.to(dev)].mean(0), y_t[tr.to(dev)].std(0) + 1e-8
    model = RockNet(2, P, len(CFG.q_levels), CFG.base, CFG.p_drop).to(dev)
    print(f"RockNet with {n_parameters(model):,} parameters on {dev}")
    opt = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.wd)
    steps = max(1, len(tr) // a.batch); total = steps * a.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=CFG.lr, total_steps=total, pct_start=0.15)
    amp = dev == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    ac = (lambda: torch.autocast("cuda", dtype=torch.float16)) if amp else contextlib.nullcontext
    g = torch.Generator().manual_seed(20_000 + seed)
    hist = []
    t_start = time.time()
    for ep in range(a.epochs):
        model.train(); perm = tr[torch.randperm(len(tr), generator=g)]; tl = 0.0
        for b in range(steps):
            idx = perm[b * a.batch:(b + 1) * a.batch]
            x = batch_x(idx, train=True); y = (y_t[idx.to(dev)] - ym) / ys
            with ac():
                out = model(x)
            loss = loss_fn(out, y, a.mode, taus, CFG.w_nll, CFG.w_q, warm=(ep * steps + b) < 0.1 * total)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward()
            scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step(); tl += loss.item()
        model.eval(); se = 0.0
        with torch.no_grad():
            for b in range(0, len(va), 128):
                idx = va[b:b + 128]
                se += ((model(batch_x(idx))[0].float() - (y_t[idx.to(dev)] - ym) / ys) ** 2).sum().item()
        hist.append(dict(epoch=ep + 1, train_loss=tl / steps, val_mse_z=se / max(1, len(va) * P)))
        print(f"epoch {ep + 1:3d}  train loss {tl / steps:.4f}  monitor MSE (z) {hist[-1]['val_mse_z']:.4f}")

    os.makedirs(os.path.join(a.out, "ckpt"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "preds"), exist_ok=True)
    torch.save(dict(model=model.state_dict(), ym=ym.cpu(), ys=ys.cpu(), state=dict(hist=hist),
                    run=dict(tag=a.run, mode=a.mode, variant="main", init_seed=seed)), os.path.join(a.out, "ckpt", a.run + ".pt"))

    # ---- predictions in target units, in the release format ----
    ymn, ysn = ym.cpu().numpy(), ys.cpu().numpy()
    flat = {}
    with torch.no_grad():
        for part, ids in split.items():
            it = torch.from_numpy(ids); acc = {k: [] for k in ("mu", "sig", "q", "feat")}
            set_mc_dropout(model, False)
            for b in range(0, len(it), 256):
                mu, lv, q, z = model(batch_x(it[b:b + 256]))
                acc["mu"].append(mu.cpu()); acc["sig"].append(torch.exp(0.5 * lv).cpu()); acc["q"].append(q.cpu()); acc["feat"].append(z.cpu())
            o = {k: torch.cat(v).numpy() for k, v in acc.items()}
            res = dict(idx=ids, y=yt[ids], mu=o["mu"] * ysn + ymn, sig=o["sig"] * ysn,
                       q=o["q"] * ysn[None, :, None] + ymn[None, :, None], feat=o["feat"].astype(np.float16))
            if part != "tr" and a.mode == "joint" and a.mc_T > 0:
                s1 = np.zeros((len(ids), P)); s2 = np.zeros_like(s1); sv = np.zeros_like(s1)
                set_mc_dropout(model, True)
                for _ in range(a.mc_T):
                    mus, vs = [], []
                    for b in range(0, len(it), 256):
                        mu, lv, _, _ = model(batch_x(it[b:b + 256])); mus.append(mu.cpu()); vs.append(torch.exp(lv).cpu())
                    m_ = torch.cat(mus).numpy() * ysn + ymn; s1 += m_; s2 += m_ ** 2; sv += torch.cat(vs).numpy() * ysn ** 2
                T = a.mc_T
                res.update(mc_mu=s1 / T, mc_vm=np.maximum(s2 / T - (s1 / T) ** 2, 0) * (T / max(T - 1, 1)), mc_mv=sv / T)
                set_mc_dropout(model, False)
            flat.update({f"{part}__{k}": v for k, v in res.items()})
    np.savez_compressed(os.path.join(a.out, "preds", a.run + ".npz"), **flat)
    json.dump(dict(run=a.run, seed=seed, mode=a.mode, epochs=a.epochs, train_s=time.time() - t_start, hist=hist,
                   n=dict((k, int(len(v))) for k, v in split.items())),
              open(os.path.join(a.out, a.run + ".json"), "w"), indent=1)
    print(f"saved checkpoint and predictions to {a.out}")


if __name__ == "__main__":
    main()
