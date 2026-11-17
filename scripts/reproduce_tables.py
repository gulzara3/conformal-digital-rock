#!/usr/bin/env python
"""Recompute the headline tables from the released predictions and compare them with the released CSV files.

This checks that the numbers in the paper follow from the saved model outputs and from the code in
``src/conformal_rock``, independently of the notebook. It needs only a CPU and takes a few minutes.

Tables recomputed
-----------------
T06  coverage and width per property and method (Table S7)
T07  coverage errors across properties (Table 3)
T23  joint coverage (Table 4)
T27  Learn-then-Test triage in distribution (Table 7, Table S24)

Usage
-----
    python scripts/fetch_release.py          # once: downloads predictions into release/preds
    python scripts/reproduce_tables.py       # writes reproduced/*.csv and prints PASS / FAIL
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from conformal_rock import conformal as C  # noqa: E402
from conformal_rock import triage as T  # noqa: E402
from conformal_rock.config import CFG, J_K, PROP_NAMES  # noqa: E402
from conformal_rock.io import MAIN_RUNS, RESULTS, load_preds  # noqa: E402
from conformal_rock.shift import novelty_scores  # noqa: E402

METHODS = ["split", "norm", "cqr", "mondrian", "gauss", "mcd", "mcd_tot"]


def table_coverage(preds_root):
    rows = []
    for s, run in enumerate(MAIN_RUNS):
        Pm = load_preds(run, preds_root)
        for j, pn in enumerate(PROP_NAMES):
            for m in METHODS:
                lo, hi, _ = C.interval(m, Pm["cal"], Pm["te"], j, CFG.alpha, tr=Pm["tr"])
                cov = C.coverage(Pm["te"]["y"][:, j], lo, hi)
                rows.append(dict(seed=s, prop=pn, method=m, coverage=cov, abs_err=abs(cov - 0.9), width=float(np.mean(hi - lo))))
    R = pd.DataFrame(rows)
    t06 = R.groupby(["prop", "method"], sort=False).agg(coverage_mean=("coverage", "mean"), width_mean=("width", "mean")).reset_index()
    t07 = R.groupby("method", sort=False).agg(mean_abs_cov_error_pp=("abs_err", lambda x: 100 * x.mean()),
                                              max_abs_cov_error_pp=("abs_err", lambda x: 100 * x.max()),
                                              mean_coverage=("coverage", "mean")).reset_index()
    return t06, t07


def table_joint(preds_root):
    sets = {"k + F": [0, 1], "all 6": list(range(6))}
    rows = []
    for s, run in enumerate(MAIN_RUNS):
        Pm = load_preds(run, preds_root)
        cal, te, tr = Pm["cal"], Pm["te"], Pm["tr"]
        rsd = (tr["y"] - tr["mu"]).std(0)
        eps = CFG.eps_rel * tr["y"].std(0)
        for name, J in sets.items():
            y = te["y"][:, J]
            hw = np.array([C.cq(np.abs(cal["y"][:, j] - cal["mu"][:, j]), CFG.alpha) for j in J])
            res = {"Independent 90% intervals": C.joint_coverage(y, te["mu"][:, J] - hw, te["mu"][:, J] + hw)}
            lo, hi = C.bonferroni_box(cal, te, J, CFG.alpha)
            res["Bonferroni"] = C.joint_coverage(y, lo, hi)
            lo, hi, _ = C.joint_box(cal, te, J, CFG.alpha, scale_cal=rsd[J])
            res["Max-score, fixed scale"] = C.joint_coverage(y, lo, hi)
            lo, hi, _ = C.joint_box(cal, te, J, CFG.alpha, scale_cal=cal["sig"][:, J] + eps[J], scale_te=te["sig"][:, J] + eps[J])
            res["Max-score, adaptive scale"] = C.joint_coverage(y, lo, hi)
            for m, v in res.items():
                rows.append(dict(seed=s, set=name, method=m, joint_coverage=v))
    return pd.DataFrame(rows).groupby(["set", "method"], sort=False).joint_coverage.mean().reset_index()


SCORE_LABEL = {"gauss_sd": "Gaussian-head sd", "cqr_w": "CQR band width", "mc_sd": "MC-dropout total sd",
               "novelty": "Feature novelty", "random": "Random (no information)"}


def tri_scores(d, part, seed, nov):
    a, b = C.quantile_pair(CFG.alpha)
    j = J_K
    out = dict(gauss_sd=d["sig"][:, j], cqr_w=np.abs(d["q"][:, j, b] - d["q"][:, j, a]),
               mc_sd=np.sqrt(d["mc_vm"][:, j] + d["mc_mv"][:, j]),
               random=np.random.default_rng(1000 * seed + {"cal": 1, "te": 2, "ood": 3}.get(part, 4)).random(len(d["y"])))
    out["novelty"] = nov
    return out


def table_triage(preds_root):
    rows = []
    for s, run in enumerate(MAIN_RUNS):
        Pm = load_preds(run, preds_root)
        f = {p: Pm[p]["feat"] for p in ("tr", "val", "cal", "te")}
        n_cal, n_te = novelty_scores(f["tr"], f["cal"], f["te"])
        _, n_val = novelty_scores(f["tr"], f["cal"], f["val"])
        U = {p: tri_scores(Pm[p], p, s, nv) for p, nv in (("cal", n_cal), ("te", n_te), ("val", n_val))}
        e_cal = np.abs(Pm["cal"]["mu"][:, J_K] - Pm["cal"]["y"][:, J_K])
        e_te = np.abs(Pm["te"]["mu"][:, J_K] - Pm["te"]["y"][:, J_K])
        for tol in CFG.tol_factors:
            bc, bt = e_cal > np.log10(tol), e_te > np.log10(tol)
            for eps in CFG.risk_levels:
                for sc in SCORE_LABEL:
                    t = T.ltt_threshold(U["cal"][sc], bc, eps, CFG.ltt_delta, T.threshold_grid(U["val"][sc]))
                    r = T.evaluate(U["te"][sc], bt, t)
                    rows.append(dict(seed=s, tol_factor=tol, eps=eps, score=SCORE_LABEL[sc], **r))
    D = pd.DataFrame(rows)
    return D.groupby(["tol_factor", "eps", "score"], sort=False).agg(
        simulations_avoided=("accepted", "mean"), bad_accept_rate=("bad_accept", "mean")).reset_index()


def compare(name, ours, ref, keys, cols, tol):
    m = ours.merge(ref, on=keys, suffixes=("", "_released"))
    worst = max(float(np.nanmax(np.abs(m[c] - m[c + "_released"]))) for c in cols)
    ok = len(m) == len(ours) and worst <= tol
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<34s} {len(m):4d} rows compared, largest difference {worst:.2e}")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preds", default=None, help="folder with preds/*.npz (default: release/preds)")
    ap.add_argument("--out", default="reproduced")
    ap.add_argument("--skip-triage", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tab = RESULTS / "tables"
    ok = []
    print("recomputing coverage tables ...")
    t06, t07 = table_coverage(a.preds)
    t06.to_csv(os.path.join(a.out, "T06_reproduced.csv"), index=False)
    t07.to_csv(os.path.join(a.out, "T07_reproduced.csv"), index=False)
    ok.append(compare("T06 coverage and width", t06, pd.read_csv(tab / "T06_coverage_and_width_by_property.csv"),
                      ["prop", "method"], ["coverage_mean", "width_mean"], 1e-6))
    ok.append(compare("T07 coverage errors (Table 3)", t07, pd.read_csv(tab / "T07_summary_across_properties.csv"),
                      ["method"], ["mean_abs_cov_error_pp", "max_abs_cov_error_pp", "mean_coverage"], 1e-6))
    print("recomputing joint coverage ...")
    t23 = table_joint(a.preds)
    t23.to_csv(os.path.join(a.out, "T23_reproduced.csv"), index=False)
    ok.append(compare("T23 joint coverage (Table 4)", t23, pd.read_csv(tab / "T23_joint_coverage.csv"),
                      ["set", "method"], ["joint_coverage"], 1e-6))
    if not a.skip_triage:
        print("recomputing Learn-then-Test triage ...")
        t27 = table_triage(a.preds)
        t27.to_csv(os.path.join(a.out, "T27_reproduced.csv"), index=False)
        ok.append(compare("T27 triage (Table 7, Table S24)", t27, pd.read_csv(tab / "T27_triage_in_distribution.csv"),
                          ["tol_factor", "eps", "score"], ["simulations_avoided", "bad_accept_rate"], 1e-6))
    print("ALL PASS" if all(ok) else "SOME TABLES DIFFER")
    sys.exit(0 if all(ok) else 1)


if __name__ == "__main__":
    main()
