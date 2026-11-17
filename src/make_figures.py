import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

UP = "../results/"
OUT = "../figures/"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12.5,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10.5,
    "axes.titleweight": "bold",
    "axes.edgecolor": "#2b2b2b",
    "axes.linewidth": 1.0,
    "savefig.dpi": 300,
    "figure.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
})

# grayscale-safe encoding: colour + marker + linestyle all differ
STYLE = {
    "Split-Conformal":        dict(color="#1f4e79", marker="o", ls="-",   mfc="#1f4e79"),
    "Norm-Conformal":         dict(color="#2e7d32", marker="s", ls="--",  mfc="white"),
    "CQR":                    dict(color="#6a1b9a", marker="^", ls="-.",  mfc="#6a1b9a"),
    "Mondrian-Conformal":     dict(color="#00838f", marker="D", ls=":",   mfc="white"),
    "Gaussian (heterosced.)": dict(color="#b35c00", marker="v", ls="-",   mfc="white"),
    "MC-Dropout":             dict(color="#8d6e00", marker="X", ls="--",  mfc="#8d6e00"),
}
# grayscale-safe fills: vary hatch as well as shade
FILL = {
    "Split-Conformal":        dict(fc="#1f4e79", hatch=""),
    "Norm-Conformal":         dict(fc="#5b8db8", hatch="//"),
    "CQR":                    dict(fc="#9fc0d9", hatch=".."),
    "Mondrian-Conformal":     dict(fc="#cfe0ec", hatch="xx"),
    "Gaussian (heterosced.)": dict(fc="#b35c00", hatch="\\\\"),
    "MC-Dropout":             dict(fc="#e0b877", hatch="++"),
}
NICE = {"log_k": "log$_{10}$ k", "porosity": "porosity $\\phi$"}
ORDER = ["Split-Conformal","Norm-Conformal","CQR","Mondrian-Conformal",
         "Gaussian (heterosced.)","MC-Dropout"]
SHORT = {"Split-Conformal":"Split","Norm-Conformal":"Normalized","CQR":"CQR",
         "Mondrian-Conformal":"Mondrian","Gaussian (heterosced.)":"Gaussian","MC-Dropout":"MC-dropout"}

def save(fig, name):
    fig.savefig(OUT+name+".png", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT+name+".pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", name)

main = pd.read_csv(UP+"table_main_coverage.csv")
ood  = pd.read_csv(UP+"table_ood.csv")
sweep= pd.read_csv(UP+"table_sweep.csv")
summ = pd.read_csv(UP+"table_summary.csv")

# ============ FIG 4 (reliability / coverage vs target) ============
props = ["log_k","porosity"]
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
for ax, prop in zip(axes, props):
    sub = sweep[sweep.property==prop]
    for m in ORDER:
        g = sub[sub.method==m].sort_values("target")
        if len(g) == 0:
            continue
        st = STYLE[m]
        ax.plot(g["target"], g["coverage"], color=st["color"], marker=st["marker"],
                ls=st["ls"], ms=5.5, lw=1.8, mfc=st["mfc"], mew=1.2, label=SHORT[m])
    ax.plot([0.68,1.0],[0.68,1.0], color="#555", ls=(0,(4,3)), lw=1.2, zorder=0,
            label="ideal" if prop=="log_k" else None)
    ax.set_xlabel("target coverage  $1-\\alpha$")
    ax.set_title(NICE[prop])
    ax.set_xlim(0.68,1.0)
axes[0].set_ylabel("empirical coverage")
# legend OUTSIDE to avoid overlap (R2 #6)
h,l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="center left", bbox_to_anchor=(1.005, 0.5), frameon=False)
fig.tight_layout()
save(fig, "Fig4_reliability")

# ============ FIG 5 (interval width distributions) ============
# reconstruct per-sample widths is not available; use reported widths as bars w/ hatching
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
conf_only = ["Split-Conformal","Norm-Conformal","CQR","Mondrian-Conformal"]
for ax, prop in zip(axes, props):
    sub = main[main.property==prop].set_index("method")
    vals = [sub.loc[m,"width_real"] for m in conf_only]
    xs = np.arange(len(conf_only))
    for x, m, v in zip(xs, conf_only, vals):
        f = FILL[m]
        ax.bar(x, v, 0.62, facecolor=f["fc"], edgecolor="#1b1b1b",
               hatch=f["hatch"], linewidth=1.1)
        ax.text(x, v*1.02, f"{v:.3g}", ha="center", va="bottom", fontsize=10)
    ax.set_xticks(xs); ax.set_xticklabels([SHORT[m] for m in conf_only], rotation=18, ha="right")
    ax.set_ylabel("mean interval width (real units)")
    ax.set_title(NICE[prop])
    ax.set_ylim(0, max(vals)*1.18)
fig.suptitle("Interval width of the conformal methods at 90% target coverage",
             y=1.02, fontsize=13, fontweight="bold")
fig.tight_layout()
save(fig, "Fig5_widths")

# ============ FIG 6 (coverage bars, marginal + worst-slice) ============
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)
w = 0.38
for ax, prop in zip(axes, props):
    sub = main[main.property==prop].set_index("method")
    xs = np.arange(len(ORDER))
    marg = [sub.loc[m,"coverage"] for m in ORDER]
    wors = [sub.loc[m,"worst_slice_cov"] for m in ORDER]
    ax.bar(xs-w/2, marg, w, facecolor="#1f4e79", edgecolor="#1b1b1b", lw=1.0, label="marginal")
    ax.bar(xs+w/2, wors, w, facecolor="#cfe0ec", edgecolor="#1b1b1b", lw=1.0,
           hatch="//", label="worst-slice (conditional)")
    ax.axhline(0.90, color="#111", ls=(0,(4,3)), lw=1.4, label="target 0.90")
    ax.set_xticks(xs); ax.set_xticklabels([SHORT[m] for m in ORDER], rotation=22, ha="right")
    ax.set_title(NICE[prop]); ax.set_ylim(0.60, 1.03)
axes[0].set_ylabel("empirical coverage")
h,l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 1.09), ncol=3, frameon=False)
fig.tight_layout()
save(fig, "Fig6_coverage_bars")

# ============ FIG 7 (OOD: coverage + width, in-dist vs OOD) ============
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
mlist = ["Split-Conformal","Norm-Conformal","CQR","Gaussian (heterosced.)","MC-Dropout"]
sub = ood[ood.property=="log_k"]
xs = np.arange(len(mlist)); w=0.38
for k,(reg,fc,ht) in enumerate([("In-distribution","#1f4e79",""),
                                ("Out-of-distribution","#cfe0ec","//")]):
    v=[sub[(sub.method==m)&(sub.regime==reg)]["coverage"].values[0] for m in mlist]
    axes[0].bar(xs+(k-0.5)*w, v, w, facecolor=fc, edgecolor="#1b1b1b", lw=1.0,
                hatch=ht, label=reg)
axes[0].axhline(0.90,color="#111",ls=(0,(4,3)),lw=1.4,label="target 0.90")
axes[0].set_xticks(xs); axes[0].set_xticklabels([SHORT[m] for m in mlist], rotation=22, ha="right")
axes[0].set_ylabel("coverage"); axes[0].set_title("(a) Coverage under label shift")
axes[0].legend(frameon=False, loc="upper right")
for k,(reg,fc,ht) in enumerate([("In-distribution","#1f4e79",""),
                                ("Out-of-distribution","#cfe0ec","//")]):
    v=[sub[(sub.method==m)&(sub.regime==reg)]["width"].values[0] for m in mlist]
    axes[1].bar(xs+(k-0.5)*w, v, w, facecolor=fc, edgecolor="#1b1b1b", lw=1.0,
                hatch=ht, label=reg)
axes[1].set_xticks(xs); axes[1].set_xticklabels([SHORT[m] for m in mlist], rotation=22, ha="right")
axes[1].set_ylabel("mean interval width (standardized)")
axes[1].set_title("(b) Interval width under label shift")
axes[1].legend(frameon=False, loc="upper right")
fig.tight_layout()
save(fig, "Fig7_ood")

# ============ NEW FIG 8: sensitivity studies (reviewer-requested) ============
eps = pd.read_csv("../results/sens_epsilon.csv")
cal = pd.read_csv("../results/sens_calsize.csv")
mon = pd.read_csv("../results/sens_mondrian.csv")

fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))

# (a) epsilon
a=axes[0]
a.semilogx(eps["epsilon"], eps["cov_logk"], color="#1f4e79", marker="o", ls="-", ms=6, label="log$_{10}$ k")
a.semilogx(eps["epsilon"], eps["cov_poro"], color="#2e7d32", marker="s", ls="--", ms=6, mfc="white", label="porosity $\\phi$")
a.axhline(0.90, color="#111", ls=(0,(4,3)), lw=1.3, label="target 0.90")
a.set_xlabel("$\\varepsilon$"); a.set_ylabel("empirical coverage")
a.set_title("(a) Sensitivity to $\\varepsilon$"); a.set_ylim(0.85,0.95)
a.legend(frameon=False, loc="lower left")

# (b) calibration size
b=axes[1]
for col,lab,st in [("split","Split","#1f4e79"),("norm","Normalized","#2e7d32"),("mond","Mondrian","#00838f")]:
    b.errorbar(cal["n_cal"], cal[col+"_mean"], yerr=cal[col+"_sd"],
               color=st, marker="o" if col=="split" else ("s" if col=="norm" else "D"),
               ls="-" if col=="split" else ("--" if col=="norm" else ":"),
               ms=5.5, lw=1.7, capsize=3, mfc="white" if col!="split" else st, label=lab)
b.axhline(0.90, color="#111", ls=(0,(4,3)), lw=1.3, label="target 0.90")
b.set_xlabel("calibration set size  $n_{cal}$"); b.set_ylabel("empirical coverage")
b.set_title("(b) Sensitivity to calibration size"); b.set_ylim(0.85,0.97)
b.legend(frameon=False, loc="upper right")

# (c) Mondrian groups
c=axes[2]
c.plot(mon["n_groups"], mon["cov_logk"], color="#1f4e79", marker="o", ls="-", ms=6, label="marginal (log$_{10}$k)")
c.plot(mon["n_groups"], mon["worst_logk"], color="#00838f", marker="D", ls=":", ms=6, mfc="white", label="worst-slice (log$_{10}$k)")
c.axhline(0.90, color="#111", ls=(0,(4,3)), lw=1.3, label="target 0.90")
c.set_xlabel("number of Mondrian groups"); c.set_ylabel("empirical coverage")
c.set_title("(c) Sensitivity to group count"); c.set_ylim(0.80,0.96)
c.legend(frameon=False, loc="lower right")

fig.suptitle("Sensitivity analyses of the conformal calibration procedure",
             y=1.03, fontsize=13, fontweight="bold")
fig.tight_layout()
save(fig, "Fig8_sensitivity")
print("ALL FIGURES DONE")
