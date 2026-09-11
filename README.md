# Distribution-Free Calibrated Uncertainty for Pore-Scale Petrophysical Property Prediction via Conformal Prediction

Reference implementation and experiment code for the manuscript submitted to
**Results in Engineering**.

This repository contains everything required to reproduce the reported results:
the full experiment notebook, the sensitivity-analysis scripts, the
figure-generation script, the result tables, and the publication figures.

---

## 1. Overview

Deep neural networks predict petrophysical properties directly from
micro-computed-tomography (µCT) images of rock far faster than pore-scale
simulation, but conventional predictors return a point estimate with no
statement of confidence, and the uncertainty methods in common use provide no
guarantee that their intervals contain the truth at the stated rate.

This work wraps a three-dimensional convolutional neural network with
**conformal prediction**, producing prediction intervals that carry a
finite-sample, distribution-free coverage guarantee. Four conformal
constructions are implemented and benchmarked against two heuristic baselines
on a single shared backbone:

| Method | Interval width | Guarantee |
|---|---|---|
| Split conformal | constant | marginal |
| Normalized (adaptive) conformal | input-dependent | marginal |
| Conformalized quantile regression (CQR) | input-dependent | marginal |
| Mondrian (group-conditional) conformal | per-stratum | conditional |
| Gaussian (heteroscedastic) — baseline | input-dependent | none |
| Monte Carlo dropout — baseline | input-dependent | none |

Targets are **log-permeability** and **porosity**, predicted from binary µCT
volumes of the DeePore dataset at 64³ resolution.

## 2. Repository layout

```
conformal-digital-rock/
├── README.md
├── LICENSE                       MIT
├── CITATION.cff
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── Conformal_DigitalRock.ipynb    full end-to-end experiment (Colab-ready)
├── src/
│   ├── sensitivity_analyses.py        epsilon, calibration-size, Mondrian-group studies
│   └── make_figures.py                regenerates all publication figures
├── results/
│   ├── table_main_coverage.csv        coverage, worst-slice, width per method/property
│   ├── table_ood.csv                  in-distribution vs. shifted regime
│   ├── table_sweep.csv                coverage-vs-alpha reliability data
│   ├── table_summary.csv              summary averaged over properties
│   ├── sens_epsilon.csv               sensitivity to the stabilising constant
│   ├── sens_calsize.csv               sensitivity to calibration-set size
│   └── sens_mondrian.csv              sensitivity to number of Mondrian strata
├── figures/
│   ├── Fig4_reliability.{png,pdf}     coverage vs. target coverage
│   ├── Fig5_widths.{png,pdf}          interval widths
│   ├── Fig6_coverage_bars.{png,pdf}   marginal and worst-slice coverage
│   ├── Fig7_ood.{png,pdf}             behaviour under label shift
│   └── Fig8_sensitivity.{png,pdf}     the three sensitivity analyses
├── docs/
│   └── USAGE.md                       detailed run instructions
└── data/
    └── README.md                      how to obtain and prepare DeePore
```

## 3. Requirements

Python 3.9 or newer. All dependencies are open-source; no proprietary software
is required.

```bash
pip install -r requirements.txt
```

The notebook is written for **Google Colab with a GPU runtime** and needs no
local setup. A full run over 4,000 volumes takes roughly 20–35 minutes on an
L4 GPU. The sensitivity scripts and the figure script run on CPU in seconds.

## 4. Quick start

**Reproduce the figures from the supplied result tables (no GPU, seconds):**

```bash
cd src
python sensitivity_analyses.py     # writes ../results/sens_*.csv
python make_figures.py             # writes ../figures/*.png and *.pdf
```

This is the fastest way to verify the installation and to confirm that the
figures in the manuscript follow from the tabulated results.

**Reproduce the full experiment (GPU recommended):**

1. Open `notebooks/Conformal_DigitalRock.ipynb` in Google Colab.
2. Select a GPU runtime.
3. Set the data path in the configuration cell:
   ```python
   USE_REAL_DEEPORE = True
   DEEPORE_NPZ = "/content/drive/MyDrive/deepore_96.npz"
   ```
4. Run all cells.

Every expensive stage is checkpointed to Google Drive, so if the session
disconnects you can simply re-run and finished stages are loaded from cache.
For a dataset-free demonstration set `USE_REAL_DEEPORE = False`, which
generates synthetic porous volumes and exercises the whole pipeline on CPU.

## 5. Experimental configuration

The configuration used for all reported results is defined in the `CFG`
dataclass in the notebook:

| Parameter | Value |
|---|---|
| Volumes | 4,000 (DeePore), 64³ resolution |
| Targets | log-permeability, porosity |
| Split | 60 % train / 20 % calibration / 20 % test |
| Backbone | 3 conv blocks, base width 24, ≈1.6 M parameters |
| Dropout | 0.20 (retained at inference for MC dropout) |
| Epochs | 40 (fixed budget; no early stopping) |
| Batch size | 16 |
| Optimiser | AdamW, lr 2 × 10⁻³, weight decay 1 × 10⁻⁴, cosine annealing |
| Gradient clipping | 1.0 |
| Loss | ℒ_MSE + 0.5·ℒ_NLL + ℒ_pinball(0.05) + ℒ_pinball(0.95) |
| Target level | α = 0.10 (90 % intervals) |
| Normalized-conformal ε | 1 × 10⁻³ |
| Mondrian strata | 5 equal-frequency bins of the predicted value |
| MC-dropout passes | T = 20 |
| Random seed | 0 |

No early stopping is used: the epoch budget is fixed in advance so the
predictor is frozen independently of the calibration split, which is required
for the conformal guarantee.

## 6. Sensitivity analyses

Three analyses examine the robustness of the calibration stage. All reuse the
frozen predictor and therefore isolate the behaviour of the conformal step.

- **Stabilising constant ε** (`sens_epsilon.csv`, Fig. 8a). Varying ε from
  10⁻⁶ to 10⁻¹ changes coverage by less than 0.002 for log-permeability, so the
  adopted value is not a tuned parameter.
- **Calibration-set size** (`sens_calsize.csv`, Fig. 8b). Coverage stays within
  0.01 of nominal for calibration fractions from 5 % to 40 %, with the
  across-split standard deviation contracting as the calibration set grows
  (20 random splits per setting).
- **Number of Mondrian strata** (`sens_mondrian.csv`, Fig. 8c). Marginal and
  worst-slice coverage are stable from 2 to 15 groups, with worst-slice
  coverage maximised at a moderate group count.

## 7. Data

The experiments use the **DeePore** dataset (Rabbani et al., 2020),
available from Zenodo. See `data/README.md` for the download link and the
preprocessing the notebook expects. Data files are not redistributed here.

## 8. Reproducibility notes

- All random number generators are seeded (seed 0).
- Targets are standardized using training-split statistics only.
- Conformal calibration is performed exclusively on the held-out calibration
  split, disjoint from both training and test data.
- Volumes are assigned to splits by uniform random permutation. For spatially
  clustered field data, grouped or block splitting at the level of the core or
  formation would be required to preserve exchangeability.

## 9. License

Released under the MIT License (see `LICENSE`).

## 10. Citation

See `CITATION.cff`, or cite the associated manuscript in *Results in
Engineering*.
