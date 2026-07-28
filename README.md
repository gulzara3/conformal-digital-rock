# Distribution-Free Calibrated Uncertainty for Pore-Scale Property Prediction via Conformal Prediction

This repository contains the reference implementation and experiment code for the paper:

> **Distribution-Free Calibrated Uncertainty for Pore-Scale Petrophysical Property Prediction via Conformal Prediction**

It trains a three-dimensional convolutional neural network (3D CNN) to predict
petrophysical properties (log-permeability and porosity) directly from
micro-computed-tomography (µCT) porous-media volumes, and wraps the predictor
with **conformal prediction** to produce prediction intervals that carry a
finite-sample, distribution-free coverage guarantee. The code reproduces every
result, table, and figure reported in the paper, including the four conformal
variants, the two heuristic baselines, the coverage-versus-α reliability sweep,
and the out-of-distribution (OOD) study.

The code is released to support reproducibility and reuse. It is provided as a
self-contained, well-documented Jupyter/Colab notebook so that reviewers and
readers can run the full pipeline end to end.

---

## 1. What the code does

The notebook (`notebooks/Conformal_DigitalRock.ipynb`) implements the complete
pipeline in eleven documented sections:

| Section | Content |
|---|---|
| 1. Environment & configuration | Reproducible seeds and a single `CFG` dataclass holding all hyperparameters. |
| 1b. Drive checkpointing | Disconnect-proof caching of data, models, and predictions to Google Drive. |
| 2. Data | Loads the DeePore dataset (with a synthetic fallback), computes targets, and auto-detects array layout and solid/pore convention. |
| 2b. Data cleaning | Log-transforms and removes label artifacts so calibration is performed on valid data. |
| 3. Splits & standardization | 60/20/20 train/calibration/test partition; conformal calibration uses the dedicated calibration split. |
| 4. Model | Heteroscedastic 3D CNN with mean, log-variance, and quantile heads (shared backbone; MC-dropout enabled). |
| 5. Training | Composite loss (MSE + Gaussian NLL + pinball) with AdamW and cosine-annealed learning rate. |
| 6. Prediction extraction | Collects mean, variance, quantile, and MC-dropout predictions on the calibration and test splits. |
| 7. Conformal methods | Split, normalized (adaptive), CQR, and Mondrian (group-conditional) conformal prediction, plus Gaussian and MC-dropout baselines. |
| 8. OOD study | Holds out the upper permeability regime and quantifies coverage degradation under distribution shift. |
| 9. Coverage-vs-α sweep | Reliability curves across target coverage levels. |
| 10. Figures | Publication-quality figures (PNG + SVG). |
| 11. Export | Writes all result tables (CSV) and figures. |

## 2. Requirements

- Python 3.9+
- PyTorch, NumPy, SciPy, scikit-learn, scikit-image, pandas, matplotlib, h5py

The notebook is designed to run on **Google Colab (GPU runtime, e.g. L4)** with
no local setup. To run locally, install the dependencies with:

```bash
pip install -r requirements.txt
```

## 3. Data

The experiments use the **DeePore** dataset (Rabbani et al., 2020):
https://doi.org/10.5281/zenodo.3820900

The notebook expects a preprocessed `.npz` cache containing the µCT volumes and
their property labels (`log_permeability`, `porosity`). The loader
auto-detects the array key, the voxel resolution (downsampling 96³ → 64³ if
required), and the solid/pore convention. If no dataset is found, the notebook
falls back to a synthetic porous-media generator so the pipeline still runs
end to end for demonstration; **quantitative results in the paper use the real
DeePore data.**

Set the data path in the configuration cell:

```python
USE_REAL_DEEPORE = True
DEEPORE_NPZ = "/path/to/deepore_96.npz"
```

## 4. How to run

**On Google Colab (recommended):**
1. Open `notebooks/Conformal_DigitalRock.ipynb` in Colab.
2. Set the runtime to a GPU (Runtime → Change runtime type → GPU).
3. Point `DEEPORE_NPZ` at your preprocessed DeePore cache (or leave
   `USE_REAL_DEEPORE = False` for a synthetic demonstration run).
4. Run all cells. The full pipeline (training, calibration, OOD study, sweep,
   figures) completes in roughly 20–35 minutes on an L4 GPU for 4,000 volumes.

All intermediate artifacts are checkpointed to Google Drive, so if the session
disconnects you can simply re-run: finished stages are loaded from cache rather
than recomputed.

**Quick demonstration run (no dataset, CPU-friendly):** set
`USE_REAL_DEEPORE = False` and reduce `cfg.n_samples` and `cfg.epochs` to obtain
a fast end-to-end smoke test on synthetic volumes.

## 5. Outputs

Running the notebook produces, in the output directory:

- `table_main_coverage.csv` — per-method, per-property coverage, worst-slice
  coverage, and interval width at the target level.
- `table_ood.csv` — in-distribution vs. out-of-distribution coverage and width.
- `table_sweep.csv` — coverage-versus-α reliability data.
- `table_summary.csv` — summary comparison averaged over properties.
- `fig1`–`fig7` — publication figures (PNG and SVG).

## 6. Configuration

All settings live in the `CFG` dataclass (Section 1). Key parameters:

| Parameter | Default | Meaning |
|---|---|---|
| `voxel` | 64 | working resolution (96³ inputs are auto-downsampled) |
| `n_samples` | 4000 | number of volumes used |
| `properties` | (`log_k`, `porosity`) | prediction targets |
| `frac_train/calib/test` | 0.60/0.20/0.20 | data partition |
| `epochs` | 40 | training epochs |
| `alpha` | 0.10 | target miscoverage (90% intervals) |
| `ood_hold_top_frac` | 0.25 | fraction of the top permeability regime held out as OOD |

## 7. Citation

If you use this code, please cite the associated paper (see `CITATION.cff`).

## 8. License

Released under the MIT License (see `LICENSE`).
