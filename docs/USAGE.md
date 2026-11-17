# Usage guide

## 1. Installation

```bash
git clone https://github.com/[USERNAME]/conformal-digital-rock.git
cd conformal-digital-rock
pip install -r requirements.txt
```

Python 3.9 or newer. All dependencies are open-source; no proprietary software
is required.

## 2. Fast path: regenerate the figures (no GPU, no dataset)

The result tables in `results/` are the ones reported in the manuscript, so the
sensitivity studies and all figures can be reproduced immediately:

```bash
cd src
python sensitivity_analyses.py     # -> ../results/sens_epsilon.csv, sens_calsize.csv, sens_mondrian.csv
python make_figures.py             # -> ../figures/Fig3..Fig8 (.png and .pdf)
```

Both scripts run on CPU in seconds. This verifies the installation and confirms
that the published figures follow from the tabulated results.

## 3. Full path: reproduce the experiment end to end

### 3.1 Obtain the data
Download DeePore (see `data/README.md`) and place the preprocessed archive at
`data/deepore_96.npz`, or upload it to Google Drive.

### 3.2 Run the notebook
Open `notebooks/Conformal_DigitalRock.ipynb` in Google Colab, select a GPU
runtime, and set:

```python
USE_REAL_DEEPORE = True
DEEPORE_NPZ = "/content/drive/MyDrive/deepore_96.npz"
```

Then run all cells. The notebook proceeds through eleven documented stages:

1. Environment and configuration (`CFG` dataclass)
2. Drive checkpointing
3. Data loading, with layout and convention auto-detection
4. Data cleaning
5. Train / calibration / test split and standardization
6. Heteroscedastic 3D CNN with mean, log-variance, and quantile heads
7. Training
8. Prediction extraction (including MC-dropout passes)
9. Conformal calibration: split, normalized, CQR, Mondrian
10. Distribution-shift study and coverage-vs-alpha sweep
11. Figure and table export

Runtime is roughly 20–35 minutes on an L4 GPU for 4,000 volumes.

### 3.3 Disconnect-proof checkpointing
Processed data, trained weights, and extracted predictions are cached to Google
Drive under `conformal_ckpt/`. On reconnect, re-running loads completed stages
from cache rather than recomputing them. Checkpoints record the configuration
that produced them and are ignored automatically if the configuration changes,
so altering the target properties or resolution starts a clean run rather than
failing. To force a fresh run, change `RUN_NAME`.

### 3.4 Demonstration without the dataset
Set `USE_REAL_DEEPORE = False` and reduce `cfg.n_samples` and `cfg.epochs`. The
notebook generates synthetic porous volumes and exercises the full pipeline on
CPU in a few minutes. Quantitative results in the manuscript use the real data.

## 4. Outputs

| File | Content |
|---|---|
| `results/table_main_coverage.csv` | coverage, worst-slice coverage, interval width per method and property |
| `results/table_ood.csv` | in-distribution versus shifted-regime coverage and width |
| `results/table_sweep.csv` | coverage as a function of target level |
| `results/table_summary.csv` | summary averaged over properties |
| `results/sens_*.csv` | the three sensitivity analyses |
| `figures/Fig3`–`Fig8` | publication figures (300 dpi PNG and vector PDF) |

## 5. Figure conventions

All figures are produced at 300 dpi with vector counterparts. Every series is
encoded redundantly by colour together with marker shape and line style (line
plots) or fill shade and hatch pattern (bar plots), so that the figures remain
readable in monochrome printing. Legends are placed outside the plotting area
where they would otherwise overlap the data.

## 6. Citation

```bibtex
@article{alam_khan_conformal_digital_rock,
  title   = {Distribution-Free Calibrated Uncertainty for Pore-Scale
             Petrophysical Property Prediction via Conformal Prediction},
  author  = {Alam, Gulzar and Khan, Sikandar},
  journal = {Results in Engineering},
  year    = {2026}
}
```
