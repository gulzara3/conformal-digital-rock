# Usage guide

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/[USERNAME]/conformal-digital-rock.git
cd conformal-digital-rock
pip install -r requirements.txt
```

Python 3.9 or newer is required. All dependencies are open-source.

## Running on Google Colab (recommended)

1. Upload `notebooks/Conformal_DigitalRock.ipynb` to Google Colab, or open it
   directly from GitHub (File → Open notebook → GitHub).
2. Select a GPU runtime (Runtime → Change runtime type → GPU; an L4 is ideal).
3. In the data configuration cell, set:
   ```python
   USE_REAL_DEEPORE = True
   DEEPORE_NPZ = "/content/drive/MyDrive/deepore_96.npz"
   ```
4. Run all cells (Runtime → Run all).

The pipeline trains the predictor, calibrates the four conformal methods,
runs the out-of-distribution study and the coverage sweep, and writes all
tables and figures. Total runtime is roughly 20–35 minutes on an L4 GPU for
4,000 volumes.

## Disconnect-proof checkpointing

Every expensive stage (processed data, trained models, extracted predictions)
is cached to Google Drive under `conformal_ckpt/`. If the Colab session
disconnects, reconnect and re-run: completed stages are loaded from cache
instead of being recomputed. To force a fresh run, change `RUN_NAME` in the
checkpoint cell.

## Quick demonstration (no dataset)

For a fast end-to-end smoke test without downloading DeePore, set
`USE_REAL_DEEPORE = False` and reduce `cfg.n_samples` (e.g. 60) and
`cfg.epochs` (e.g. 2). The notebook generates synthetic porous volumes and runs
the full pipeline in a few minutes on CPU. Note that quantitative results in
the paper use the real DeePore data.

## Reproducing paper results

With `USE_REAL_DEEPORE = True` and the default configuration, running all cells
reproduces the coverage tables, the OOD study, the reliability sweep, and the
seven figures reported in the paper. Result CSVs and figures are written to the
output directory and mirrored to Google Drive.

## Configuration reference

All hyperparameters are defined in the `CFG` dataclass in Section 1 of the
notebook. See the table in the main `README.md` for the key parameters.
