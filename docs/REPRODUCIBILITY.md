# Reproducibility

The results in the paper come from one run of the notebook on Google Colab, release **v2.0.0** of this repository. This page lists what is released, how to check it, and how to rerun each part.

## Three levels of reproduction

| Level | What it checks | Needs | Time |
|---|---|---|---|
| 1. Verify | Every released table, figure, split and log matches the SHA-256 manifest written by the run | CPU | seconds |
| 2. Recompute | Tables 3, 4, 7, S7 and S24 follow from the released predictions and the package code | CPU, 185 MB download | under a minute |
| 3. Rerun | The full study from the public DeePore archive | GPU (Colab L4), about 13 GB disk | about 3.5 h |

### Level 1

```bash
python scripts/verify_release.py
```

### Level 2

```bash
python scripts/fetch_release.py      # or: --zip path/to/release_paper.zip
python scripts/verify_release.py     # now also checks the 164 files of the archive
python scripts/reproduce_tables.py
```

### Level 3

Open [`notebooks/conformal_digital_rock.ipynb`](../notebooks/conformal_digital_rock.ipynb) in Colab with an L4 GPU and run all cells. The profile is chosen by the environment variable `ROCK_PROFILE`:

| Profile | Data | Seeds | Use |
|---|---|---|---|
| `paper` (default) | all 14,174 valid volumes | 10 | the paper |
| `standard` | 8,000 volumes | 5 | a faster full pass |
| `smoke` | a small **synthetic** stand-in | 2 | code test only; figures are watermarked and must never be reported |

Every expensive stage is cached to Google Drive (`MyDrive/conformal_rock_v2/<profile>/`) so a disconnected session resumes where it stopped. Drive is used only as a cache; no private input file is read. Set `ROCK_H5` to an existing copy of `DeePore_Dataset.h5` to skip the download.

From the command line, the same pipeline is:

```bash
python scripts/prepare_deepore.py --raw data/raw --out data/processed
python scripts/train.py --data data/processed --run main_s0 --out runs/
```

`train.py` reuses the released split of the named run, so its predictions are directly comparable with `release/preds/main_s0.npz`.

## Measured cost (NVIDIA L4, 12 CPU cores, PyTorch 2.14, mixed precision)

| Step | Time |
|---|---|
| Preprocessing of 14,174 volumes (256³ → 2 × 64³), once | 29.7 min |
| Training one network for 24 epochs | 6.7 min |
| Inference, all heads | 0.72 ms per volume |
| MC-dropout, 30 passes | 20.5 ms per volume |
| Conformal calibration and intervals for 2,834 test volumes | 0.35 ms in total |
| Deep ensemble of five networks, training | 33.4 min |
| All 30 training runs | 2.9 h |

## Determinism

Splits, calibration draws, partitions and every random choice in the analysis use fixed seeds, so Levels 1 and 2 reproduce the released numbers exactly. Training on a GPU uses cuDNN and mixed precision, which are not bit-for-bit deterministic across hardware or library versions. A rerun therefore gives statistically equivalent results, not identical predictions. The released predictions are the reference for all reported numbers.

## Software

The Colab run used Python 3.13, PyTorch 2.14.1 (CUDA 13.0), NumPy 2.4, SciPy 1.16, scikit-learn 1.6, pandas 2.2 and h5py 3.14 ([`results/logs/environment.json`](../results/logs/environment.json)). The complete `pip freeze` of that session is [`results/logs/requirements_frozen.txt`](../results/logs/requirements_frozen.txt). The package itself needs only the short list in [`requirements.txt`](../requirements.txt).

## Release assets

| Asset | Size | Where |
|---|---|---|
| `release_paper.zip`: tables, figures, splits, logs, data indices, predictions of all 30 runs, manifest | 191 MB | GitHub release v2.0.0 and Zenodo |
| `checkpoints_v2.0.0.zip`: the 30 trained networks | 509 MB | GitHub release v2.0.0 and Zenodo |

The checksums of both are in [`results/MANIFEST.json`](../results/MANIFEST.json) and [`checkpoints/SHA256SUMS`](../checkpoints/SHA256SUMS).
