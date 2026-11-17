# Notebook

`conformal_digital_rock.ipynb` is the notebook that produced every result of release v2.0.0 (run on Google
Colab with an NVIDIA L4 GPU). It is self-contained: it downloads the DeePore archive, verifies it, builds the
dataset, trains all 30 networks, runs every analysis and writes the tables, figures, logs and release archive.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gulzara3/conformal-digital-rock/blob/main/notebooks/conformal_digital_rock.ipynb)

1. Runtime → Change runtime type → L4 GPU.
2. Run all. The first cell mounts Google Drive as a cache, so a disconnected session resumes where it stopped.
3. Outputs are written to `MyDrive/conformal_rock_v2/paper/` (tables, figures, splits, predictions, checkpoints,
   logs and `release_paper.zip`).

Sections: 1 to 3 setup and configuration · 4 data · 5 splits and run plan · 6 network and training ·
7 conformal library · 8 input representation · 9 benchmark · 10 validity and calibration size · 11 heuristics ·
12 joint coverage · 13 distribution shift, guard and weighted conformal · 14 triage · 15 cost and release ·
16 your own volumes.
