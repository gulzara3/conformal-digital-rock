# Data

The experiments use the **DeePore** dataset of micro-computed-tomography
porous-media volumes with simulation-derived property labels
(Rabbani et al., 2020, *Advances in Water Resources* 146, 103787).

Download: https://zenodo.org/records/4297035

## Expected format

The notebook loads a preprocessed NumPy archive containing the volumes and the
property label arrays:

```
deepore_96.npz
├── volumes            uint8, shape (N, 1, 96, 96, 96)   binary occupancy
├── porosity           float32, shape (N,)
├── log_permeability   float32, shape (N,)
└── (further property arrays, unused here)
```

The loader is tolerant of layout variation: it squeezes a singleton channel
axis, auto-detects the array key, downsamples 96³ to the working resolution of
64³ in memory-safe chunks, and infers the solid/pore convention from the
porosity labels.

Place the file here as `data/deepore_96.npz`, or point `DEEPORE_NPZ` in the
notebook at its location (for example a Google Drive path when running in
Colab).

## Data cleaning

Two label corrections are applied before training, and are implemented in the
notebook (Section 2b):

1. `formation_factor` is stored as a raw ratio spanning many orders of
   magnitude and is not used as a target in this study.
2. `log_permeability` contains an epsilon-floor artifact at the lower bound;
   affected samples are removed so that calibration is performed on valid data
   only.

Data files are gitignored and are not redistributed with this repository.
