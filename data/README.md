# Data directory

The experiments use the **DeePore** dataset (Rabbani et al., 2020),
available at https://doi.org/10.5281/zenodo.3820900.

Place the preprocessed cache here as `deepore_96.npz` (or point
`DEEPORE_NPZ` in the notebook to its location). The cache should contain the
µCT volumes and the property label arrays (`log_permeability`, `porosity`).

Data files are gitignored and are not distributed with this repository; please
download DeePore from the link above and preprocess it as described in the
notebook (Section 2).
