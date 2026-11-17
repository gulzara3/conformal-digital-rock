# Data pipeline

This document describes how the working dataset is built from the public DeePore archive. It answers, item by item, the questions raised in review about selection, preprocessing, units and the pairing of inputs with labels. The code is [`src/conformal_rock/data.py`](../src/conformal_rock/data.py), run by [`scripts/prepare_deepore.py`](../scripts/prepare_deepore.py) and by Section 4 of the notebook.

## Source

| Item | Value |
|---|---|
| File | `DeePore_Dataset.h5` |
| Record | Zenodo [4297035](https://zenodo.org/records/4297035) (Rabbani et al., 2020) |
| MD5 | `d56fbd79f451353af131402c42b179d7` (checked before use) |
| Volumes | 17,700 binary volumes of 256³ voxels, dataset `X`, stored 1 = solid |
| Labels | 1,515 pore-network features per volume, dataset `Y`; the first 15 are single-value properties |

No private or intermediate files are used. The `deepore_96.npz` file of the first submission is not part of this pipeline.

## Selection and exclusions

Every volume is screened with DeePore's own outlier rules, applied in the order below. A volume removed by an earlier rule is not counted again in "newly removed". No subsampling follows: all valid volumes are used.

| Rule | Volumes flagged | Newly removed |
|:---|---:|---:|
| NaN or Inf in any feature | 734 | 734 |
| formation factor > 120 | 2,261 | 2,208 |
| tortuosity > 1.8 | 1,256 | 144 |
| permeability < 1e-4 px² | 114 | 0 |
| cementation factor < 1e-5 | 0 | 0 |
| relative Young's modulus > 0.7 | 498 | 440 |
| non-positive value in a log-transformed target | 0 | 0 |
| **Total excluded** | | **3,526** |
| **Valid volumes used** | | **14,174** |

The retained DeePore indices are in [`results/data/selected_h5_indices.csv`](../results/data/selected_h5_indices.csv) and the counts in [`results/tables/T01_exclusions.csv`](../results/tables/T01_exclusions.csv).

## Targets and units

| Name | DeePore column | Transform | Unit |
|---|---:|---|---|
| `log_k` permeability | 0 | log10 | px² of the 256³ grid |
| `log_F` formation factor | 1 | log10 | dimensionless |
| `tau` tortuosity | 4 | none | dimensionless |
| `log_ssa` specific surface | 10 | log10 | px⁻¹ |
| `log_rp` mean pore radius | 7 | log10 | px |
| `E_rel` relative Young's modulus | 14 | none | dimensionless |

DeePore reports lengths in voxels of the 256³ grid. Physical permeability is `k[px²] × (voxel edge length)²`; for example, with a 3 µm voxel, 1 px² is 9 × 10⁻¹² m², or about 9.1 darcy. The paper reports widths of log-permeability intervals as multiplicative factors (Table S10).

Porosity is **not** a target. For a binary volume it is the pore-voxel fraction, an exact count that the input conserves; it is used only as a grouping variable and as an image descriptor.

## Phase convention

DeePore's documentation states 0 = void and 1 = solid. The pipeline checks this automatically: permeability must rise with pore fraction, so the Spearman correlation between the fraction of stored ones and log permeability must be negative if ones are solid. On 48 probe volumes it was **−0.88**, so ones are treated as solid.

## Input representation

The labels describe the full 256³ geometry, so the input must describe the same field of view. **No cropping and no re-binarisation** are applied. Each volume is coarse-grained by a factor of four into two channels at 64³, both computed at full resolution:

1. **Pore fraction** of each 4³ block (stored as pore-voxel counts 0 to 64). Porosity is conserved exactly.
2. **Local pore size**: the block maximum of a distance-to-solid map from alternating 6- and 26-neighbour erosions (octagonal metric), capped at 24 voxels.

The second channel keeps sub-block pore-size information that plain down-sampling destroys.

## Connectivity check

On 300 random volumes (the CSV holds them all: [`results/data/connectivity.csv`](../results/data/connectivity.csv)):

| Check | Value |
|---|---|
| Pore space spans the sample at 256³ | 100% |
| Still spans after majority re-binarisation to 128³ | 100% |
| Still spans after majority re-binarisation to 64³ | 99.67% (1 of 300 changed) |
| Median absolute porosity change from re-binarisation to 64³ | 0.0024 |
| Maximum absolute porosity change from re-binarisation to 64³ | 0.029 |
| Maximum absolute porosity change in the two-channel input | 0 |

An ablation on 2,000 volumes compares five input forms (two-channel 64³, pore fraction only, re-binarised, 32³ and 128³) on one split (Table S6, [`T13`](../results/tables/T13_input_representation_ablation.csv)).

## Image descriptors

Label-free descriptors are computed on the full volume: porosity, pore-solid surface density, mean, median, 90th-percentile and maximum pore radius from the distance map, and the standard deviation of porosity over an 8³ grid of sub-blocks. They define the morphology families of the shift experiments and serve the descriptor baselines and the density-ratio model.

## Licence of the data

The DeePore dataset belongs to its authors and is distributed through Zenodo under the terms stated on the record. This repository does not redistribute any volume; it stores only indices, derived summary tables and model outputs.
