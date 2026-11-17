# Review comments and where the evidence is

The first version of this repository (v1.0) accompanied manuscript AIIG-D-26-00274. The reviewers found that it could not reproduce the paper. This page lists each point that concerns code, data or results, and where v2.0.0 addresses it. The full point-by-point response is part of the resubmission.

## Reviewer 1: reproducibility

> "It contains no volumes, weights, or predictions, and no script that builds the required deepore_96.npz from the public Zenodo archive; the notebook points to a private Drive file."

| Missing in v1.0 | Now |
|---|---|
| Build script from the public archive | [`scripts/prepare_deepore.py`](../scripts/prepare_deepore.py) and Section 4 of the notebook download `DeePore_Dataset.h5` from Zenodo, check its MD5 and build the dataset. `deepore_96.npz` is no longer used. |
| Volumes | The DeePore volumes belong to their authors and are fetched from Zenodo. The exact retained indices are in [`results/data/selected_h5_indices.csv`](../results/data/selected_h5_indices.csv). |
| Weights | All 30 checkpoints are release assets, with checksums in [`checkpoints/SHA256SUMS`](../checkpoints/SHA256SUMS). |
| Predictions | Predictions of all 30 runs for every split are in `release_paper.zip`, fetched by [`scripts/fetch_release.py`](../scripts/fetch_release.py). |
| Private Drive file | None. Drive is used only as an optional cache in Colab. |

## Reviewer 2

| Comment | Evidence in this repository |
|---|---|
| 1. Selection, preprocessing, units, pairing | [`docs/DATA.md`](DATA.md); `T01`, `T02`, `T12`, `T13`; `F01`, `F03` |
| 2. Related samples across splits | Morphology families and index blocks: `T03`, `T24`, `T24b`; `F02`, `F13`; split indices in [`results/splits`](../results/splits) |
| 3. Mondrian partition from the calibration set | `conformal.interval(..., "mondrian")` takes edges from training predictions; `T17`, `T18`; `F10` |
| 4. Joint vs marginal coverage | `conformal.joint_box`, `conformal.bonferroni_box`; `T23`; `F12` |
| 5. Figures, tables and sources disagree | Every file was written from the same saved predictions in one run; [`scripts/reproduce_tables.py`](../scripts/reproduce_tables.py) recomputes the headline tables |
| 6. Calibration size and disjointness | `T14` lists counts and disjointness for every setting; `T15`; `F09` |
| 7. Absolute error after averaging | `metrics.coverage_error_pp`; `T07` |
| 8. One seed; per-bin tables; validity wording | Ten seeds (`results/logs/main_s*.json`); `T06` to `T09`; exact-law test `T15` |
| 9. MC-dropout variance, passes, modes, auxiliary heads | `T20` to `T22`; `F11`; modes in `model.set_mc_dropout` |
| 10. OOD details and framing | `T04` (counts), `T24`, `T24b`, `T25`; `F13`, `F14` |
| 11. Metrics, porosity, units, physical widths | `T05`, `T11`, `T12`; units in [`docs/DATA.md`](DATA.md) |
| 12. Requirements and cost | `T31`, `T30` |
| 13. Equations, quantile convention, Algorithm 1 | `conformal.conformal_quantile` and [`docs/METHODS.md`](METHODS.md) |
| 14. Quantile pairs, crossing, negative adjustments | `conformal.quantile_pair`; `T10`, `T19` |
| 15. Figure omissions | `F05` to `F09` show all four conformal methods |
| 17. Immutable release | Tag `v2.0.0`, SHA-256 manifest [`results/MANIFEST.json`](../results/MANIFEST.json), `scripts/verify_release.py`, and the CI check on every push |

## Changes to results of v1.0

Two problems of the first version are corrected in v2.0.0 and disclosed in the resubmission:

1. The calibration-sensitivity analysis of v1.0 (`src/sensitivity_analyses.py`, `results/sens_*.csv`) ran on synthetic predictions rather than on the trained network. It has been removed. All sensitivity analyses now use the saved predictions of trained networks (`T14` to `T19`).
2. The v1.0 data came from an undocumented 96³ file prepared for another project. The dataset is now rebuilt from the public archive by a released script.

See [`CHANGELOG.md`](../CHANGELOG.md) for the complete list of changes.
