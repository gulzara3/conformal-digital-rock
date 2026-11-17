# Changelog

## v2.0.0 (17 November 2026)

Complete rebuild for the revised and extended manuscript (resubmission of AIIG-D-26-00274).

### Data
- Dataset rebuilt from the public DeePore archive (Zenodo 4297035) with an MD5 check, by a released script
  (`scripts/prepare_deepore.py`). The private `deepore_96.npz` file of v1.0 is no longer used.
- All 14,174 volumes that pass DeePore's outlier rules are used, with exclusion counts per rule.
- Two-channel 64³ input without cropping or re-binarisation; porosity conserved exactly.
- Six simulated targets; porosity removed as a target because it is an exact voxel count.

### Experiments
- Ten independent training seeds and splits; 30 networks in total.
- Mondrian groups from training-set predictions (exact group-conditional guarantee).
- Ten quantile heads; CQR uses the pair matched to each target level.
- Exact beta-binomial test of the implementation; calibration-size study with disjoint draws.
- Joint coverage: Bonferroni and max-score conformal box.
- Distribution shift: label extrapolation, morphology-family holdout, index blocks; label-free conformal
  guard; weighted conformal prediction.
- Learn-then-Test and conformal-risk-control simulation triage.
- Baselines: MC-dropout in two forms up to 100 passes, deep ensemble, dedicated Gaussian network,
  mean-only ablation, conformalized heuristics.

### Repository
- Installable package `conformal_rock` with tests (statistical tests of the guarantees, model, data, release).
- Scripts to build data, train, fetch and verify the release, recompute tables and apply a network to new volumes.
- SHA-256 manifest of every released file; CI checks it on every push.

### Removed and corrected
- `src/sensitivity_analyses.py` and `results/sens_*.csv` of v1.0 are removed. That analysis ran on synthetic
  predictions, not on the trained network. All sensitivity analyses now use saved predictions of trained networks.
- The coverage-error summary now averages absolute errors per property and seed.

## v1.0.0 (September 2026)

Code accompanying the first submission (two targets, 4,000 volumes, one seed). Superseded by v2.0.0.
