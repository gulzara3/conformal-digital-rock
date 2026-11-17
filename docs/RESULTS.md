# Map from paper to released files

Every table and figure of the paper is a file in [`results/`](../results). This map gives the file behind each number. All tables were written by the notebook from the same saved predictions in one run, so they cannot disagree with each other.

## Main paper

| Paper | File | Content |
|---|---|---|
| Fig. 1 | [`paper/Fig01_workflow.png`](../paper/Fig01_workflow.png) | Workflow schematic |
| Table 1 | [`T31_method_requirements.csv`](../results/tables/T31_method_requirements.csv) | What each method needs and guarantees |
| Table 2 | [`T05_point_accuracy_by_property.csv`](../results/tables/T05_point_accuracy_by_property.csv) | MAE, RMSE, R² of the network and two baselines |
| Fig. 2 | [`F04_parity.png`](../results/figures/F04_parity.png) | Parity plots |
| Table 3 | [`T07_summary_across_properties.csv`](../results/tables/T07_summary_across_properties.csv) | Coverage errors, worst quintiles, relative width |
| Fig. 3 | [`F06_coverage_by_seed.png`](../results/figures/F06_coverage_by_seed.png) | Coverage per seed |
| Fig. 4 | [`F07_interval_width.png`](../results/figures/F07_interval_width.png) | Interval widths |
| Fig. 5 | [`F09_validity_and_calibration_size.png`](../results/figures/F09_validity_and_calibration_size.png) | Exact-law test and calibration size |
| Fig. 6 | [`F08_conditional_coverage.png`](../results/figures/F08_conditional_coverage.png) | Coverage within quintiles |
| Fig. 7 | [`F11_heuristics_and_ablations.png`](../results/figures/F11_heuristics_and_ablations.png) | Heuristic baselines |
| Table 4 | [`T23_joint_coverage.csv`](../results/tables/T23_joint_coverage.csv) | Joint coverage |
| Fig. 8 | [`F12_joint_coverage.png`](../results/figures/F12_joint_coverage.png) | Joint coverage |
| Table 5 | [`T24_shift_coverage_log_k.csv`](../results/tables/T24_shift_coverage_log_k.csv) | Coverage under shift, permeability |
| Fig. 9 | [`F13_distribution_shift.png`](../results/figures/F13_distribution_shift.png) | Coverage under shift |
| Table 6 | [`T25_ood_guard.csv`](../results/tables/T25_ood_guard.csv) | Conformal guard |
| Fig. 10 | [`F14_ood_guard.png`](../results/figures/F14_ood_guard.png) | Conformal guard |
| Fig. 11 | [`F15_weighted_conformal.png`](../results/figures/F15_weighted_conformal.png) | Weighted conformal prediction |
| Table 7 | [`T27_triage_in_distribution.csv`](../results/tables/T27_triage_in_distribution.csv) | Triage, tolerance factor 1.25 rows |
| Table 8 | [`T29_triage_under_shift.csv`](../results/tables/T29_triage_under_shift.csv) | Triage under shift, 5% risk, Gaussian score |
| Fig. 12 | [`F16_risk_controlled_triage.png`](../results/figures/F16_risk_controlled_triage.png) | Risk-controlled triage |
| Table 9 | [`T30_measured_cost.csv`](../results/tables/T30_measured_cost.csv) | Measured cost on one NVIDIA L4 |

## Supplementary material

| Paper | File |
|---|---|
| Fig. S1 | [`F01_data_overview.png`](../results/figures/F01_data_overview.png) |
| Fig. S2 | [`F02_provenance_and_shift_folds.png`](../results/figures/F02_provenance_and_shift_folds.png) |
| Fig. S3 | [`F03_input_representation.png`](../results/figures/F03_input_representation.png) |
| Fig. S4 | [`F05_reliability.png`](../results/figures/F05_reliability.png) |
| Fig. S5 | [`F10_sensitivity.png`](../results/figures/F10_sensitivity.png) |
| Table S1 | [`T01_exclusions.csv`](../results/tables/T01_exclusions.csv) |
| Table S2 | [`T02_target_and_descriptor_ranges.csv`](../results/tables/T02_target_and_descriptor_ranges.csv) |
| Table S3 | [`T04_run_plan_and_sample_counts.csv`](../results/tables/T04_run_plan_and_sample_counts.csv) |
| Table S4 | [`T03_morphology_families.csv`](../results/tables/T03_morphology_families.csv) |
| Table S5 | [`T12_connectivity_and_porosity_checks.csv`](../results/tables/T12_connectivity_and_porosity_checks.csv) |
| Table S6 | [`T13_input_representation_ablation.csv`](../results/tables/T13_input_representation_ablation.csv) |
| Table S7 | [`T06_coverage_and_width_by_property.csv`](../results/tables/T06_coverage_and_width_by_property.csv) |
| Table S8 | [`T08_paired_differences_vs_split.csv`](../results/tables/T08_paired_differences_vs_split.csv) |
| Table S9 | [`T09_conditional_coverage_per_bin.csv`](../results/tables/T09_conditional_coverage_per_bin.csv) |
| Table S10 | [`T11_permeability_interval_in_physical_terms.csv`](../results/tables/T11_permeability_interval_in_physical_terms.csv) |
| Table S11 | [`T10_cqr_and_mondrian_bookkeeping.csv`](../results/tables/T10_cqr_and_mondrian_bookkeeping.csv) |
| Table S12 | [`T14_calibration_size_sample_counts.csv`](../results/tables/T14_calibration_size_sample_counts.csv) |
| Table S13 | [`T15_coverage_distribution_vs_exact_law.csv`](../results/tables/T15_coverage_distribution_vs_exact_law.csv) |
| Table S14 | [`T16_epsilon_sensitivity.csv`](../results/tables/T16_epsilon_sensitivity.csv) |
| Table S15 | [`T17_mondrian_group_count.csv`](../results/tables/T17_mondrian_group_count.csv) |
| Table S16 | [`T18_mondrian_grouping_rule.csv`](../results/tables/T18_mondrian_grouping_rule.csv) |
| Table S17 | [`T19_cqr_quantile_pair.csv`](../results/tables/T19_cqr_quantile_pair.csv) |
| Table S18 | [`T20_mc_dropout_convergence.csv`](../results/tables/T20_mc_dropout_convergence.csv) |
| Table S19 | [`T21_heuristics_and_their_conformalized_versions.csv`](../results/tables/T21_heuristics_and_their_conformalized_versions.csv) |
| Table S20 | [`T22_training_ablation.csv`](../results/tables/T22_training_ablation.csv) |
| Table S21 | [`T24b_shift_coverage_all.csv`](../results/tables/T24b_shift_coverage_all.csv) (rows with `part = shifted`) |
| Table S22 | [`T26_weighted_conformal_tilt.csv`](../results/tables/T26_weighted_conformal_tilt.csv) |
| Table S23 | [`T26b_weighted_conformal_family_holdout.csv`](../results/tables/T26b_weighted_conformal_family_holdout.csv) |
| Table S24 | [`T27_triage_in_distribution.csv`](../results/tables/T27_triage_in_distribution.csv) |
| Table S25 | [`T27b_triage_crc.csv`](../results/tables/T27b_triage_crc.csv) |
| Table S26 | [`T28_triage_guarantee_check.csv`](../results/tables/T28_triage_guarantee_check.csv) |
| Table S27 | [`T29_triage_under_shift.csv`](../results/tables/T29_triage_under_shift.csv) |
| Table S28 | Layer table of [`src/conformal_rock/model.py`](../src/conformal_rock/model.py), checked by `tests/test_model_data.py` |
| Table S29 | [`results/logs/config.json`](../results/logs/config.json) and [`environment.json`](../results/logs/environment.json) |

Other files: [`T32_reviewer_comment_map.csv`](../results/tables/T32_reviewer_comment_map.csv) maps review comments to evidence, and [`key_numbers.json`](../results/tables/key_numbers.json) collects the numbers quoted in the text.

## Units and conventions

* Widths of log-transformed targets are in decades (base 10). Other widths are in the units of the target.
* Coverage is a fraction. "pp" means percentage points.
* Unless stated otherwise, values are means over ten seeds at a 90% target.
* Property codes: `log_k` permeability, `log_F` formation factor, `tau` tortuosity, `log_ssa` specific surface, `log_rp` mean pore radius, `E_rel` relative Young's modulus.
* Method codes: `split`, `norm` (normalized), `cqr`, `mondrian`, `gauss` (Gaussian head), `mcd` (MC-dropout, spread of means), `mcd_tot` (MC-dropout, total variance), `ens` (deep ensemble), `split_gbm` and `split_kc` (split conformal on the descriptor baselines).
