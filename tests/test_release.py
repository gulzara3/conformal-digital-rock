"""The released files match the manifest, and the tables carry the numbers quoted in the paper."""
import pandas as pd

from conformal_rock.io import RESULTS, load_manifest, sha256


def test_released_results_match_manifest():
    checked = 0
    for e in load_manifest()["files"]:
        top, rest = e["file"].split("/", 1)
        if top == "preds":
            continue
        p = RESULTS / top / rest
        assert p.exists(), e["file"]
        assert sha256(p) == e["sha256"], e["file"]
        checked += 1
    assert checked > 100


def test_headline_numbers():
    t07 = pd.read_csv(RESULTS / "tables" / "T07_summary_across_properties.csv").set_index("method")
    assert round(t07.loc["split", "mean_abs_cov_error_pp"], 2) == 0.58
    assert round(t07.loc["mcd", "mean_abs_cov_error_pp"], 2) == 20.14
    t23 = pd.read_csv(RESULTS / "tables" / "T23_joint_coverage.csv")
    indep = t23[(t23.set == "all 6") & (t23.method == "Independent 90% intervals")].joint_coverage.item()
    assert round(indep, 3) == 0.637
    t27 = pd.read_csv(RESULTS / "tables" / "T27_triage_in_distribution.csv")
    row = t27[(t27.tol_factor == 1.25) & (t27.eps == 0.01) & (t27.score == "Gaussian-head sd")]
    assert round(row.simulations_avoided.item(), 2) == 0.56
