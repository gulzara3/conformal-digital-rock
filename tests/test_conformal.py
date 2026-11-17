"""Statistical and unit tests of the conformal predictors."""
import numpy as np
import pytest

from conformal_rock import conformal as C


def gaussian_case(n_cal=400, n_te=800, seed=0, hetero=False):
    rng = np.random.default_rng(seed)
    n = n_cal + n_te
    x = rng.uniform(0.2, 1.0, n)
    sig = x if hetero else np.ones(n)
    y = rng.standard_normal(n) * sig
    q = np.stack([-1.645 * sig, 1.645 * sig], 1)[:, None, :]          # one property, one quantile pair
    d = dict(y=y[:, None], mu=np.zeros((n, 1)), sig=sig[:, None], q=q)
    cal = {k: v[:n_cal] for k, v in d.items()}
    te = {k: v[n_cal:] for k, v in d.items()}
    return cal, te


def test_quantile_rule():
    s = np.arange(1, 11, dtype=float)                 # n = 10
    assert C.conformal_quantile(s, 0.1) == 10.0       # k = ceil(11 * 0.9) = 10
    assert C.conformal_quantile(s, 0.2) == 9.0        # k = ceil(11 * 0.8) = 9
    assert np.isinf(C.conformal_quantile(s, 0.05))    # k = 11 > n
    assert np.isinf(C.conformal_quantile([], 0.1))


def test_quantile_pair_matches_target():
    assert C.quantile_pair(0.10) == (2, 7)            # 0.05 / 0.95 heads
    assert C.quantile_pair(0.02) == (0, 5)            # 0.01 / 0.99 heads
    assert C.quantile_pair(0.30) == (4, 9)            # 0.15 / 0.85 heads


def test_split_coverage_follows_exact_law():
    covs = []
    for s in range(300):
        cal, te = gaussian_case(seed=s)
        lo, hi, _ = C.interval("split", cal, te, 0, 0.1)
        covs.append(C.coverage(te["y"][:, 0], lo, hi))
    m, sd = C.coverage_law(400, 800, 0.1)
    assert abs(np.mean(covs) - m) < 4 * sd / np.sqrt(300)
    assert abs(np.std(covs) / sd - 1) < 0.2


@pytest.mark.parametrize("method", ["split", "norm", "mondrian"])
def test_marginal_coverage_heteroscedastic(method):
    covs = []
    for s in range(200):
        cal, te = gaussian_case(seed=100 + s, hetero=True)
        tr, _ = gaussian_case(seed=10_000 + s, hetero=True)
        lo, hi, _ = C.interval(method, cal, te, 0, 0.1, tr=tr, n_groups=3)
        covs.append(C.coverage(te["y"][:, 0], lo, hi))
    assert 0.885 < np.mean(covs) < 0.915


def test_cqr_handles_crossed_and_inverted_bands():
    cal, te = gaussian_case(seed=1)
    te = dict(te)
    te["q"] = te["q"][:, :, ::-1].copy()              # cross every test band
    import conformal_rock.config as cfg
    old = cfg.CFG.q_lo
    cfg.CFG.q_lo = [0.05]
    try:
        lo, hi, info = C.interval("cqr", cal, te, 0, 0.1)
    finally:
        cfg.CFG.q_lo = old
    assert np.all(lo <= hi)
    assert info["n_crossed"] == len(lo)


def test_beta_binomial_pmf_and_ks():
    pmf = C.coverage_law_pmf(50, 200, 0.1)
    assert abs(pmf.sum() - 1) < 1e-9
    a, b = C.coverage_law_params(50, 0.1)
    from scipy import stats
    h = stats.betabinom.rvs(200, a, b, size=300, random_state=np.random.default_rng(0))
    _, p = C.discrete_ks_test(h, 50, 200, 0.1, n_mc=300)
    assert p > 0.01
    _, p_bad = C.discrete_ks_test(np.minimum(h + 8, 200), 50, 200, 0.1, n_mc=300)
    assert p_bad < 0.01                                # an off-by-a-few shift is detected


def test_max_score_box_gives_joint_coverage():
    rng = np.random.default_rng(3)
    n = 6000
    e = rng.multivariate_normal([0, 0, 0], [[1, .5, .2], [.5, 1, .3], [.2, .3, 1]], n)
    d = dict(y=e, mu=np.zeros((n, 3)))
    cal = {k: v[:3000] for k, v in d.items()}
    te = {k: v[3000:] for k, v in d.items()}
    lo, hi, _ = C.joint_box(cal, te, [0, 1, 2], 0.1, scale_cal=np.ones(3))
    assert abs(C.joint_coverage(te["y"], lo, hi) - 0.9) < 0.02
    hw = np.array([C.cq(np.abs(cal["y"][:, j]), 0.1) for j in range(3)])
    assert C.joint_coverage(te["y"], -hw, hw) < 0.85    # independent intervals under-cover jointly
