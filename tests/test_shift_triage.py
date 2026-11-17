"""Tests of the out-of-distribution guard, weighted conformal prediction and triage thresholds."""
import numpy as np

from conformal_rock import shift as S
from conformal_rock import triage as T


def test_conformal_pvalue_false_alarm_rate():
    rng = np.random.default_rng(0)
    rates = []
    for _ in range(200):
        s = rng.standard_normal(1001)
        p = S.conformal_pvalue(s[:1000], s[1000:])
        rates.append(p[0] <= 0.05)
    assert np.mean(rates) < 0.09
    p = S.conformal_pvalue(np.arange(10.0), np.array([-1.0, 4.5, 100.0]))
    assert np.allclose(p, [11 / 11, 6 / 11, 1 / 11])


def test_weighted_threshold_reduces_to_split_with_equal_weights():
    from conformal_rock.conformal import conformal_quantile
    s = np.random.default_rng(1).random(99)
    q = S.weighted_threshold(s, np.ones(99), np.ones(3), 0.1)
    assert np.allclose(q, conformal_quantile(s, 0.1))


def test_density_ratio_and_ess():
    rng = np.random.default_rng(2)
    xc = rng.normal(0, 1, (2000, 1))
    xu = rng.normal(0.5, 1, (2000, 1))
    wc, we = S.density_ratio(xc, xu, np.array([[-2.0], [2.0]]))
    assert we[1] > 1 > we[0]
    assert 0 < S.effective_sample_size(wc) <= len(wc)


def test_ltt_bounds_bad_accept_rate():
    rng = np.random.default_rng(3)
    viol = []
    for _ in range(100):
        u = rng.random(4000)
        bad = rng.random(4000) < 0.3 * u              # error probability grows with the score
        grid = T.threshold_grid(rng.random(500))
        t = T.ltt_threshold(u[:2000], bad[:2000], 0.05, 0.1, grid)
        viol.append(T.evaluate(u[2000:], bad[2000:], t)["bad_accept"] > 0.05 + 0.01)
    assert np.mean(viol) <= 0.15


def test_crc_and_evaluate():
    u = np.linspace(0, 1, 101)
    bad = u > 0.8
    t = T.crc_threshold(u, bad, 0.05)
    r = T.evaluate(u, bad, t)
    assert r["bad_accept"] <= 0.05 and r["accepted"] > 0.8
    assert T.bad_accept_labels([0.0, 0.2], [0.0, 0.0], 1.25).tolist() == [False, True]
