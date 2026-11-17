"""Tests of the network and the DeePore build on a tiny synthetic archive."""
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("h5py")


def test_parameter_count_matches_paper():
    from conformal_rock.model import RockNet, n_parameters
    assert n_parameters(RockNet()) == 1_479_560          # Table S28


def test_forward_shapes_and_loss():
    from conformal_rock.config import CFG
    from conformal_rock.model import RockNet, loss_fn, set_mc_dropout
    m = RockNet()
    out = m(torch.zeros(2, 2, 16, 16, 16))
    assert out[0].shape == (2, 6) and out[2].shape == (2, 6, 10) and out[3].shape == (2, 128)
    taus = torch.tensor(CFG.q_levels).view(1, 1, -1)
    for mode in ("joint", "mean_only", "gaussian"):
        assert torch.isfinite(loss_fn(out, torch.zeros(2, 6), mode, taus))
    set_mc_dropout(m, True)
    assert m.drop.training and not m.stem[1].training


def test_featurise_conserves_porosity():
    from conformal_rock.data import featurise
    rng = np.random.default_rng(0)
    pore = torch.from_numpy(rng.random((2, 32, 32, 32)) < 0.3)
    out, desc = featurise(pore, [4], cap=6)
    frac = out[4][:, 0].astype(float).sum((1, 2, 3)) / 32 ** 3
    assert np.allclose(frac, pore.float().mean((1, 2, 3)).numpy())
    assert np.allclose(desc[:, 0], frac)


def test_exclusions_and_build(tmp_path):
    from conformal_rock.data import apply_exclusions, build_dataset, make_synthetic_h5, read_labels
    h5 = str(tmp_path / "toy.h5")
    make_synthetic_h5(h5, n=40, dv=16)
    valid, rows = apply_exclusions(read_labels(h5))
    assert rows[-1]["n_new"] == len(valid) and rows[-2]["n_new"] == 40 - len(valid)
    res = build_dataset(h5, str(tmp_path / "out"), work_res=4, cap=4, batch=8, device="cpu", progress=False)
    meta = json.load(open(tmp_path / "out" / "proc_meta.json"))
    assert res.ones_is_solid and meta["N"] == len(valid)
    x = np.load(tmp_path / "out" / "X_work.npy")
    assert x.shape == (len(valid), 2, 4, 4, 4)
