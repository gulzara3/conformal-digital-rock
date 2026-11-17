# Trained networks

The 30 checkpoints of the study (about 17 MB each, 509 MB in total) are too large for the git history.
They are attached to the GitHub release **v2.0.0** as `checkpoints_v2.0.0.zip` and deposited on Zenodo.

```bash
# after downloading checkpoints_v2.0.0.zip into this folder
unzip checkpoints_v2.0.0.zip -d checkpoints/
sha256sum -c checkpoints/SHA256SUMS        # run from the repository root after `cd checkpoints`
python scripts/verify_release.py           # checks them against results/MANIFEST.json as well
```

Each file stores the model weights (`model`), the training-set mean and standard deviation used to standardise
the targets (`ym`, `ys`), the training history and the run settings. Load one with

```python
from conformal_rock.model import load_checkpoint
model, ym, ys, run = load_checkpoint("checkpoints/main_s0.pt")
```

| Runs | Role |
|---|---|
| `main_s0` ... `main_s9` | main study, ten seeds |
| `meanonly_s0..2`, `gauss_s0_m0`, `gauss_s1_m0`, `gauss_s2_m0` | training ablations |
| `gauss_s0_m1` ... `gauss_s0_m4` | deep-ensemble members (with `gauss_s0_m0`) |
| `shiftA`, `family_f0..2`, `block` | distribution-shift experiments |
| `res_main`, `res_frac`, `res_bin`, `res_lo`, `res_hi` | input-representation ablation (2,000 volumes) |
