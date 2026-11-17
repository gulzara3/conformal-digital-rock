# Released results (v2.0.0)

Everything here was written by the notebook in a single run and is covered by the SHA-256 manifest
`MANIFEST.json`. Please do not edit these files; `python scripts/verify_release.py` and CI check them.

| Folder | Content |
|---|---|
| `tables/` | T01 to T32 as CSV and `key_numbers.json` (map to paper numbers: `docs/RESULTS.md`) |
| `figures/` | F01 to F16 as PNG and PDF |
| `splits/` | one `.npz` per run: row indices `tr`, `val`, `cal`, `te` and `h5_index` (row → DeePore index) |
| `data/` | retained DeePore indices, exclusion counts, connectivity check, preprocessing metadata |
| `logs/` | `config.json`, `environment.json`, frozen requirements, one JSON log per network |

Predictions (`preds/*.npz`) and checkpoints are release assets; see the main README.
