# Contributing

Thank you for your interest. Bug reports, questions and pull requests are welcome.

## Ground rules

* The files in `results/` are the record of the run that produced the paper. They must keep matching
  `results/MANIFEST.json`, so please do not edit them. New results belong in a new folder or a new release.
* Keep the package free of hidden state: functions in `src/conformal_rock` take arrays and return arrays.
* Add a test for every new statistical routine. Where possible, test the guarantee itself on synthetic data
  with a known answer, as in `tests/test_conformal.py`.

## Development setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or a CUDA build
pip install -e ".[dev]"
pytest
ruff check src scripts tests
make smoke
```

## Reporting a problem with a number in the paper

Please include the table or figure number, the file in `results/` it comes from (see `docs/RESULTS.md`),
and the output of `python scripts/verify_release.py`.
