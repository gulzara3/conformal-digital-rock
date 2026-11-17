.PHONY: install test lint verify fetch reproduce data smoke clean

install:        ## install the package with development tools
	pip install -e ".[dev,torch]"

test:           ## run the test suite
	pytest

lint:           ## static checks
	ruff check src scripts tests

verify:         ## check released files against the SHA-256 manifest
	python scripts/verify_release.py

fetch:          ## download the predictions of all 30 runs into release/
	python scripts/fetch_release.py

reproduce:      ## recompute the headline tables from the predictions
	python scripts/reproduce_tables.py

data:           ## download DeePore and build the working dataset
	python scripts/prepare_deepore.py --raw data/raw --out data/processed

smoke:          ## end-to-end code test on synthetic data (CPU, about a minute)
	python scripts/prepare_deepore.py --synthetic --out /tmp/cdr_smoke --work-res 8 --cap 6
	python scripts/train.py --data /tmp/cdr_smoke --run smoke --epochs 2 --batch 8 --mc-T 3 --out /tmp/cdr_runs

clean:
	rm -rf reproduced .pytest_cache build *.egg-info src/*.egg-info
