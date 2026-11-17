"""Reading the released predictions, splits and checksum manifest."""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results"
RELEASE = REPO_ROOT / "release"          # where scripts/fetch_release.py unpacks predictions

MAIN_RUNS = [f"main_s{s}" for s in range(10)]
SHIFT_RUNS = ["shiftA", "family_f0", "family_f1", "family_f2", "block"]


def _pred_dir(root: Path | None = None) -> Path:
    if root is not None:
        return Path(root)
    for cand in (RELEASE / "preds", RESULTS / "preds"):
        if cand.is_dir():
            return cand
    raise FileNotFoundError("Prediction files not found. Run `python scripts/fetch_release.py` first "
                            "(or pass the folder that holds preds/*.npz).")


@lru_cache(maxsize=64)
def _load(path: str) -> dict:
    z = np.load(path)
    out: dict = {}
    for k in z.files:
        part, name = k.split("__", 1)
        v = z[k]
        out.setdefault(part, {})[name] = v.astype(np.float64) if v.dtype.kind == "f" else v
    return out


def load_preds(run: str, root: str | Path | None = None) -> dict:
    """Predictions of one run as ``{part: {name: array}}`` with parts ``tr``, ``val``, ``cal``, ``te`` (and ``ood``)."""
    return _load(str(_pred_dir(root) / f"{run}.npz"))


def load_split(run: str) -> dict:
    """Row indices of one run's split (into the working dataset) plus ``h5_index`` mapping rows to DeePore."""
    z = np.load(RESULTS / "splits" / f"{run}.npz")
    return {k: z[k] for k in z.files}


def sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def load_manifest(path: str | Path | None = None) -> dict:
    return json.loads(Path(path or RESULTS / "MANIFEST.json").read_text())


def load_config() -> dict:
    return json.loads((RESULTS / "logs" / "config.json").read_text())
