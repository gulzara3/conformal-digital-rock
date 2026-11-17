#!/usr/bin/env python
"""Download the release archive (predictions of all 30 runs) and unpack it into ``release/``.

The archive is too large for the git history, so it is attached to the GitHub release and deposited on
Zenodo. After unpacking, every file is checked against ``results/MANIFEST.json``.

Usage
-----
    python scripts/fetch_release.py                         # from the GitHub release v2.0.0
    python scripts/fetch_release.py --url <zenodo file url> # from Zenodo or a mirror
    python scripts/fetch_release.py --zip release_paper.zip # from a local copy
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from conformal_rock.io import REPO_ROOT, load_manifest, sha256  # noqa: E402

DEFAULT_URL = "https://github.com/gulzara3/conformal-digital-rock/releases/download/v2.0.0/release_paper.zip"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--zip", help="local copy of release_paper.zip")
    ap.add_argument("--dest", default=str(REPO_ROOT / "release"))
    a = ap.parse_args()

    src = a.zip
    tmp = None
    if src is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".zip", delete=False).name
        print(f"downloading {a.url}")
        with urllib.request.urlopen(a.url) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f, 1 << 22)
        src = tmp

    want = {e["file"]: e["sha256"] for e in load_manifest()["files"]}
    os.makedirs(a.dest, exist_ok=True)
    n_ok = n_bad = 0
    with zipfile.ZipFile(src) as z:
        for name in z.namelist():
            parts = name.split("/")
            if ".." in parts or name.startswith("/") or name.endswith("/"):
                continue                         # never write outside the destination
            target = os.path.join(a.dest, *parts)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with z.open(name) as fi, open(target, "wb") as fo:
                shutil.copyfileobj(fi, fo, 1 << 22)
            if name in want:
                if sha256(target) == want[name]:
                    n_ok += 1
                else:
                    n_bad += 1
                    print(f"  checksum mismatch: {name}")
    if tmp:
        os.remove(tmp)
    print(f"unpacked into {a.dest}: {n_ok} files verified, {n_bad} mismatches")
    sys.exit(1 if n_bad else 0)


if __name__ == "__main__":
    main()
