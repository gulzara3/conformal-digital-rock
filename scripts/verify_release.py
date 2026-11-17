#!/usr/bin/env python
"""Check every released file against the SHA-256 manifest of the v2.0.0 run.

The manifest (``results/MANIFEST.json``) was written by the notebook at the end of the run that produced
the paper. Files are looked up in this repository:

    tables/, figures/, logs/, splits/, data/  ->  results/<same path>
    preds/                                    ->  release/preds/   (after scripts/fetch_release.py)
    ckpt/                                     ->  checkpoints/     (after downloading the checkpoints)

Usage
-----
    python scripts/verify_release.py            # report, exit code 1 on any mismatch
    python scripts/verify_release.py --strict   # also fail when predictions or checkpoints are absent
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from conformal_rock.io import REPO_ROOT, RESULTS, load_manifest, sha256  # noqa: E402


def local_path(rel: str):
    top, rest = rel.split("/", 1)
    if top == "preds":
        return REPO_ROOT / "release" / "preds" / rest
    if top == "ckpt":
        return REPO_ROOT / "checkpoints" / rest
    return RESULTS / top / rest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strict", action="store_true")
    a = ap.parse_args()
    man = load_manifest()
    entries = [("file", e) for e in man["files"]] + [("checkpoint", e) for e in man.get("checkpoints_not_in_zip", [])]
    counts = dict(ok=0, bad=0, missing=0)
    missing_groups = set()
    for kind, e in entries:
        p = local_path(e["file"])
        if not p.exists():
            counts["missing"] += 1
            missing_groups.add(e["file"].split("/", 1)[0])
            continue
        if sha256(p) == e["sha256"]:
            counts["ok"] += 1
        else:
            counts["bad"] += 1
            print(f"  MISMATCH {e['file']}")
    print(f"{counts['ok']} files match, {counts['bad']} differ, {counts['missing']} not present locally")
    if missing_groups:
        print("  not present:", ", ".join(sorted(missing_groups)),
              "(predictions: scripts/fetch_release.py; checkpoints: see checkpoints/README.md)")
    if counts["bad"] or (a.strict and counts["missing"]):
        sys.exit(1)


if __name__ == "__main__":
    main()
