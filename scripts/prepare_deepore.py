#!/usr/bin/env python
"""Download the public DeePore archive, verify it, and build the working dataset.

This is the build script for every result in the paper. It replaces the private ``deepore_96.npz`` file of
the first submission.

Examples
--------
    # full build (about 30 min on a Colab L4; needs ~13 GB free disk for the archive)
    python scripts/prepare_deepore.py --raw data/raw --out data/processed

    # use an archive you already have
    python scripts/prepare_deepore.py --h5 /path/to/DeePore_Dataset.h5 --out data/processed

    # tiny synthetic check of the code path (no download)
    python scripts/prepare_deepore.py --synthetic --out /tmp/cdr_smoke --work-res 8 --cap 6
"""
from __future__ import annotations

import argparse
import os
import sys
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from conformal_rock.config import CFG, ZENODO_MD5, ZENODO_URL  # noqa: E402
from conformal_rock.data import build_dataset, make_synthetic_h5, md5sum  # noqa: E402


def download(url: str, dest: str) -> None:
    """Stream the file to ``dest`` with resume support."""
    part = dest + ".part"
    have = os.path.getsize(part) if os.path.exists(part) else 0
    req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
    with urllib.request.urlopen(req) as r, open(part, "ab" if have else "wb") as f:
        total = int(r.headers.get("Content-Length", 0)) + have
        done = have
        while True:
            b = r.read(1 << 22)
            if not b:
                break
            f.write(b)
            done += len(b)
            if total:
                print(f"\r  {done / 2**30:6.2f} / {total / 2**30:.2f} GB", end="", flush=True)
    print()
    os.replace(part, dest)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5", help="existing DeePore_Dataset.h5 (skips the download)")
    ap.add_argument("--raw", default="data/raw", help="folder for the downloaded archive")
    ap.add_argument("--out", default="data/processed", help="output folder")
    ap.add_argument("--work-res", type=int, default=CFG.work_res)
    ap.add_argument("--cap", type=int, default=CFG.poremap_cap, help="distance-map cap in voxels")
    ap.add_argument("--n-max", type=int, default=0, help="random subset size (0 = all valid volumes)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-md5", action="store_true", help="skip the checksum check")
    ap.add_argument("--synthetic", action="store_true", help="build from a tiny synthetic file (code test only)")
    a = ap.parse_args()

    if a.synthetic:
        h5 = os.path.join(a.out, "SYNTHETIC_test.h5")
        os.makedirs(a.out, exist_ok=True)
        make_synthetic_h5(h5)
        check = None
    else:
        h5 = a.h5 or os.path.join(a.raw, "DeePore_Dataset.h5")
        if not os.path.exists(h5):
            os.makedirs(a.raw, exist_ok=True)
            print(f"downloading {ZENODO_URL}")
            download(ZENODO_URL, h5)
        check = None if a.no_md5 else ZENODO_MD5
        if check:
            print("checking MD5 ...")
            got = md5sum(h5)
            print(f"  {got} ({'matches Zenodo' if got == ZENODO_MD5 else 'MISMATCH'})")
            if got != ZENODO_MD5:
                sys.exit("Checksum mismatch: delete the file and run again.")
            check = None   # already verified

    res = build_dataset(h5, a.out, work_res=a.work_res, cap=a.cap, n_max=a.n_max, batch=a.batch,
                        device=a.device, check_md5=check)
    print(f"built {res.n} volumes in {res.out_dir}; stored 1 = {'solid' if res.ones_is_solid else 'pore'} "
          f"(Spearman of ones fraction with log k = {res.phase_rho:+.2f})")


if __name__ == "__main__":
    main()
