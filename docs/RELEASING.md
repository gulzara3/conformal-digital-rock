# Publishing release v2.0.0 (for the authors)

These steps put the repository on GitHub, attach the large files, and give it a citable DOI.

## 1. Push the code

```bash
cd conformal-digital-rock
git init -b main
git add .
git commit -m "Release v2.0.0: rebuilt study for the AIIG resubmission"
git remote add origin https://github.com/gulzara3/conformal-digital-rock.git
git push -u origin main --force      # --force replaces the v1.0 history; omit it to keep v1.0 and merge instead
```

If you would rather keep the v1.0 history, clone the existing repository, delete its old files, copy this folder's
contents in, commit, and push without `--force`. Then tag the old commit as `v1.0.0` for the record.

## 2. Create the release and attach the large files

1. On GitHub: Releases → Draft a new release → tag `v2.0.0` on `main` → title "v2.0.0: AIIG resubmission".
2. Attach `release_paper.zip` (the archive written by the notebook, 191 MB). Its SHA-256 entries are in
   `results/MANIFEST.json`, so `scripts/fetch_release.py` verifies it after download.
3. Zip the 30 checkpoints from `MyDrive/conformal_rock_v2/paper/ckpt/` as `checkpoints_v2.0.0.zip` and attach it.
   `checkpoints/SHA256SUMS` lists the expected hashes.
4. Publish. Check that `python scripts/fetch_release.py` downloads and verifies the archive.

## 3. Archive on Zenodo and get a DOI

1. Log in to Zenodo with GitHub and switch on the repository under Account → GitHub.
2. Publishing a GitHub release then creates a Zenodo record automatically. Alternatively, create a Zenodo upload
   by hand with `release_paper.zip` and `checkpoints_v2.0.0.zip`.
3. Put the DOI in the README badge, in `CITATION.cff`, and in the manuscript's Data availability statement in
   place of `[ZENODO DOI]` and `[REPOSITORY URL, RELEASE TAG]`.

## 4. Before submission

- [ ] CI is green on GitHub.
- [ ] `python scripts/verify_release.py --strict` passes with predictions and checkpoints present.
- [ ] The repository is public, or a private-link token is given to the editor.
- [ ] The README no longer says "submitted" once the paper is accepted.
