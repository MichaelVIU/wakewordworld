# Release runbook

Two independent release trains: the **harness** (Python package, tags `harness-vX.Y.Z`)
and the **dataset** (frozen manifest plus audio, tags `dataset-vX.Y.Z`).

## Dataset release

Run on the machine that holds the data root (audio is never in CI).

1. Ingest, transcribe and index are complete for every source in the release;
   `uv run wakewordworld ingest status --all` shows no failures you have not triaged.
2. Build the public manifest (sealed rows are split off automatically):
   ```bash
   uv run wakewordworld manifest build --version X.Y.Z --public
   uv run wakewordworld manifest validate manifests/X.Y.Z
   uv run python scripts/licence_gate.py
   ```
   `manifests/X.Y.Z/sealed/` is git-ignored and must stay on the maintainer machine.
3. Add a changelog entry to `manifests/CHANGELOG.md` (new sources, removed items,
   tombstones, size per language).
4. Build the release files and inspect them:
   ```bash
   uv run wakewordworld hub build-release --manifest manifests/X.Y.Z --out dist/dataset-X.Y.Z
   uv run wakewordworld hub card --manifest manifests/X.Y.Z --out /tmp/README.md   # review the card
   ```
   Check `dist/dataset-X.Y.Z/build.json`: `missing_audio` must be empty.
5. Upload to the gated Hugging Face dataset (private first, flip to public after a
   final look at the card and the viewer):
   ```bash
   HF_TOKEN=... uv run wakewordworld hub upload --dir dist/dataset-X.Y.Z --repo wakewordworld/benchmark --tag X.Y.Z --private
   ```
6. Commit `manifests/X.Y.Z` (without `sealed/`) and tag:
   ```bash
   git add manifests/X.Y.Z manifests/CHANGELOG.md
   git commit -m "data: release X.Y.Z"
   git tag dataset-vX.Y.Z && git push --tags
   ```
   The `Release` workflow validates the manifest and attaches `release.json`,
   `SHA256SUMS`, `sources.md` and the card to a GitHub release.
7. Deposit on Zenodo (sandbox first):
   ```bash
   ZENODO_TOKEN=... uv run wakewordworld hub zenodo --dir dist/dataset-X.Y.Z --sandbox --dry-run
   ZENODO_TOKEN=... uv run wakewordworld hub zenodo --dir dist/dataset-X.Y.Z \
       --github-tag-url https://github.com/MichaelVIU/wakewordworld/releases/tag/dataset-vX.Y.Z \
       --hf-url https://huggingface.co/datasets/wakewordworld/benchmark --publish
   ```
   Record the DOI in `manifests/CHANGELOG.md` and the dataset card.
8. Make the Hugging Face repo public (gated, automatic approval) and mirror the
   internal copy (public manifest + sealed split + audio) to the private
   `wakewordworld/benchmark-internal` repo used by evaluation jobs.

## Harness release

1. Bump `version` in `pyproject.toml` and `CITATION.cff`; move the `Unreleased`
   section of `CHANGELOG.md` under the new version.
2. `uv run pytest && uv run ruff check . && uv run mypy`.
3. Commit, tag `harness-vX.Y.Z`, push tags. The `Release` workflow runs the tests,
   builds the wheel and creates the GitHub release with the changelog section.
4. Rebuild and push the engine images with the new tag so verified runs use it.

## Tombstones and takedowns

A removed recording keeps its manifest row in the next release with
`audio_sha256 = null` and a `notes` entry in the changelog naming the takedown issue.
Never rewrite a published release.
