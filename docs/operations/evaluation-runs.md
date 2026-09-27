# Evaluation runs

Only results produced by maintainers from a pinned container image are marked
*verified*. This page is the operator's guide; the rules are in
`docs/methodology/protocol.md` and `engine-submission.md`.

## Local verified run

```bash
docker build -f engines/<engine>/Dockerfile -t wakewordworld-<engine>:X.Y.Z .
docker image inspect --format '{{index .RepoDigests 0}}' wakewordworld-<engine>:X.Y.Z   # after push
docker run --rm --network none \
  -v "$WWW_DATA_ROOT:/data:ro" -v "$PWD/results:/results" \
  wakewordworld-<engine>:X.Y.Z eval run <engine> --manifest /data/manifests/X.Y.Z --out /results
```

The data root mounted at `/data` contains `manifests/X.Y.Z/` including the `sealed/`
folder, so the run covers public and sealed chunks; the run writes a `sealed=yes/no`
slice into `summary.parquet`. Record the image digest in the run's `run.json`
(`container_digest`) before publishing.

## Runs on Hugging Face Jobs

The private dataset repo `wakewordworld/benchmark-internal` mirrors the data root
(public manifest, sealed split, chunk audio, word index). Jobs mount it read-only:

```bash
HF_TOKEN=... uv run python scripts/hf_jobs_eval.py \
  --engine openwakeword --engine microwakeword --engine vosk \
  --manifest-version X.Y.Z --image-prefix ghcr.io/<owner>/wakewordworld --image-tag X.Y.Z \
  --dataset-repo wakewordworld/benchmark-internal --results-repo wakewordworld/results
```

Flavour `cpu-upgrade` (8 vCPU, 32 GB) is enough for CPU engines; a 100-hour pool costs
cents. Results are uploaded from inside the job to the results dataset under
`results/X.Y.Z/<engine>/`. The weekly workflow (`eval-weekly.yml`) re-runs every engine
to catch drift; it needs the `HF_TOKEN` secret and the `IMAGE_PREFIX`, `DATASET_REPO`,
`RESULTS_REPO`, `EVAL_ENGINES` repository variables.

To trigger a run automatically when a submission pull request is opened on the
submissions dataset repo, create a template job once and register a webhook:

```bash
python scripts/hf_webhook_setup.py --watch-repo wakewordworld/submissions --job-id <template job id>
```

## Publishing a result

1. Copy the run directory into `results/<manifest version>/<engine>/<run id>/`.
2. Open a GitHub issue "Result: <engine> <version> on <manifest>" linking the run,
   the image digest and the summary table. The objection window is two weeks.
3. After the window: label the issue `verified`, add the run to the leaderboard build
   (`wakewordworld report build`), and close the issue with a link to the report.
4. Objections are resolved in the issue; a changed configuration means a new run and a
   new window.

## Sealed split hygiene

- `manifests/*/sealed/` is git-ignored; never copy it into a public repo or artefact.
- Sealed results appear only as the `sealed` slice and the public-vs-sealed gap.
- Rotate yearly: rebuild the manifest with a new seed, publish the old sealed rows.
