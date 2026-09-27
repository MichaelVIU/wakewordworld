# Submitting an engine: checklist

Full rules: `docs/methodology/engine-submission.md`.

- [ ] `engines/<id>/engine.yaml` with id, name, homepage, licence (runtime, models),
      version, models (each with `sha256` or `unpinned`), wake_words,
      custom_word_method (none | text | training | enrolment), on_device.
- [ ] `engines/<id>/Dockerfile`: runs offline, models pre-downloaded, non-root `USER`,
      entrypoint `wakewordworld`.
- [ ] `engines/<id>/DISCLOSURE.md` with the headings **Training data**,
      **Benchmark overlap**, **Commercial status**.
- [ ] Adapter `src/wakewordworld/engines/<id>.py` registered in `engines/registry.py`,
      with unit tests that use a fake SDK.
- [ ] `python scripts/check_engine_folder.py engines/<id>` passes.
- [ ] Pull request opened; the `Engine submission` workflow validates the folder, builds
      the image and runs the smoke test. Add the label `run-eval` to request a
      maintainer-run evaluation.

You do not run the evaluation that appears on the leaderboard; maintainers do, from
the image digest recorded in the pull request.
