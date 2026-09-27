# Submitting an engine

## What a submission contains

A pull request adding `engines/<engine_id>/` with:

- `engine.yaml`: id, name, homepage, licence of runtime and models, pinned version,
  model files with SHA-256, wake words available, custom-word method (none, text,
  training, enrolment), on-device or API, notes.
- `Dockerfile`: builds an image that runs `wakewordworld eval run` offline, with all
  models pre-downloaded. Non-root user. The image digest is recorded with results.
- `DISCLOSURE.md`: training data used by the submitted models, in particular whether
  any source listed in `sources/` (or the benchmark's public release) was used; commercial
  status; any known limitations.
- An adapter in `src/wakewordworld/engines/<engine_id>.py` implementing the `Engine`
  protocol and registered in `engines/registry.py`.
- Tests with a fake SDK so CI does not need the runtime.

For custom wake words (the per-language names), the submitter trains or configures
the models following the engine's documented procedure and states exactly what data
went in. Training on benchmark audio disqualifies the submission.

## What the maintainers do

1. Build the image, record the digest.
2. Run the public and sealed sets on the reference hardware.
3. Publish score tables, curves and the summary with a two-week objection window in a
   GitHub issue.
4. Mark the result *verified* after the window closes, or record the objection and
   the resolution.

Vendors are welcome to propose configurations and dispute settings in the open. They
do not run evaluations that appear on the leaderboard.

## Commercial and API engines

Engines that need a licence key are run with a key held by the maintainers under the
vendor's free or evaluation terms; the terms are linked in `engine.yaml`. API-only
engines are evaluated the same way but flagged *not on-device*, and their latency
includes network time from the reference hardware.
