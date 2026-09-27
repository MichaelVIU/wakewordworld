# Contributing

Thank you for helping build an independent wake word benchmark.

## Ground rules

- Be honest about provenance. Every source needs licence evidence. Every engine
  submission needs a training-data disclosure.
- No self-reported numbers on the leaderboard. Results are produced by the maintainers
  from pinned containers.
- No synthetic test audio, no self-recorded test audio.

## Development setup

```bash
uv sync --all-extras --dev
uv run pre-commit install
uv run pytest
uv run ruff check . && uv run ruff format --check .
uv run mypy
```

## Adding a source

1. Copy an existing file in `sources/`, name it `<id>.yaml` with a lower-case id.
2. Fill in access, licence and evidence. Quote the licence statement verbatim and record
   where and when you read it.
3. Run `uv run wakewordworld sources validate`.
4. Open a pull request. The licence gate in CI rejects tier-A claims without evidence.

## Adding an engine

See `docs/methodology/engine-submission.md`. In short: a folder under `engines/<id>/`
with a Dockerfile, pinned versions, model hashes, a config, and the disclosure form; an
adapter under `src/wakewordworld/engines/` implementing the `Engine` protocol.

## Pull requests

- One topic per pull request.
- Tests for new behaviour; `ruff`, `mypy` and `pytest` green.
- Conventional commit messages (`feat:`, `fix:`, `docs:`, `data:`, `engine:`).
- Do not commit audio, model files, or anything under `data/`.

## Code of conduct

This project follows the Contributor Covenant (see `CODE_OF_CONDUCT.md`).
