## What

<!-- one paragraph -->

## Type

- [ ] code
- [ ] source (new or changed `sources/*.yaml`; licence evidence quoted verbatim, captured date set)
- [ ] engine (folder under `engines/`, adapter, Dockerfile with pinned versions and model hashes, disclosure form filled)
- [ ] docs

## Checklist

- [ ] `uv run pytest` green
- [ ] `uv run ruff check . && uv run ruff format --check . && uv run mypy` green
- [ ] no audio, model files or anything under `data/` committed
- [ ] for engines: training-data disclosure states whether any benchmark source was used
