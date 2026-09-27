# WakeWordWorld

An independent, reproducible benchmark for wake word and keyword spotting engines,
evaluated on real, non-synthetic speech in English, German, French and Spanish.

> Status: pre-release. Pipeline, harness, seven engine adapters and release tooling are
> in place; a first internal slice (about 11 hours, four languages) has been evaluated
> for pipeline validation only. No published result yet. See `docs/PLAN.md` section 12.

## Why

Every published wake word comparison today is run by a party that wins it, on data
nobody else can obtain, reported as a single operating point. There is no shared test
set, no confusable-word measurement, no speaker or domain breakdown, no confidence
intervals, and no version pinning. `docs/research/01-existing-benchmarks.md` documents
the state of the art and its gaps in detail.

## What this project does differently

- **One pooled test set of real speech.** Podcasts, meetings, conference Q&A, community
  radio, parliament, read speech. No TTS, no self-recorded data. Every file is fully
  transcribed with word timestamps, so *any* word can be evaluated as a wake word after
  the fact. Wake words are chosen from names that actually occur in the data.
- **Licence-first.** Every recording carries its licence and the evidence for it. Tier A
  material (CC0, CC BY, CC BY-SA, CDLA) is published; tier B material (NC, ND, research
  only) is evaluated internally and only the results are published. Anything unclear
  fails closed to tier B.
- **Curves, not points.** Full DET curves with bootstrap confidence intervals, false
  accepts per hour per language and domain, false reject rate per word, confusable
  false accepts, cross-language false accepts, latency, compute cost. Raw per-frame
  scores are published so anyone can re-threshold.
- **Uniform detection semantics.** Rising edge, debounce, and a hit window tied to the
  word end are applied by the harness, identically for every engine.
- **Independence.** Maintainers ship no engine. Results are only marked verified when
  the maintainers ran them from a pinned container image.

## Layout

```
src/wakewordworld/   Python package: sources, ingest, transcribe, index, manifest, engines, eval, report
sources/             one YAML per audio source with licence evidence
engines/             one folder per engine: Dockerfile, pinned versions, model hashes
manifests/           frozen release manifests (no audio)
results/             evaluation results per run
docs/                plan, methodology, legal, research
```

## Quick start (development)

```bash
uv sync --dev --extra transcribe --extra engines-oww --extra engines-mww
uv run wakewordworld sources validate            # 21 sources, licence evidence checked
uv run wakewordworld ingest run kuechenradio --max-items 2 --max-hours 3
uv run wakewordworld transcribe run kuechenradio --backend faster-whisper --model large-v3-turbo
uv run wakewordworld index build kuechenradio && uv run wakewordworld index names --language de
uv run wakewordworld manifest build --version 0.0.1 --internal
uv run wakewordworld eval run openwakeword --manifest manifests/0.0.1-internal
uv run wakewordworld report build --results results --out site/index.html
uv run pytest
```

Pipeline stages: `sources` -> `ingest` -> `transcribe` -> `index` -> `manifest` -> `eval`
-> `report`; `augment` builds the noise/RIR lane, `hub` publishes releases.

Audio and other large artefacts live under `./data` (or `WWW_DATA_ROOT`) and are never
committed.

## Documents

- `docs/PLAN.md` — the project plan and milestones
- `docs/methodology/` — metrics, protocol, and dataset design
- `docs/legal/` — licence policy, tiers, takedown, relicensing requests
- `docs/research/` — the source, engine, benchmark and infrastructure surveys this
  project is built on

## Licence

Code: Apache-2.0 (see `LICENSE`). Documentation: CC BY 4.0. Audio: per file, as recorded
in the manifest; see `docs/legal/licence-policy.md`.
