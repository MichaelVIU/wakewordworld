# WakeWordWorld — Project Plan

Status: draft, 2026-09-26. This document is the working plan for turning the research in `docs/research/` into an open-source, independent wake word benchmark. It is meant to be edited as decisions are made.

## 0. Goals and non-goals

Goals
- An independent, reproducible benchmark for wake word / keyword spotting engines, run by a party that ships no engine.
- One pooled test set of real, non-synthetic speech in English, German, French and Spanish, built only from existing openly licensed recordings. Every file is fully transcribed with word timestamps, so any word can be evaluated as a wake word after the fact.
- Full DET curves with confidence intervals, false accepts per hour per domain and language, false reject rate per word, latency and compute cost. Raw per-frame scores are published so anyone can re-threshold.
- Results for engines we may not redistribute audio for (NC-licensed sources) are still published; only the audio stays internal.

Non-goals
- No self-recorded data. No TTS or otherwise synthetic test audio. No training set (engines bring their own models).
- No claim that a mid-sentence name is "not a wake word". We measure acoustic detection; the report states that no engine can distinguish vocative from mention.

## 1. Project foundations

| Item | Decision needed | Proposed default |
|---|---|---|
| Name | yes | WakeWordWorld (working title); check name collisions on PyPI, GitHub, HF |
| Hosting | yes | GitHub organisation (code, issues, results), Hugging Face organisation (datasets, leaderboard), Zenodo (DOIs) |
| Code licence | yes | Apache-2.0. Note: the Jaco-Assistant benchmark is AGPLv3, so its code cannot be vendored; collaborate instead |
| Data licence | fixed by sources | Per-file licence carried in the manifest; dataset configs split by licence (cc0, cc-by, cc-by-sa); derived sets of CC BY-SA files are CC BY-SA |
| Docs licence | yes | CC BY 4.0 |
| Language of docs | yes | English for everything public |
| Python | yes | 3.11+, `uv` for env and lockfile, `ruff`, `pytest`, `pre-commit`, `mypy` on the core package |
| Versioning | yes | SemVer for the harness; SemVer for dataset releases (independent); CHANGELOG per component |

Repository layout (monorepo, single Python package):

```
wakewordworld/
  pyproject.toml
  src/wakewordworld/
    sources/      # source registry loader + fetchers (rss, peertube, internet_archive, ccc, hf, openslr, commons)
    ingest/       # download, licence evidence capture, normalisation, dedupe, segmentation
    transcribe/   # ASR with word timestamps, forced alignment, diarisation (optional)
    index/        # word index, name lexicon, vocative heuristics, phonetic near-miss index
    manifest/     # schema, validation, release freezing, checksums
    engines/      # adapter interface + one adapter per engine
    eval/         # streaming protocol, detection semantics, metrics, bootstrap CIs
    report/       # tables, DET plots, static site
  sources/        # one YAML per source (licence evidence, access method, filters)
  engines/        # one folder per engine: Dockerfile, pinned versions, model hashes, config
  manifests/      # frozen release manifests (Parquet + JSONL), no audio
  results/        # per-run results (Parquet) + run metadata
  docs/           # plan, methodology, dataset card, engine submission guide, research
  tests/
```

Governance documents to write before the first public release: README, CONTRIBUTING, CODE_OF_CONDUCT, GOVERNANCE (independence statement: maintainers ship no engine and accept no vendor funding tied to results), SECURITY, CITATION.cff, LICENSE files per component.

## 2. Legal and data policy

Licence tiers
- Tier A (redistributable): CC0, CC BY, CC BY-SA, CDLA-Permissive, public domain dedication, permissive software-style licences on recordings. Published on HF as gated dataset.
- Tier B (internal only): CC BY-NC, ND variants, research-only, click-through, Public Domain Mark without a dedication, unlicensed. Audio never leaves maintainer storage. Only per-clip scores and aggregated results are published. The manifest row is published (source, URL, duration, tags) so others can rebuild the set themselves.
- Not used at all: sources whose terms prohibit AI or data-mining use (example: iHeartMedia feeds), YouTube downloads outside curated CC corpora, LDC/ELRA corpora.

Rules
- Every file carries: source id, item URL, author/channel, licence, licence evidence (feed tag text, page URL, API field, snapshot hash), retrieval date, SHA-256.
- Fail closed: an item without machine-verifiable licence evidence goes to Tier B until a human adds evidence.
- Public Domain Mark on own work: request written CC0 confirmation from the uploader before promoting to Tier A.
- Gated dataset terms: attribution, no use for training wake word models, canary GUID present in metadata and file names.
- Takedown process documented; removed files are tombstoned in the next manifest release, never silently dropped.
- Privacy: only recordings that are already public; no enrichment with personal data; names in audio are treated as words, speaker identities are not linked across sources.
- Relicensing requests: template letter; candidate list in `docs/research/06-fr-es-sources.md` and `04-conversational-sources.md` (LQDN, AFPy/PyConFR, KDE España, Atareao, PDM radio uploaders, Thomas Munier).

## 3. Source registry and ingestion

Source spec (`sources/<id>.yaml`):
```
id: kuechenradio
language: de
domain: podcast
access: {type: rss, url: https://www.kuechenradio.org/feed/mp3/}
licence: {spdx: CC-BY-3.0-DE, tier: A, evidence: {type: page, url: ..., quote: "..."}}
filters: {min_duration_s: 600, exclude_title_regex: "..."}
expected_hours: 500
notes: "..."
```

Fetchers to implement (one per access type): RSS/podcast feeds (parse `podcast:license`, `creativeCommons:license`, `copyright`), PeerTube API (licence id + language, captions), Internet Archive advancedsearch + metadata (licenseurl per item), media.ccc.de API (events, audio-only recordings, subtitles), Hugging Face datasets (Common Voice, VoxPopuli, MSWC, CIEMPIESS, YODAS), OpenSLR/Zenodo tarballs (ICSI, AMI, DiPCo, NOTSOFAR, DEMAND, MUSAN), Wikimedia Commons API (P275 licence, P407 language).

Ingestion steps
1. Fetch item list, capture licence evidence per item, assign tier.
2. Download original; store as-is (content-addressed by SHA-256).
3. Normalise a working copy: 16 kHz mono, 16-bit PCM FLAC; record loudness stats and clipping.
4. Deduplicate (audio fingerprint, e.g. chromaprint) across sources.
5. Detect and tag music/jingles and long silences (do not cut in v0; tag only).
6. Segment long files into eval chunks (target 10–20 min, cut at silences, overlap-free) while keeping the parent file id for streaming evaluation.
7. Write ingestion manifest rows.

Initial slice (M1) per language, roughly 20–40 h each: EN ICSI + Hacker Public Radio; DE Küchenradio + media.ccc.de Q&A segments; FR Libre à vous! (has transcripts); ES KDE España + Espika FM.

## 4. Transcription and word index

- ASR with word timestamps for untranscribed audio: Parakeet-TDT-0.6B-v3 (EN, DE, FR, ES) as default; WhisperX as second opinion on a sample. Store both when they disagree on a name.
- Existing transcripts (ICSI, AMI, Libre à vous!, c3subtitles, Common Voice sentences, VoxPopuli segments) are force-aligned with Montreal Forced Aligner (`english_mfa`, `german_mfa`, `french_mfa`, `spanish_mfa`) rather than re-transcribed.
- Word index per file: `(word_normalised, word_raw, start, end, confidence, segment_id, speaker_id?)` in Parquet.
- Diarisation (pyannote) is optional in v0; needed later for speaker counts and vocative detection.
- Name lexicon per language, seeded from MSWC keyword counts and public first-name lists; vocative heuristic: name token at utterance start or end, followed or preceded by a pause or punctuation in the transcript.
- Phonetic near-miss index: G2P via espeak-ng phonemes, phoneme edit distance to each candidate wake word, stored per word so confusable negatives can be scored separately.
- Quality control: sampling protocol (per source, per language, N clips) with human listening; gold set of 200 name occurrences per language to measure ASR name error rate; known confusions list (Michel/Michael/Michaela, Jean/Jeanne, Juan/Juana).

## 5. Manifest and dataset releases

Manifest schema (Parquet + JSONL mirror), one row per eval chunk:
`chunk_id, file_id, source_id, language, licence_spdx, licence_tier, attribution, item_url, duration_s, domain (podcast|meeting|conference|parliament|read|radio|dinner), mic (close|far|array|phone|unknown), background_tags[], speakers_est, transcript_ref, word_index_ref, sha256, engine_training_overlap[] (which engines are known to have trained on this source), canary`

Rules
- Pooled set, no train/dev/test split. Slices are computed from tags at report time.
- A sealed subset (target 10 % of Tier A hours, all languages) is held back from the public release, used only by maintainer-run evals, rotated yearly and released at rotation.
- Each release: frozen manifest, checksums, dataset card with per-source attribution table, CHANGELOG, git tag, Zenodo DOI, HF dataset repo with configs per licence (`cc0`, `cc-by`, `cc-by-sa`) and Parquet audio for chunks under 1 MB, WebDataset tars otherwise.

Size targets for v0.1: per language at least 100 h Tier A negative audio across at least three domains, and at least three candidate wake words with at least 200 occurrences from at least 100 distinct source files each.

## 6. Evaluation harness

Engine adapter interface
- `load(config) -> Engine`; `reset()`; `process(frame: int16[N]) -> list[Detection | Score]`; `capabilities` (raw scores per frame, boolean only, sensitivity sweep supported, sample rate, frame size).
- Streaming protocol: audio is fed in fixed 80 ms frames at 16 kHz in file order; no lookahead; engines that need other frame sizes get an internal buffer.
- Detection semantics normalised in the harness, not in the adapter: rising edge over threshold, one detection per debounce window (default 1.0 s), a hit counts only if it fires in `[word_end - 0.5 s, word_end + 1.0 s]`; every other detection is a false accept. Boolean-only engines (Porcupine) are swept over sensitivity to draw curves.
- Raw per-frame scores are written to Parquet per engine and chunk.

Metrics
- False accepts per hour, per language and per domain, pooled and per slice.
- False reject rate per wake word; DET curve; FRR at 0.1, 0.5, 1 and 3 FA/h; EER; area under DET.
- Confusable false accepts per 1,000 near-miss occurrences.
- Cross-language false accepts (engine model for language X evaluated on languages Y, Z).
- Detection latency percentiles (word end to detection).
- Real-time factor, CPU, RAM on named hardware.
- Bootstrap confidence intervals over source files (not chunks) and over speakers where known; rank by lower bound.
- Stability: repeated runs with shuffled chunk order.

Augmentation lane (separate, labelled, never mixed with raw results): real noise from DEMAND, MUSAN, FSD50K CC0/CC BY at fixed SNRs; measured room impulse responses; speed perturbation 0.9–1.1.

Reference engines for v0.1: openWakeWord, microWakeWord (pymicro-wakeword), Vosk keyword grammar (ASR baseline). v0.2: LiveKit wakeword, Porcupine (free key, non-commercial use), ViolaWake, EfficientWord-Net, sherpa-onnx KWS. Each engine folder has a Dockerfile with pinned versions and model hashes.

Tracks
- Built-in words: words that ship with engines and occur in our data (alexa, computer, hey jarvis if present).
- Custom words: the per-language names chosen from the data. Engines that need training get a documented recipe; the submitter trains, discloses the data used, and the maintainers verify no benchmark audio was used.

## 7. Reporting

- Results repo: one Parquet per run with run metadata (harness version, dataset version, engine digest, hardware, timestamp).
- Report generator: static HTML per release (tables per language and slice, DET plots, per-word FRR, latency, cost), published on GitHub Pages first; HF Space leaderboard later.
- Badges: `verified` only for maintainer-run results from a pinned image digest; `community` for reproduced-by-others results; never self-reported numbers.
- Every report states: dataset version, sealed vs public gap per engine, engines whose training overlaps benchmark sources.

## 8. Infrastructure and automation

- CI on every PR: lint, unit tests, manifest and source-spec schema validation, licence gate (no Tier B rows in public artifacts, evidence present for every Tier A row), harness smoke test on a 5-minute fixture.
- Eval runs: HF Jobs `cpu-upgrade` for accuracy runs (a 100-hour pool costs cents); scheduled weekly re-run of all engines for drift; HF webhook on the submissions repo triggers a run on new PRs.
- On-device lane (later): self-hosted runner on a Raspberry Pi 5, only for maintainer-approved commits, no fork PRs, no secrets; ESP32-S3 as a flashed device under test for microWakeWord and Porcupine.
- Storage: Tier A on HF (Xet); Tier B and originals on maintainer storage with an offsite copy; Zenodo mirror of each Tier A release.

## 9. Community and submissions

- Submission = PR adding `engines/<name>/` with Dockerfile, config, model hashes, licence, and a disclosure form (training data, benchmark overlap, commercial/API status). Maintainers build, run, and publish. Objection window of two weeks via GitHub issues before a result is marked verified.
- Commercial engines are welcome; keys are held by maintainers; API-only engines are flagged as not on-device.
- Outreach once v0.1 exists: openWakeWord and microWakeWord maintainers, the Jaco-Assistant benchmark author (collaboration on protocol), Home Assistant / OHF-Voice, Rhasspy and OpenVoiceOS communities, LiveKit.

## 10. Milestones

| Milestone | Scope | Exit criterion |
|---|---|---|
| M0 Skeleton | Repo, licences, governance docs, package scaffold, CI, source spec schema, manifest schema, first six source specs with evidence | `uv run pytest` green; `wakewordworld sources validate` passes on all specs |
| M1 First slice | Fetchers for RSS, PeerTube, IA, ccc, HF, OpenSLR; ingestion + normalisation; transcription + alignment; word index; stats report | 20–40 h per language ingested; name frequency table per language; wake words chosen per language and recorded in docs |
| M2 Harness | Adapter interface, streaming protocol, detection semantics, metrics, bootstrap CIs, raw score storage; openWakeWord, microWakeWord, Vosk adapters in Docker | First internal report on the M1 slice with DET curves and CIs |
| M3 Release v0.1 | Scale ingestion to size targets; sealed split; dataset card; HF gated dataset (Tier A) + Zenodo DOI; results v0.1; static report on GitHub Pages | Public dataset and results published; reproducibility check by one outside person |
| M4 Breadth | LiveKit, Porcupine, ViolaWake, EfficientWord-Net, sherpa-onnx; augmentation lane; FR/ES scale-up; relicensing requests sent | Eight engines on the board; augmentation results published separately |
| M5 Open submissions | Submission process, webhook-triggered HF Jobs, weekly re-runs, objection window, HF Space leaderboard | First external engine submitted and verified |
| M6 On-device | Raspberry Pi 5 runner, ESP32-S3 DUT, latency and power lane | On-device numbers for at least three engines |

## 11. Open decisions (resolve at M0)

1. Project and package name; GitHub and HF organisation names.
2. Apache-2.0 vs MIT for code (proposal: Apache-2.0 for the patent grant).
3. Whether to approach the Jaco-Assistant author before M2 to align the streaming protocol.
4. Gated vs fully open HF dataset (proposal: gated with automatic approval, for the terms and the access log).
5. Where Tier B audio lives long-term (maintainer NAS plus encrypted offsite copy).
6. Which four to six candidate wake words per language, decided from M1 statistics, not before.

## 12. Status (2026-09-27)

| Milestone | State | Notes |
|---|---|---|
| M0 Skeleton | done | repo, licences, governance, schemas, CLI, CI with licence gate |
| M1 First slice | done (small) | 21 source specs; fetchers for 7 access types; ingest, transcription, word index, manifest; first internal slice of ~11 h (de/en/fr/es) limited by local disk |
| M2 Harness | done | streaming protocol, detection semantics, metrics with cluster bootstrap, 7 engine adapters (openWakeWord, microWakeWord, Vosk, Porcupine, LiveKit, EfficientWord-Net, sherpa-onnx); first internal runs |
| M3 Release v0.1 | tooling done, release pending | report generator, dataset card, HF publish and Zenodo clients, release workflow exist; publishing needs a Hugging Face organisation and tokens, and the size targets need more disk than the development machine has |
| M4 Breadth | done (code) | augmentation lane (DEMAND/MUSAN/RIRS), 4 additional adapters, relicensing letter drafts; noise sets not downloaded locally (disk) |
| M5 Open submissions | done (code) | submission checks, PR workflow with image build and smoke test, weekly HF Jobs workflow, webhook script, ops runbooks; needs HF credentials to go live |
| M6 On-device | scripted, hardware pending | measurement script, self-hosted runner workflow, ESP32 procedure documented; no Raspberry Pi / ESP32 available in this environment |
