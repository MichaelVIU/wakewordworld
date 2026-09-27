# Test set design

## Principles

1. Real speech only. No TTS, no self-recorded clips.
2. Existing openly licensed corpora only; licence evidence per item (see
   `docs/legal/licence-policy.md`).
3. One pool, tagged. No train/dev/test split (there is no training). Slices are
   computed from tags at report time.
4. Word-indexed. Every chunk is fully transcribed with word timestamps, so any word can
   be evaluated as a wake word.
5. Frozen releases with checksums, a DOI, and a changelog. Corrections are new
   releases with tombstones.

## Sources

The source registry is `sources/*.yaml`. The surveys behind it are in
`docs/research/03-datasets.md`, `04-conversational-sources.md` and
`06-fr-es-sources.md`. The first slice per language:

| Language | Conversational (vocative-rich) | Negative hours |
|---|---|---|
| en | ICSI meetings, Hacker Public Radio, AMI, DiPCo, NOTSOFAR-1 | VoxPopuli, Common Voice, LibriSpeech |
| de | Küchenradio, media.ccc.de Q&A, Forschergeist, Radio Tux | VoxPopuli, Common Voice, Tuda-De, SWC |
| fr | Libre à vous!, Radio Cause Commune, PSES/JdLL débats | VoxPopuli, Common Voice, YODAS fr000 |
| es | KDE España, Espika FM, Una Radio Muchas Voces, D-Strip-Ando | VoxPopuli, Common Voice, CIEMPIESS |

Tier B material (NC-licensed podcasts, CHiME-6, TalkBank, ORTOLANG family corpora,
AMERESCO) is evaluated but never redistributed.

## Wake words

Chosen per language from the word index after the first ingestion, not before. Criteria
for a candidate: at least 200 occurrences from at least 100 parent files across at
least two sources, at least 30 vocative occurrences, and at least one phonetic
near-miss word with 50+ occurrences. Candidates are recorded in
`docs/methodology/wake-words.md` with their counts.

## Chunks and units

Files are cut into 10 to 20 minute chunks at silences. Chunks share the parent file
id; the parent file is the resampling unit for confidence intervals so that a
three-hour podcast does not count as twelve independent observations.

## Sealed subset

Ten percent of tier A parent files (deterministic hash of the file id) are withheld
from the public release. Only maintainer-run evaluations see them. The gap between
public and sealed results is reported per engine as a contamination signal. The sealed
subset is rotated once a year; the previous one becomes public.

## Size targets

Per language: at least 100 hours of tier A negative audio across at least three
domains; at least three candidate wake words meeting the criteria above. Resolution
argument: at 100 hours, 0.1 FA/h corresponds to ten events, the minimum for a usable
interval.

## Quality control

- ASR name error rate measured on a gold set of 200 name occurrences per language,
  human-listened, refreshed per release.
- Sampling protocol: 20 random chunks per source per release, listened to for licence
  compliance (no embedded third-party music beyond tagged segments) and transcript
  sanity.
- Music/jingle segments are tagged and excluded from the word index; they remain in
  the negative audio (engines must not fire on music).

## Augmented lane

Real noise (DEMAND, MUSAN, FSD50K CC0/CC BY) mixed at fixed SNRs and measured room
impulse responses are a separate, labelled lane. Augmented results are never pooled
with raw results.
