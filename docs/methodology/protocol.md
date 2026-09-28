# Evaluation protocol

This document fixes the rules every engine is evaluated under. Changing any of them
is a methodology change (see `GOVERNANCE.md`) and bumps the benchmark's major version.

## Audio

- 16 kHz, mono, 16-bit PCM, decoded from the normalised FLAC of each evaluation chunk.
- Chunks are 10 to 20 minutes long, cut at silences, and belong to a parent file. The
  parent file is the statistical unit for confidence intervals.
- Audio is fed in fixed frames of 1280 samples (80 ms) in file order, with no lookahead.
  The final frame of a chunk is zero-padded. Adapters that need other frame sizes
  buffer internally.
- The engine is reset at the start of every chunk.

## Scores

After every frame the adapter returns one score in [0, 1] per wake word. Engines that
only expose a boolean detection return 1.0 on the frame where they fire and 0.0
otherwise; their trade-off curve is obtained by sweeping the engine's sensitivity
parameter and re-running.

Raw per-frame scores are stored (`scores/<engine>/<run>/<chunk>.parquet`) and
published with the results. Everything below is derived from them and can be
recomputed.

## Detection semantics

Applied identically to all engines, implemented in `wakewordworld.eval.detect`:

1. A detection fires on a rising edge: score ≥ threshold where the previous frame was
   below.
2. After a detection, further detections are suppressed for 1.0 s (debounce).
3. A detection is a hit for a target occurrence if it fires within the occurrence's hit
   window [min(word start, word end − 0.5 s), word end + 1.0 s]. For precisely aligned
   words that is [word end − 0.5 s, word end + 1.0 s]; for reference occurrences without
   word alignment (single-word clips, where the word spans the clip) the window covers
   the clip. Each detection matches at most one occurrence and each occurrence at most
   one detection (earliest first).
4. Every unmatched detection is a false accept. Every unmatched occurrence is a miss.
5. Negative time is the chunk duration minus the union of the hit windows, so false
   accepts per hour are computed over time where the target word was not spoken.

Word ends come from the word index (forced alignment for reference transcripts, ASR
word timestamps otherwise). Occurrences of phonetic near-miss words are tracked so
false accepts inside their windows can be reported separately as *confusable false
accepts*; they still count as false accepts.

## Targets

A wake word is a sequence of one or more normalised tokens. Occurrences are found in
the word index per chunk; multi-token phrases must be consecutive with gaps ≤ 0.5 s.

The benchmark measures acoustic detection. It does not distinguish a name used as an
address from a name mentioned in a sentence, and no engine can; the report says so.

## Thresholds

For continuous-score engines, thresholds are the 200 quantiles of the observed score
distribution plus 0 and 1. All metrics are reported as curves over these thresholds;
headline numbers are read off the curve at fixed false-accept budgets.

## Runs

A run is identified by: benchmark manifest version, engine id, engine version and model
hashes, container image digest, detection configuration, harness version, hardware.
Two runs with the same identity must produce identical score tables (adapters are
required to be deterministic; any nondeterminism is a reported defect).
