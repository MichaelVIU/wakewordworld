# Metrics

All metrics are computed from a *count table*: for each unit (parent file) and each
threshold, the number of hits, misses, false accepts, and the negative seconds
(`wakewordworld.eval.metrics.CountTable`). Threshold sweeps and bootstrap resampling
are array operations on that table.

| Metric | Definition | Reported as |
|---|---|---|
| False accepts per hour (FA/h) | false accepts ÷ negative hours | per language, per domain, pooled; per engine and threshold |
| False reject rate (FRR) | misses ÷ (hits + misses) | per wake word, per language; curve over thresholds |
| FRR at budget | FRR at the operating point with the lowest FRR whose FA/h ≤ budget | budgets 0.1, 0.5, 1, 3 FA/h |
| DET curve | (FA/h, FRR) over all thresholds | plot, and table download |
| EER | point where FRR equals the false-accept rate normalised by positives per hour | single number, secondary |
| AUT | mean best-achievable FRR over log10(FA/h) in [0.1, 3] | single number, primary ranking metric |
| Confusable FA | false accepts inside windows of phonetic near-miss words | count per 1,000 near-miss occurrences |
| Cross-language FA/h | FA/h of a model on languages it was not built for | per language pair |
| Latency | detection time − word end, for hits at the 0.5 FA/h operating point | p50, p90, p99 |
| Real-time factor | wall time ÷ audio time on the named reference hardware | per engine |

## Confidence intervals

Cluster bootstrap over parent files, 1,000 resamples, 95 % percentile intervals, for
FRR at every budget and for AUT. A budget unreachable in a resample counts as FRR = 1.
Rankings use the *lower* bound of AUT's complement (that is, engines are ordered by the
pessimistic end of their interval), and an engine is listed only when its positives
exceed 100 occurrences from 50 parent files for the wake word in question.

## Slices

Every metric is available for slices defined by manifest tags: language, domain,
microphone, background tags, licence tier, and "excluding sources the engine trained
on" (`engine_training_overlap`). The headline table is the pooled tier A + tier B set
excluding training overlap; the public-only table (tier A) is reported alongside so
others can reproduce it exactly.

## What is not reported

- Accuracy, precision or F1 on balanced clips; these hide the trade-off.
- Any number at a threshold chosen by the engine vendor without the curve next to it.
- Results from audio that was used to train the engine, in the headline table.
