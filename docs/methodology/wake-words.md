# Wake word candidates

Wake words are chosen from the word index after ingestion, never before. A candidate
must meet all of: at least 200 occurrences, from at least 100 parent files, across at
least two sources, at least 30 vocative occurrences (name at utterance start or end or
next to a pause), and at least one phonetic near-miss word with 50 or more occurrences.

The tables below are produced by `wakewordworld index names --language <lang>` and
`wakewordworld index near-miss <word> --language <lang>` on the current internal pool
and are refreshed at every dataset release. Counts from the first internal slice are
indicative only.

## Status

First internal slice (2026-09-27, about 11 hours, two or three episodes per source, limited
by local disk). Counts from `wakewordworld index names`; occurrences / vocatives / distinct
parent files. None of these meets the candidate criteria yet; they show that the
vocative-rich sources deliver exactly the material the benchmark needs.

| Language | Pool | Leading names (occurrences / vocatives / files) | Decision |
|---|---|---|---|
| en | Hacker Public Radio, 0.7 h | arthur 10 / 1 / 2, lee 5 / 2 / 1, walter 5 / 1 / 1 | pending: ingest ICSI (jane, dan, adam, morgan, liz) and more HPR |
| de | Küchenradio, 1.4 h | andi 9 / 4 / 1 (host) | pending: full Küchenradio run, media.ccc.de Q&A |
| fr | Libre à vous!, 3.8 h | florian 46 / 31 / 2, laurent 11 / 5 / 2, isabelle 10 / 6 / 3, gilles 4 / 4 / 2 | pending: full Libre à vous! run; "florian" is the strongest vocative candidate so far |
| es | KDE España + Espika FM, 5.3 h | fernando 21 / 8 / 2, juan 18 / 4 / 2, lucía 9 / 7 / 2, víctor 10 / 6 / 1, pancho 6 / 5 / 1 | pending: full runs plus Una Radio Muchas Voces |

First engine runs on data-derived names (Vosk grammar baseline, internal slice):
"andi" (de) 5 of 9 detected at 1,021 false accepts per hour; "víctor" (es) 5 of 10 at
304 false accepts per hour. This is the expected failure mode of an ASR grammar used as a
wake word engine and the reason the benchmark reports curves, not detection counts.

## Seed expectations from the source surveys

From the Multilingual Spoken Words Corpus counts (read speech, third-person mentions):

| Language | Frequent first names (clips / speakers) |
|---|---|
| en | michael 778 / 599 |
| de | michael 234 / 145, michaela 46 / 46 |
| fr | jean 2,761, pierre 1,684, louis 1,028, marie 985, paul 922, michel 587 |
| es | juan 603, maría 529, francisco 364, carlos 343, luis 294, antonio 266 |

From the ICSI meeting transcripts (English, genuine vocatives): jane 170, dan 137,
adam 135, morgan 131, liz 111, dave 108, chuck 101.
