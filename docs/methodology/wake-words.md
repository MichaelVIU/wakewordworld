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

| Language | Pool status | Candidates | Decision |
|---|---|---|---|
| en | first slice ingesting | pending | pending |
| de | first slice ingested (Küchenradio, 1.4 h) | pending | pending |
| fr | first slice ingesting | pending | pending |
| es | first slice ingesting | pending | pending |

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
