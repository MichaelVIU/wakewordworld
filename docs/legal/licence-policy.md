# Licence policy

## Tiers

| Tier | Meaning | Examples | What is published |
|---|---|---|---|
| A | Redistributable inside the benchmark dataset, with attribution; share-alike honoured | CC0, CC BY (any version), CC BY-SA (any version), CDLA-Permissive, public domain dedication, MIT/Apache-style terms on recordings | Audio, manifest row, transcript, results |
| B | Internal evaluation only | CC BY-NC*, CC BY-ND*, research-only, click-through, Public Domain Mark without a dedication, unlicensed, unversioned "CC BY" | Manifest row (source, URL, duration, tags), transcript word index without audio, results |
| forbidden | Not ingested | Terms that prohibit text/data mining or AI use, YouTube downloads outside curated CC corpora, LDC/ELRA | Nothing |

The implementation of this table is `src/wakewordworld/licences.py`; it fails closed:
anything not on the explicit tier-A list is tier B.

## Evidence

A source or item is tier A only if the licence is backed by evidence: a feed tag, a
sentence on the publisher's page, an API field, a dataset card, or an e-mail. The
evidence (verbatim quote, URL, capture date) is stored in the source spec and, for
per-item licences, in the item record.

Public Domain Mark on a publisher's own work is a statement, not a dedication. Such
items stay tier B until the publisher confirms CC0 in writing.

## Share-alike

Chunks derived from CC BY-SA recordings are published under the same CC BY-SA version in
a separate dataset configuration. Mixing CC0, CC BY and CC BY-SA in one download is
avoided by keeping one configuration per licence family.

## Attribution

Every published chunk carries an attribution string rendered from the source template
(title, author, URL, licence). The dataset card lists every source with its licence and
link. For large aggregated sources (Internet Archive, PeerTube, Commons) the per-item
attribution is in the manifest.

## Dataset terms (gated access)

Downloading the public dataset requires accepting: attribution as recorded in the
manifest, no use of the audio to train wake word or keyword spotting models, and
acknowledgement that the data contains a canary string. The canary is
`WWW-CANARY-7f3a9c2e-4b1d-4e8a-9f6b-2c5d8e1a3b7f`.

## Takedown

Anyone may request removal of a recording by opening an issue or e-mailing the
maintainers. The recording is removed from the working set immediately and tombstoned in
the next manifest release (the row stays, marked removed, audio deleted). Results are
recomputed at the next scheduled run.

## Privacy

Only recordings that are already public are used. Speaker identities are not linked
across sources. Names in the audio are treated as words. No demographic data is inferred.
