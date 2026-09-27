# Openly licensed conversational speech sources (DE + EN) — Sept 2026

"Verified" = read from the source page/feed/API in the research session. Safe = CC BY / CC BY-SA / CC0 / CDLA-Permissive (redistributable with attribution; SA requires same licence on derived set). Internal = NC/ND/research-only, usable for own evaluation but not redistributable.

## 1. Podcasts under Creative Commons

### Licence metadata in feeds
- `<podcast:license>` (Podcasting 2.0): https://podcasting2.org/docs/podcast-namespace/tags/license — Podlove Podcast Publisher (Metaebene, many German podcasts) emits it.
- Legacy `<creativeCommons:license>` (TWiT).
- Podcast Index API: no licence filter (https://github.com/Podcastindex-org/podcast-namespace/discussions/603). fyyd.de: no licence filter.
- Internet Archive is the only directory filterable by licence: `collection:podcasts AND licenseurl:"..."` via https://archive.org/advancedsearch.php.
- Wikipedia Category:Creative_Commons-licensed_podcasts: seed list, re-verify each.
- Recipe: crawl feeds, parse `podcast:license`, `creativeCommons:license`, `<copyright>`, site footer.

### German podcasts (licence verified)

| Podcast | Licence | Format | Vocatives | Status |
|---|---|---|---|---|
| Küchenradio https://www.kuechenradio.org | CC BY 3.0 DE (site) | Round-table of friends in a Berlin kitchen, ~500 eps since 2005 | High | **Safe. Best German CC-BY conversational podcast.** |
| Forschergeist https://forschergeist.de | CC BY-SA 4.0 (feed + impressum) | 1:1 interviews, ~70 eps × 1–2 h | Moderate | Safe (SA) |
| Der Lautsprecher https://der-lautsprecher.de | CC BY 3.0 DE (feed) | 2–3 hosts, ~30 eps | Yes | Safe |
| Radio Tux https://blog.radiotux.de | CC BY-SA 3.0 (feed) | Multi-host Linux talk | Yes | Safe (SA) |
| CRE https://cre.fm | CC BY-NC-SA 3.0 DE | Long interviews, ~230 eps | Moderate | Internal |
| Logbuch:Netzpolitik | CC BY-NC-SA 3.0 DE | Two hosts weekly, 560 eps, WebVTT auto transcripts | High | Internal |
| Freak Show https://freakshow.fm | CC BY-NC-ND 3.0 DE | 4–6 people round-table, 311 eps, WebVTT | High | Internal |
| Raumzeit | CC BY-NC-ND 3.0 DE | Interviews | Moderate | Internal |
| UKW https://ukw.fm | CC BY-NC 4.0 | Political talk | Yes | Internal |
| Chaosradio | "All Rights Reversed (K)" kopimi, no formal CC | Live radio talk | High | Unclear, do not redistribute |
| Rechtsbelehrung | CC BY-ND 4.0 DE | Two hosts | Yes | Internal (ND) |
| Methodisch inkorrekt | "CC 3.0" module unspecified | Two hosts | High | Unverifiable |
| Bits und so, WRINT, Alternativlos, Sternengeschichten, Hoaxilla, Fokus Europa, NSFW | No licence found | | | Skip |

### English podcasts (verified)

| Podcast | Licence | Format | Vocatives | Status |
|---|---|---|---|---|
| Hacker Public Radio https://hackerpublicradio.org | CC BY-SA 4.0 | 1 ep per weekday since 2005, home-recorded, kitchens/cars/outdoors, many multi-host chats; 4,441 items on IA | Moderate–high | **Safe (SA). Best English pool.** |
| Bad Voltage https://www.badvoltage.org/about/ | CC BY-SA | 3 hosts banter since 2013 | High | Safe (SA) |
| GNU World Order | CC BY-SA 4.0 | Single host | Low | Safe, filler |
| Linux Matters | CC BY-NC 4.0 | 3 hosts | High | Internal |
| TWiT network | CC BY-NC-ND 4.0 | Multi-host round tables | High | Internal |
| Democracy Now! | CC BY-NC-ND | News + interviews | Low | Internal |
| No Agenda | No licence found | | | Skip |
| Radiolab/WNYC, NPR | Not CC | | | Not usable |

## 2. media.ccc.de (verified via API)
- Licence per talk, typically CC BY 4.0 (https://media.ccc.de/about); API has no licence field, check conference policy / file outro.
- Volume: 461 conferences, 17,147 recordings, ~11,750 h; EN ~6,430 h (9,230 talks), DE ~5,290 h (7,880 talks). Largest: rc3 257 h, 36c3 213 h, 38c3 171 h, 39c3 145 h.
- Conversational parts: Q&A segments after nearly every talk, panels, Sendezentrum podcast stage, Chaosradio live, lightning talks. Hall reverb, audience noise, applause.
- Subtitles: c3subtitles (https://c3subtitles.de, https://c3voc.de/wiki/subtitles), SRT/VTT exposed in API for a subset.
- Bulk: yt-dlp CCC extractor; https://github.com/muesli/sync3c; CDN https://cdn.media.ccc.de; API https://api.media.ccc.de/public/conferences, /public/conferences/{id}, /public/events/{guid} (audio-only opus/mp3 + subtitles).

## 3. Wikimedia Commons (verified via API)
- Query by licence: `list=search&srnamespace=6&srsearch=filetype:audio haswbstatement:P275=Q20007257` (Q20007257 = CC BY 4.0, Q18199165 = CC BY-SA 4.0, Q6938433 = CC0, Q14946043 = CC BY-SA 3.0); language `haswbstatement:P407=Q188` (German), Q1860 (English).
- 75,903 CC BY 4.0 audio files (overwhelmingly Lingua Libre single words); 27,560 German audio (Lingua Libre + spoken articles) — not useful.
- Conversational: Wikimania talk videos 535 files ≈ 260 h, CC BY-SA, mostly EN, incl. Q&A/panels. Category:Audio files of interviews 108 files ≈ 28 h mixed. Wikitongues ~600 videos ≈ 34 h, only CC BY-SA opt-ins on Commons, self-narrated monologue, almost no German. Category:Videos in German 630 files ~130 h, mostly how-to/event footage.
- Verdict: Wikimania (EN) is the only sizeable block; German conversational audio on Commons is negligible.

## 4. YouTube CC-BY
- YouTube CC option (https://support.google.com/youtube/answer/2797468): CC BY (3.0 link); only original content; no Content ID claims.
- Caveats: uploader may not own rights; YouTube ToS forbid downloading outside the service (CC licence is rights-holder permission, not YouTube's); videos vanish; attribution to thousands of channels.
- YODAS/YODAS2 (https://huggingface.co/datasets/espnet/yodas): CC BY 3.0; de000 manual 3,064 h, de100–102 auto ≈ 10,950 h, en000–005 manual ≈ 29,155 h. **audio_ids do not resolve to live YouTube IDs**, so per-video attribution/re-verification impossible. Genre from de000 shard 0 (61 videos): Hessian state parliament sessions, software tutorials, maths explainers, vlogs, Let's Play, church devotionals; 70–80 % monologue; formal "Herr/Frau + surname" common, first-name vocatives rare.
- Emilia-YODAS (https://huggingface.co/datasets/amphion/Emilia-Dataset): CC BY, DE 5.6k h; but pipeline denoises, source-separates, filters overlapped/noisy segments and cuts into short single-speaker clips, so natural background and turn-taking are removed. Same ID obfuscation.
- YouTube-Commons (PleIAs): CC BY 4.0 transcripts only; audio must be re-fetched (ToS risk).
- Granary (NVIDIA): CC BY 4.0, ~1M h, 25 languages, `de_yodas`, `de_voxpopuli`, `en` configs; pseudo-labelled.
- Verdict: CC-BY negative material at scale, but attribution defective; mark as internal/research unless dataset-level attribution accepted.

## 5. Internet Archive, Europeana, German research corpora
- IA audio by licence: CC BY 4.0 10,615; CC BY 3.0 67,379; CC BY-SA 4.0 4,269; CC BY-SA 3.0 53,689; CC0 75,467. German CC audio ~270 items, thin. Hacker Public Radio collection is the standout. Query: `mediatype:audio AND licenseurl:"..." AND language:(ger OR German OR deu)`.
- Europeana: only 7 German-language open sound records. DDB: no CC audio of note.
- FOLK (IDS Mannheim, https://agd.ids-mannheim.de/folk.shtml): ~400 h everyday German conversation with first names, exactly the right register, but research-and-teaching only, no download, no CC. GeWiss, GRASS: research only. CLARIN German entries: only JuBe (CC BY 4.0) open.
- Conclusion: no CC-licensed German everyday-conversation research corpus exists.

## 6. Meeting corpora with vocatives (verified by downloading annotations)

| Corpus | Licence | Hours | Vocatives | Status |
|---|---|---|---|---|
| ICSI Meeting Corpus https://groups.inf.ed.ac.uk/ami/icsi/ | CC BY 4.0 | ~72 h, 76 meetings, transcripts | Very high: Jane 170, Dan 137, Adam 135, Morgan 131, Liz 111, Dave 108, Chuck 101 tokens; clear vocative forms hundreds of times | **Safe. Best open source of natural first-name vocatives in English.** |
| AMI https://groups.inf.ed.ac.uk/ami/corpus/ | CC BY 4.0 | 100 h, NXT words, PERSON/PARTICIPANT named-entity layer (~1,675 person mentions), far-field arrays | Yes (kate 28, david 34, christine 21, ...) | Safe |
| CHiME-5/6 | University of Sheffield licence; free non-commercial, commercial GBP 2,000 | ~40 h dinner parties in homes | High | **Not redistributable** |
| DiPCo https://zenodo.org/records/8122551 | CDLA-Permissive 1.0 | ~5 h, far-field dinner | Moderate | Safe |
| NOTSOFAR-1 https://github.com/microsoft/NOTSOFAR1-Challenge | CC BY 4.0 | ~24 h real meetings, 30 rooms, 35 speakers, multi-device | Moderate | Safe |
| SBCSAE (TalkBank) | no CC statement; names altered in transcripts | ~20 h everyday US conversation | High | Internal |
| CANDOR | data request, no CC | 850 h video calls | High | Internal |
| Fisher, Switchboard, CallHome (incl. German), Mixer 6 | LDC | | | Not open |

No public corpus marks the addressee explicitly; derive vocatives by regex (name at utterance start/end followed by comma/question mark) on ICSI/AMI.

## 7. Real-world background and mixing sets (licences verified)
- In-home with TV/music: CHiME-5/6 (Sheffield licence), DiPCo (CDLA, no TV), VOiCES https://iqtlabs.github.io/voices/ (CC BY 4.0; read LibriSpeech replayed through loudspeakers in 3 rooms with TV/music/babble/HVAC distractors), CHiME-Home (CC BY-NC-SA 3.0, internal).
- Mixing: DEMAND https://zenodo.org/records/1227121 (CC BY 4.0; kitchen, living room, washing machine, café...), MUSAN https://www.openslr.org/17/ (CC BY 4.0; ~109 h music/speech/noise), WHAM! noise (CC BY-NC 4.0, internal), FSD50K https://zenodo.org/records/4060432 (clip-level: CC0 19,873, CC BY 23,506, CC BY-NC 6,041 — filter), FMA (per-track licence, filter), AudioSet (labels only, audio from YouTube), Freesound (API filter by licence).

## Ranked shortlist for a pooled EN+DE conversational set
1. ICSI Meeting Corpus (EN, CC BY 4.0, 72 h, dense first-name vocatives) — Safe
2. AMI (EN, CC BY 4.0, 100 h, far-field) — Safe
3. media.ccc.de (DE ~5,300 h + EN ~6,400 h, CC BY 4.0 per talk; Q&A/panel segments) — Safe, check per talk
4. Hacker Public Radio (EN, CC BY-SA 4.0, 4,441 eps, home recordings) — Safe (SA)
5. Küchenradio (DE, CC BY 3.0 DE, ~500 round-table eps) — Safe
6. Forschergeist + Der Lautsprecher + Radio Tux (DE) — Safe
7. DiPCo (EN, CDLA, 5 h far-field dinner) — Safe
8. NOTSOFAR-1 (EN, CC BY 4.0, 24 h meetings) — Safe
9. Bad Voltage + GNU World Order (EN, CC BY-SA) — Safe (SA)
10. Wikimania talks on Commons (EN, CC BY-SA, ~260 h) — Safe (SA)
11. Emilia-YODAS / YODAS2 / Granary — Internal/research unless dataset-level attribution accepted

Background mixing (Safe): DEMAND, MUSAN, FSD50K CC0/CC-BY, FMA CC-BY, Freesound CC0/CC-BY, VOiCES.
Internal-only conversational: Freak Show, Logbuch:Netzpolitik, CRE, UKW, TWiT, Linux Matters, Democracy Now, Emilia, CHiME-6, FOLK/GeWiss/GRASS.

Gap: no CC-licensed German in-home multi-party corpus with TV/music exists. Closest redistributable approximation: Küchenradio + media.ccc.de Q&A for German, mixed with DEMAND/FSD50K domestic noise; ICSI/AMI/DiPCo for the English vocative-rich far-field component.
