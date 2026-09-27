# Openly licensed speech corpora for an EN+DE wake-word benchmark (status: 25 Sep 2026)

Scope: (a) hours of realistic negative speech (no wake word) to measure false accepts/hour (FA/h) in English and German; (b) natural in-sentence utterances of "Michael" (and Hey Jarvis / Alexa / Computer / Okay Nabu) by many real speakers. Everything below is non-synthetic. Counts marked "measured" were verified by downloading transcripts (VoxPopuli TSVs, Common Voice sentence pools, MSWC split CSVs).

Licence shorthand for redistribution of clips in a published benchmark: CC0 = yes; CC-BY = yes with attribution; CC-BY-SA = yes but derived set must be CC-BY-SA; NC/ND/"research only" = do not redistribute.

## 1. Large negative corpora (read + spontaneous), EN and DE

| Dataset | Licence / redistribution | EN | DE | Speakers, conditions | Transcripts | Download |
|---|---|---|---|---|---|---|
| Mozilla Common Voice Scripted Speech 27.0 (2026-09-11) | CC0. Since Oct 2025 only via Mozilla Data Collective (account + terms; data stays CC0). Older versions on HF and Kaggle. | measured: 2,591,456 clips, 3,795.9 h recorded / 2,826.8 h validated, 100,546 speakers | 1,019,724 clips, 1,491.9 h / 1,394.8 h validated, 20,558 speakers | Crowdsourced browser/phone mics, read sentences (~5.3 s), huge speaker/accent diversity, demographic metadata. Not conversational. | Sentence-level (`sentence` column in validated.tsv). No word timestamps. | https://mozilladatacollective.com ; https://github.com/common-voice/cv-dataset |
| Common Voice Spontaneous Speech 5.0 | CC0 | 5,898 clips / 642 speakers | only 382 clips / 30 speakers (measured) | Unscripted monologues | Transcribed | same |
| Common Voice 7.0 Single-Word segment | CC0 | digits, yes/no, "hey", "firefox" in 34 languages | | isolated words | word labels | https://datacollective.mozillafoundation.org/datasets/cmkzhp64p00wlno07elrmt20y |
| Multilingual LibriSpeech (MLS) | CC-BY-4.0 | 44,659 h | 1,966 h train, ~200 readers | LibriVox audiobooks | Segment level (10–20 s) | https://www.openslr.org/94/ ; https://huggingface.co/datasets/facebook/multilingual_librispeech |
| LibriSpeech | CC-BY-4.0 | 960 h + dev/test, 2,484 speakers | – | LibriVox | Utterance level; MFA word alignments available | https://www.openslr.org/12 ; https://github.com/CorentinJ/librispeech-alignments |
| Libriheavy / Libri-Light | CC-BY-4.0 | 50,000 h | – | LibriVox | Segment text with casing | https://github.com/k2-fsa/libriheavy |
| VoxPopuli | CC0 | 543 h transcribed (asr_en.tsv: 412,485 segments) | 282 h transcribed (asr_de.tsv: 189,549 segments) | European Parliament plenary, 4,295 speakers, formal spontaneous-ish | Segment text with start/end, speaker_id, gender | https://github.com/facebookresearch/voxpopuli ; https://huggingface.co/datasets/facebook/voxpopuli |
| The People's Speech | CC-BY-4.0 subset and CC-BY-SA subset | 30,000+ h | – | archive.org: meetings, legal proceedings, speeches; noisy, real-world; ~44 % of hours have >human WER | Segment level | https://huggingface.co/datasets/MLCommons/peoples_speech |
| Loquacious Set (SpeechBrain) | CC0 / CC-BY | 25,000 h mix; small (250 h) and medium (2,500 h) subsets | – | mix of read, talks, spontaneous | normalised text | https://huggingface.co/datasets/speechbrain/LoquaciousSet |
| GigaSpeech | Non-commercial research only, gated, no redistribution | 10,000 h | – | audiobooks, podcasts, YouTube | segments with times | https://huggingface.co/datasets/speechcolab/gigaspeech |
| YODAS / YODAS2 (ESPnet) | CC-BY-3.0 (YouTube CC-BY videos) | manual captions ≈ 29,155 h | de000 (manual captions) 3,064 h; auto ≈ 10,949 h | YouTube vlogs, talks, podcasts | Segment timestamps + caption text | https://huggingface.co/datasets/espnet/yodas ; https://huggingface.co/datasets/espnet/yodas2 |
| YODAS-Granary / NVIDIA Granary | CC-BY-3.0 | 40.8 M segments | 3.75 M segments | YODAS2 with Whisper-large-v3 pseudo-labels | segment level | https://huggingface.co/datasets/nvidia/Granary |
| Emilia / Emilia-YODAS | Emilia: CC-BY-NC-4.0 (internal only). Emilia-YODAS: CC-BY-4.0. Gated. | 92.2k h (CC-BY part) | 5.6k h (CC-BY part) | Podcasts, talk shows, interviews; the best large DE spontaneous source | Utterance-level Whisper transcripts | https://huggingface.co/datasets/amphion/Emilia-Dataset |
| MOSEL (FBK) | OSI-compliant sources only | yes | yes | catalogue + pseudo-labels | pseudo-labels | https://huggingface.co/datasets/FBK-MT/mosel |
| AMI Meeting Corpus | CC-BY-4.0 | 100 h, mostly non-native English | – | Real/scenario meetings; headset and far-field arrays | Word-level timestamps and speakers | https://groups.inf.ed.ac.uk/ami/corpus/ ; https://huggingface.co/datasets/edinburghcstr/ami |
| CHiME-5 / CHiME-6 | CC-BY-SA-4.0 since 2024 | ≈50.6 h | – | Dinner parties in real homes, 6 Kinect arrays, heavy overlap – ideal far-field negatives | JSON utterances per speaker | https://openslr.org/150/ |
| DiPCo (Amazon) | CDLA-Permissive | ~5.3 h, 10 sessions | – | far-field 7-mic arrays, dining table (the FA/h set used by openWakeWord/microWakeWord) | transcripts with times | https://zenodo.org/records/8122551 |
| TED-LIUM 3 | CC-BY-NC-ND-3.0 – no redistribution | 452 h | – | stage talks | STM | https://huggingface.co/datasets/LIUM/tedlium |
| Spotify Podcast Dataset | Discontinued | – | – | – | – | https://podcastsdataset.byspotify.com/ |
| Open Yap 1K | sample 8.9 h CC-BY-4.0; full 1,000 h under DUA | 1,000 h real two-party phone calls | – | genuinely conversational | transcribed | https://huggingface.co/blog/TheAgenticDataCompany/open-yap-1k |
| FLEURS | CC-BY-4.0 | ~12 h | ~12 h | read Wikipedia | sentence | https://huggingface.co/datasets/google/fleurs |
| YouTube-Commons (PleIAs) | CC-BY (transcripts; audio from YouTube) | 71 % EN | DE available | – | transcripts only | https://huggingface.co/datasets/PleIAs/YouTube-Commons |

## 2. German-specific corpora

| Dataset | Licence | Hours / speakers | Conditions | Transcripts | Download |
|---|---|---|---|---|---|
| Tuda-De | CC-BY | ~36 h per mic, 180 speakers; 4 parallel channels incl. Kinect (far-field) | controlled room; Wikipedia, parliament, commands | sentence-level XML | https://github.com/uhh-lt/kaldi-tuda-de |
| Spoken Wikipedia Corpus (SWC) DE/EN | CC-BY-SA-4.0 | DE 386 h, 339 readers, 249 h word-aligned; EN 395 h | volunteer home recordings, name-dense text | word- and phone-level timestamps | https://nats.gitlab.io/swc/ |
| lumaku/german-corpus-aligned | audio = LibriVox PD + SWC | LibriVox DE 804 h / 251 speakers | read | CTC-segmentation alignments | https://github.com/lumaku/german-corpus-aligned |
| HUI-Audio-Corpus-German | CC0 | 326 h, 122 speakers | audiobook | sentence-level | https://github.com/iisys-hof/HUI-Audio-Corpus-German |
| M-AILABS German | free, PD sources | 237 h, 29 speakers | LibriVox | sentence | https://github.com/imdatceleste/m-ailabs-dataset |
| Thorsten-Voice / CSS10-DE | CC0 / Apache-2.0 | 23 h / 17 h, single speaker | studio | sentence | https://www.openslr.org/95/ |
| ASR Bundestag (Hof) | NOT CC (Bundestag terms) – internal only | 610 h clean + 156 h dirty | plenary | sentence aligned | https://opendata.iisys.de/dataset/asr-bundestag/ |
| EuroSpeech (NeurIPS 2025) | per-parliament terms | DE ≥1k h | parliament | CER-filtered | https://github.com/SamuelPfisterer/EuroSpeech |
| BAS (LMU) corpora | licensed; transfer to third parties prohibited | Verbmobil = real spontaneous DE dialogues | – | word-level (partly) | https://www.bas.uni-muenchen.de/Bas/BasKorporaeng.html |
| GECO, GRASS | research licences | dialogues | spontaneous | word-aligned | https://www.ims.uni-stuttgart.de/en/research/resources/corpora/ims-geco/ |

There is no CC-licensed German conversational corpus of the AMI/CHiME type. For DE spontaneous negatives the realistic options are Emilia-YODAS DE (5.6k h, CC-BY-4.0), YODAS de000 (3,064 h, CC-BY-3.0), VoxPopuli DE (282 h CC0) and the tiny CV Spontaneous Speech DE.

## 3. Keyword / wake-word corpora (positives)

| Dataset | Licence | Content | Notes | URL |
|---|---|---|---|---|
| Multilingual Spoken Words Corpus (MSWC) | CC-BY-4.0 | 1-s clips cut from Common Voice by forced alignment; EN 6.6 M clips, DE 3.7 M clips | Measured (clips / distinct speakers): EN "michael" 778 / 599, DE "michael" 234 / 145, DE "michaela" 46 / 46, "michel" EN 44 / DE 57; "computer" EN 1,427 / 1,147, DE 516 / 363; "hey" EN 2,482, DE 53; "okay" EN 564, DE 152; "alexa" EN 7, DE 15; "jarvis" EN 17, DE 0; "nabu"/"siri" 0. The LINK column maps each clip back to the full Common Voice sentence clip, giving both the isolated word and the in-sentence recording. | https://mlcommons.org/datasets/multilingual-spoken-words/ ; https://huggingface.co/datasets/MLCommons/ml_spoken_words |
| Google Speech Commands v2 | CC-BY-4.0 | 35 words, 105,829 clips, 2,618 speakers | no names | https://huggingface.co/datasets/google/speech_commands |
| Picovoice wake-word-benchmark | Apache-2.0 | alexa 329, computer 411, jarvis 384, snowboy 401, smart mirror, view glass; >50 speakers | de-facto public positive set for "Alexa"/"Computer"/"Jarvis" | https://github.com/Picovoice/wake-word-benchmark ; https://huggingface.co/datasets/Picovoice/wake-word-benchmark |
| Hey Snips (Sonos) | research only, no redistribution | 11k "Hey Snips" + 86.5k negatives | on request | https://github.com/sonos/keyword-spotting-research-datasets |
| Qualcomm Hey Snapdragon | research only | 4,270 recordings | – | via https://picovoice.ai/blog/open-source-keyword-spotting-data/ |
| OVOS community wake-word dataset | mixed per contributor | 1,687 real clips: hey-mycroft, computer, ... | per-wake-word bench splits incl. MSWC-negative sets for de-DE/en-US | https://huggingface.co/datasets/OpenVoiceOS/ovos-community-wakewords-dataset ; https://huggingface.co/collections/Jarbas/wake-word-datasets |
| HA Wake Word Collective ("Okay Nabu") | promised CC0 | >5,800 samples in 30 languages | not publicly released as of Sept 2026; ask OHF/Nabu Casa | https://ohf-voice.github.io/wake-word-collective/ |
| openWakeWord data | features only (CC-BY-NC-SA-4.0) | "hey jarvis"/"hey mycroft" test clips = author's own recordings, not released | Real "Hey Jarvis" multi-speaker recordings do not exist publicly | https://huggingface.co/datasets/davidscripka/openwakeword_features |
| microWakeWord data | spectrogram features only (CC-BY-NC-4.0) | not raw audio | | https://huggingface.co/datasets/kahrendt/microwakeword |

## 4. Tooling to find word occurrences with timestamps in long audio

| Tool | Method | Languages | Notes |
|---|---|---|---|
| Montreal Forced Aligner 3.x | GMM-HMM + dictionary | `english_mfa`, `german_mfa` v3.0.0 (CC-BY-4.0) | Most accurate word boundaries; needs transcript; pre-segment long noisy files. https://montreal-forced-aligner.readthedocs.io |
| WhisperX | Whisper + wav2vec2 CTC alignment | EN, DE | End-to-end "grep any audio"; timestamps <100 ms but less precise than MFA. https://github.com/m-bain/whisperX |
| ctc-forced-aligner / torchaudio MMS_FA | CTC forced alignment with MMS-300M | EN, DE | Simplest multilingual aligner; MMS_FA model is CC-BY-NC-4.0 (tooling only). https://github.com/MahmoudAshraf97/ctc-forced-aligner |
| NeMo Forced Aligner + Parakeet-TDT-0.6B-v3 | native word timestamps for 25 European languages incl. DE | EN, DE | Handles 1 h+ files. https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3 |
| whisper-timestamped, CrisperWhisper | Whisper cross-attention/DTW | multilingual | https://github.com/linto-ai/whisper-timestamped |

Recommended pipeline: grep corpus transcripts (or run Parakeet/Whisper on untranscribed audio) for the target word -> cut a 10–20 s window -> run MFA or ctc-forced-aligner on the window with the known text -> store word start/end -> human spot-check (names are frequently mis-transcribed: "Michel", "Michaela", "Mikhail").

## 5. Measured "Michael" statistics

| Source | Measured | Rate |
|---|---|---|
| VoxPopuli DE (~485 h incl. invalid split) | 60 segments contain "Michael" (102 tokens), 38 distinct speakers | ~0.12 segments/h |
| VoxPopuli EN (~1,271 h) | 185 segments, 84 distinct speakers | ~0.15/h |
| Common Voice DE sentence pool (2,094,447 sentences) | 1,909 sentences contain "Michael"; "Computer" 575; "Alexa" 24; "Jarvis" 14 | 0.09 % |
| Common Voice EN sentence pool (1,581,123 sentences) | 2,218 "Michael"; "Computer" 1,737; "Jarvis" 76; "Alexa" 27 | 0.14 % |
| MSWC (cut from CV ≈ v6/7) | EN "michael" 778 clips / 599 speakers; DE 234 / 145 | – |

Extrapolations (not measured): Common Voice 27.0 today should hold roughly 1,500–3,000 EN and 500–1,200 DE clips with "Michael" in a sentence, from >1,000 EN and 200–500 DE speakers. MLS DE ~100–400 tokens; MLS EN ~2,000–6,000; SWC DE/EN (word-aligned already) ~200–600 each; Emilia-YODAS DE ~500–1,500 and EN tens of thousands.

## 6. Recommended shortlist for an EN+DE benchmark

Negative set (FA/h), redistributable:
1. EN far-field conversation: CHiME-6 dev+eval (~10 h, CC-BY-SA-4.0) and DiPCo (5.3 h, CDLA-Permissive); AMI sdm (far-field) 20–30 h, CC-BY-4.0.
2. EN read/crowdsourced: Common Voice EN validated sample 20–50 h (CC0), LibriSpeech test-clean/other (Picovoice convention, CC-BY-4.0).
3. EN spontaneous/noisy: People's Speech CC-BY clean sample (10–20 h), Emilia-YODAS EN or VoxPopuli EN 10–20 h.
4. DE: Common Voice DE validated sample (CC0), VoxPopuli DE (CC0), Emilia-YODAS DE / YODAS de000 (CC-BY, spontaneous podcasts), Tuda-De Kinect channel (CC-BY, the only DE far-field set), SWC DE (CC-BY-SA), MLS DE (CC-BY).
Avoid for redistribution: GigaSpeech, TED-LIUM 3, Emilia (non-YODAS), Hey Snips, BAS corpora, ASR Bundestag, openWakeWord/microWakeWord feature sets.

Positive "Michael" set: (a) MSWC "michael" clips (EN 778 / DE 234) and their parent Common Voice sentence clips via the LINK id – ~1,000 in-sentence utterances from ~750 speakers immediately; (b) grep Common Voice 27.0 validated.tsv for "Michael" (est. 2,000–4,000 clips EN+DE) and align with MFA; (c) VoxPopuli 245 segments from 122 speakers (CC0, real spontaneous parliamentary speech); (d) SWC DE/EN with ready word timestamps; (e) Emilia-YODAS DE/EN for conversational contexts. Realistic total: 3,000–6,000 natural in-sentence "Michael" utterances (DE ~1,000–2,000; EN ~2,000–4,000) from well over 1,000 speakers, entirely under CC0/CC-BY/CC-BY-SA. Note: nearly all are read speech and "Michael" is followed by a surname, so also include VoxPopuli/Emilia items for vocative use.

Other wake words: "Alexa"/"Computer"/"Jarvis" -> Picovoice benchmark plus MSWC "computer"; "Hey Jarvis" and "Okay Nabu" -> no public real multi-speaker recordings exist as of Sept 2026; plan your own collection or ask OHF/Nabu Casa.
