# Wake Word / KWS Benchmarks and Evaluation Methodology: State of the Art as of September 2026

## 1. Executive summary

- There is still no neutral, maintained, multi-engine wake word benchmark. Every widely cited number comes from either (a) a vendor benchmarking itself against dead or misconfigured baselines, (b) an open-source project evaluating its own models with small, self-chosen test sets, or (c) academic datasets that measure isolated-word classification rather than always-on detection. VoxRT's own comparison page states the situation bluntly: "Every accuracy and speed figure above is vendor-self-reported, and no independent academic benchmark covers all of these engines on a common test set" (https://voxrt.com/wake-word-comparison).
- The metric vocabulary has converged (false rejects at a fixed false-accepts-per-hour budget, DET/ROC curves), but the test data has not. Picovoice uses LibriSpeech + DEMAND at 10 dB SNR; openWakeWord uses the Dinner Party Corpus (DiPCo) for false accepts; microWakeWord uses both; livekit uses its own undisclosed 25 h validation set; DaVoice quotes customer anecdotes. Numbers are therefore not comparable across projects.
- The one genuinely independent, code-and-data-open, multi-engine comparison found is brand new: the Jaco-Assistant "Benchmark-KeywordSpotting" GitLab project (created 15 Sep 2026, posted to openWakeWord issue #349 on 17 Sep 2026). It reuses Picovoice's protocol and data, adds a DiPCo background condition, evaluates Porcupine, openWakeWord, microWakeWord, local-wake and an ASR-based spotter, and documents its own caveats. It is one person's work, covers one main keyword ("alexa"), and uses an old Porcupine build, but it is the closest thing to a clean comparison that exists (https://gitlab.com/Jaco-Assistant/Benchmark-KeywordSpotting).
- The gaps are concrete and consistent across sources: no shared negative corpus of many hours of realistic speech/media; no confusable/near-miss negative set; no SNR or distance sweeps; no accent/gender/age stratification; no reporting of full curves (only a single operating point); no version/threshold pinning; no independence from the entities being measured.

## 2. Vendor benchmarks

### 2.1 Picovoice wake-word-benchmark
Sources: https://github.com/Picovoice/wake-word-benchmark ; https://picovoice.ai/docs/benchmark/wake-word/ ; https://picovoice.ai/blog/benchmarking-a-wake-word-detection-engine/ ; https://picovoice.ai/blog/wake-word-benchmarks/

Methodology (the de facto reference protocol):
- Positives: 300+ crowdsourced recordings of six keywords (alexa, computer, jarvis, smart mirror, snowboy, view glass) from 50+ speakers; repackaged on Hugging Face as 2,293 clips, ~213 MB, Apache 2.0 (https://huggingface.co/datasets/domdomegg/picovoice-wake-word-benchmark).
- Negatives/background: LibriSpeech test_clean (read audiobook speech), keywords inserted at intervals into a long file (roughly 24 h).
- Noise: DEMAND dataset, 18 environments, mixed at a single fixed 10 dB SNR.
- Metrics: miss rate at a fixed false alarm budget of 1 per 10 hours; runtime as real-time factor and CPU on Raspberry Pi 5 (32-bit).
- Engines: Porcupine, Snowboy, PocketSphinx only. Published result: Porcupine 97.3% detection vs Snowboy 68.1% vs PocketSphinx 48.0% at 1 FA/10 h, 10 dB SNR.

Criticisms and bias signals:
- Baselines are dead: Snowboy archived 2020, PocketSphinx is CMU-Sphinx era, Mycroft Precise dormant since 2019. openWakeWord, microWakeWord, livekit-wakeword and every commercial competitor are absent. A September 2025 issue asking for openWakeWord (#17) was closed with no visible reply (https://github.com/Picovoice/wake-word-benchmark/issues/17).
- Configuration fairness has been disputed: issue #4 flagged that Snowboy must run with ApplyFrontend=True; Picovoice's own runtime README admits PocketSphinx timing was measured via its CLI and is "essentially an upper bound".
- Issue #13 (2024) reports up to 10 percentage points of variance in Porcupine's true positive rate depending only on the temporal spacing of inserted keywords, with no maintainer response (https://github.com/Picovoice/wake-word-benchmark/issues/13).
- Issue #11 questioned overlap between the "alexa" test recordings and public Kaggle data (training/test leakage risk).
- Single SNR, single background domain (read speech, not conversational or media audio), no far-field/reverberation, no confusables, no per-speaker breakdown.
- The Jaco project ported Picovoice's mixer and notes that labels span the whole inserted recording + 0.5 s rather than the spoken word, so "a detection counts as a hit even when the audio it came from holds no part of the keyword"; enforcing word-containment costs Porcupine roughly 10 points of recall. Absolute Picovoice-protocol numbers are therefore optimistic.

### 2.2 Other vendor comparisons (all self-reported, none reproducible)
- Sensory / Vocalize.ai (https://sensory.com/revisiting-wake-word-accuracy-and-privacy/): "Alexa" trigger, 24 h of "random noise", TrulyHandsfree 250 KB model 0 false accepts vs Amazon 13; FRR 3% vs 10% in babble. No data released.
- DaVoice (https://davoice.io/benchmark): two customer anecdotes, threshold 0.99, "zero false positives in a month"; competitors unnamed; openWakeWord scored 62-69% detection but was "not tested for false positives because its true positive rate was too low". No controlled sweeps.
- VoxRT (https://voxrt.com/wake-word-comparison): ROC AUC 0.9966 on 5,240 positives / 6,416 "hard negatives" for its own phrase; acknowledges everything is self-reported.
- ViolaWake (https://violawake.com/compare/picovoice/): quotes 0.8% EER on its reference model.
- Outspoken (https://outspoken.cloud/blog/best-wake-word-tools): feature-table comparison of 9 tools, no measurements.
- livekit-wakeword (https://livekit.com/blog/livekit-wakeword ; https://github.com/livekit/livekit-wakeword): claims 0.08 vs 8.50 false positives/hour and 86% vs 69% recall versus openWakeWord on a "hey livekit" validation set (15,000 positives, 45,084 negatives, 25 h). Test set undisclosed; openWakeWord "optimal threshold 0.01" means no threshold could meet the FPPH target, i.e. the baseline was not tuned to a comparable operating point.

## 3. Open-source engine self-evaluations

### 3.1 openWakeWord (dscripka)
Source: https://github.com/dscripka/openWakeWord (README "Performance and Evaluation")
- Targets: FRR < 5% at < 0.5 false accepts/hour.
- False accepts: measured on DiPCo (Dinner Party Corpus), ~5.5 h of far-field 4-person dinner conversation with music (https://arxiv.org/pdf/1909.13447 ; https://zenodo.org/records/8122551).
- False rejects: clean recordings mixed with noise at 5-10 dB SNR and convolved with room impulse responses, or manually collected far-field recordings.
- README itself says "sample sizes are small and there are issues with the evaluation of the other libraries ... results should be interpreted cautiously".
- Discussion #135: a user model with 0.62 FA/h on the benchmark still triggering ~every 2 h with TV/radio; maintainer replied that "those metrics are only directional" (https://github.com/dscripka/openWakeWord/discussions/135). Issue #350 (2026): default threshold 0.5 yielding 48% detection in loud noise vs 83% at 0.25 with zero false positives on a 5-minute negative test (https://github.com/dscripka/openWakeWord/issues/350).

### 3.2 microWakeWord (Kevin Ahrendt / OHF-Voice, ESPHome)
Sources: https://github.com/OHF-Voice/micro-wake-word ; https://github.com/OHF-Voice/micro-wake-word/releases ; https://www.home-assistant.io/blog/2024/06/26/voice-chapter-7/
- Training-time FA/h estimated by slicing long ambient clips into spectrograms at 100 ms stride; README admits this "is not a perfect estimate of the streaming model's real-world false accepts per hour".
- Test sets: "testing_ambient" negatives plus DiPCo and the Picovoice benchmark.
- v2.1 (July 2024): default cutoffs "at most 0.16 false accepts per hour on DipCo and less than 0.1 FA/h on the PicoVoice benchmark".
- Authors state they cannot compare against commercial engines because FA/h numbers are kept as trade secrets.
- Detection is a sliding-window average over per-10 ms scores, not a raw threshold; the Jaco project found a 10 ms vs 20 ms frame-step units bug that initially produced a bogus ~99% miss rate.

### 3.3 Home Assistant / Nabu Casa
Sources: https://www.home-assistant.io/voice_control/about_wake_word/ ; https://www.home-assistant.io/blog/2024/10/24/wake-word-collective/ ; https://community.home-assistant.io/t/ok-nabu-wake-word-reliability-for-non-native-speakers-in-home-assistant-voice-preview/883494 ; https://community.home-assistant.io/t/openwakeword-vs-microwakeword/1000318
- HA has published no head-to-head numbers.
- Wake Word Collective (Oct 2024): 5,800+ "okay nabu" samples in 30 languages; retrained model FRR 5% vs 18% before, CC0 licensed. Only public, consented, accent-diverse positive corpus for a current open wake word (not yet released as dataset as of Sept 2026).
- Community threads document "tons of false positives" (hey jarvis, TV), gender/age dependence; users asking in March 2026 for openWakeWord vs microWakeWord comparison with none available.

### 3.4 Howl (Mozilla Firefox Voice, 2020)
https://arxiv.org/pdf/2008.09606 : "hey Firefox" on Common Voice; FRR 10% at 4 FA/h; notes their negative set "likely contains more adversarial examples that misrepresent real-world usage" because many negatives contain "Firefox".

## 4. Academic datasets and benchmarks

- Google Speech Commands v1/v2 (https://arxiv.org/pdf/1804.03209): 105,829 one-second clips, 35 words. Classification accuracy only; non-streaming, no unbounded negatives.
- Hey Snips (https://arxiv.org/pdf/1811.07684): ~11K positives from 2.2K speakers, 86.5K negatives (~96 h); metric FRR at 0.5 FA/h, clean and 5 dB SNR (MUSAN); DET curves.
- MobvoiHotwords, HI-MIA (https://arxiv.org/pdf/1912.01231): standard far-field academic sets; WeKws recipes report FRR vs FA/h DET curves (https://arxiv.org/html/2210.16743v1).
- Multilingual Spoken Words Corpus (MLCommons, NeurIPS 2021): 23.4 M one-second clips, 340K keywords, 50 languages, forced-aligned out of Common Voice. No continuous negative stream, but a good source of multilingual positives and hard negatives.
- MLPerf Tiny KWS (https://github.com/mlcommons/tiny/tree/master/benchmark/training/keyword_spotting); v1.3 (Sept 2025) added a streaming wake word task: 20-minute recording with 50 "Marvin" instances; detection counts only within 1 s of keyword end (https://mlcommons.org/2025/09/mlperf-tiny-v1-3-tech/). Hardware/energy benchmark, not engine accuracy.
- LibriPhrase (https://arxiv.org/pdf/2206.15400): "easy" and "hard" negatives by Levenshtein distance; standard open-vocabulary confusable benchmark.
- 2025 Google papers on confusables: GraphemeAug (https://arxiv.org/abs/2505.14814), LLM-Synth4KWS (https://arxiv.org/abs/2505.22995) introduces c-AUC, mean area under DET over confusable groups.
- PVTC 2020 (https://arxiv.org/pdf/2101.01935): close-talk and far-field array tracks.
- Amazon DiPCo (https://arxiv.org/pdf/1909.13447): ~5.5 h, so 0.1 FA/h resolution is roughly one event.
- SynTTS-Commands (Nov 2025, https://arxiv.org/abs/2511.07821): most open engines now train on synthetic speech and need real-speech test sets to detect synthetic-to-real gaps.
- Room simulation (https://arxiv.org/abs/2006.02774): RIR realism changes results by up to 35.8% relative, so RIR choice must be specified.
- Media-playback false triggers: Northeastern "When speakers are all ears" (https://moniotrlab.khoury.northeastern.edu/smart-speakers-study-preliminary): 125 h of Netflix dialogue across 5 smart speakers; 1.5-19 activations per day per device; only 8.44% of activations reproduced on repeat; confusables like "kevin's car", "pickle", "I can work".

## 5. Independent / community comparisons (2024-2026)

- Jaco-Assistant Benchmark-KeywordSpotting (Sept 2026, AGPLv3, https://gitlab.com/Jaco-Assistant/Benchmark-KeywordSpotting ; https://github.com/dscripka/openWakeWord/issues/349 ; https://github.com/st-matskevich/local-wake/issues/5):
  - Protocol: Picovoice mixer ported; 24.7 h LibriSpeech + DEMAND mix with 329 "alexa" recordings; second run with DiPCo background; "computer" (411 keywords) and "smart mirror"; transcript-aware filtering so background clips containing the keyword are removed.
  - Metric: miss rate at FA budgets 0, 0.1, 0.25, 0.5, 1 per hour, plus detection latency distribution.
  - Results ("alexa", LibriSpeech, miss at 0.5 FA/h): local-wake 3.4%, openWakeWord 4.0%, slungt-large 5.2%, Porcupine 5.8%, microWakeWord 38.9%. On DiPCo background: Porcupine 4.3%, slungt 4.3%, local-wake 4.9%, openWakeWord 10.6%, microWakeWord 45.6%.
  - Caveats: Porcupine pinned below 2.x (free-tier version); local-wake enrolled with speakers from the same test recordings; only pretrained "alexa" models; microWakeWord evaluated outside ESPHome; single-person effort.
  - Methodological find: openWakeWord has no built-in debounce, Porcupine bakes sensitivity into construction, microWakeWord thresholds a windowed average, so a fair harness has to normalize detection semantics (rising edge + debounce) per engine.
- local-wake's own benchmark (https://github.com/st-matskevich/local-wake/tree/main/benchmark): Qualcomm Keyword Speech Dataset, 98.6% same-speaker vs 81.9% cross-speaker.
- Nyumaya and eqiihuu forks of the Picovoice benchmark: no new independent data.

## 6. Metrics and standardization status

Converged in practice:
- Primary pair: false reject rate (%) on positives vs false accepts per hour on a continuous negative stream. Picovoice (1 FA/10 h), openWakeWord (0.5 FA/h), Hey Snips (0.5 FA/h), Howl (4 FA/h), Jaco (0-1 FA/h table).
- Curves: ROC/DET with FA/h on the x-axis; AUC/AUT summaries; EER. Picovoice, Sensory and DaVoice publish single points only.
- Runtime: real-time factor, CPU %, memory; MLPerf Tiny adds duty cycle and energy.

Not converged / absent:
- The negative corpus; SNR (Picovoice 10 dB fixed; Snips 5 dB; openWakeWord 5-10 dB); far-field; confusables; speaking rate; detection semantics (window, debounce, latency tolerance); statistical reporting (no confidence intervals).

## 7. What is missing or broken

1. No shared test set.
2. Baselines are stale or mis-run.
3. Single operating points hide the trade-off.
4. Negative audio is too short and too clean.
5. No confusable or near-miss negatives.
6. No stratification by gender, age, accent, native language, distance.
7. Far-field is simulated inconsistently or not at all.
8. Detection semantics differ by engine; naive harnesses mis-score.
9. Label tolerance is loose, inflating recall for all engines.
10. Results are unstable (10-point swings from keyword spacing; only 8.44% of smart speaker false triggers repeat).
11. Training/test leakage is unverified.
12. Versions and thresholds are not pinned.
13. No independence: every published comparison is by a party that wins it.

## 8. What a good methodology looks like

- Metrics: full FRR-vs-FA/h DET curve per engine with bootstrap confidence intervals; tabulate FRR at 0.1, 0.5, 1 and 3 FA/h; EER and AUT; detection latency percentiles; RTF/CPU/RAM on named hardware. Report the operating-point threshold used for each engine.
- Positives: several hundred speakers per keyword, balanced by gender, age, native/non-native, accents; multiple keywords (name-like, dictionary word, two-word phrase); consented, redistributable licences. Real far-field re-recordings at 1/3/5 m.
- Negatives: 50-100+ hours, partitioned by domain: conversational far-field, read speech, TV/film/podcast media, music, household noise, silence. FA/h per domain. Transcript-aware filtering.
- Confusables: scripted near-miss set per keyword, scored separately.
- Conditions: SNR sweep (clean, 20, 10, 5, 0 dB); RIR set with documented RT60 and distances; speaking-rate sweep; loudspeaker playback vs live speech.
- Detection semantics: rising edge over threshold, one detection per debounce window, hit only if fired within [word end, +1 s], anything else a false accept; keyword timestamps at the word, not the clip.
- Reproducibility: pinned engine versions, model hashes, containers, seeds, published mixer code, published raw per-frame scores; leakage check.
- Governance: run by a party that ships no engine; results open; engines added by PR with vendor-supplied configs.

## 9. Requirements for an independent benchmark

- Independence: maintained by a party with no engine to sell.
- Common, fully open, redistributable test set for every engine.
- Curves, not points, with bootstrap confidence intervals.
- Large, domain-partitioned negative corpus (50-100 h), FA/h per domain.
- Confusable / near-miss negative set per keyword.
- Speaker-stratified positives (hundreds of speakers per keyword).
- Multiple keywords; custom-keyword track separate from built-in-keyword track.
- SNR sweep; far-field track with measured RIRs and real re-recordings.
- Speaking-rate / whisper / shout subsets; playback vs live.
- Uniform detection semantics across engines.
- Pinned versions, model hashes, containers, raw per-frame scores published.
- Latency and cost on named reference hardware (Raspberry Pi, ESP32-S3 class).
- Stability reporting: repeated runs with different keyword spacings.
- Engine-inclusion policy covering commercial engines (licence keys obtained by maintainers).

## 10. Source list

- https://github.com/Picovoice/wake-word-benchmark
- https://picovoice.ai/docs/benchmark/wake-word/
- https://picovoice.ai/blog/benchmarking-a-wake-word-detection-engine/
- https://github.com/Picovoice/wake-word-benchmark/issues/13
- https://github.com/Picovoice/wake-word-benchmark/issues/17
- https://huggingface.co/datasets/domdomegg/picovoice-wake-word-benchmark
- https://github.com/dscripka/openWakeWord
- https://github.com/dscripka/openWakeWord/discussions/135
- https://github.com/dscripka/openWakeWord/issues/350
- https://github.com/dscripka/openWakeWord/issues/349
- https://github.com/OHF-Voice/micro-wake-word
- https://www.home-assistant.io/blog/2024/06/26/voice-chapter-7/
- https://www.home-assistant.io/blog/2024/10/24/wake-word-collective/
- https://gitlab.com/Jaco-Assistant/Benchmark-KeywordSpotting
- https://github.com/st-matskevich/local-wake/issues/5
- https://github.com/livekit/livekit-wakeword
- https://davoice.io/benchmark
- https://voxrt.com/wake-word-comparison
- https://violawake.com/compare/picovoice/
- https://sensory.com/revisiting-wake-word-accuracy-and-privacy/
- https://arxiv.org/pdf/1804.03209 (Speech Commands)
- https://arxiv.org/pdf/1811.07684 (Hey Snips)
- https://arxiv.org/pdf/2111.10592 (Deep Spoken KWS overview)
- https://ebooks.iospress.nl/doi/10.3233/FAIA230695
- https://github.com/mlcommons/tiny/tree/master/benchmark/training/keyword_spotting
- https://mlcommons.org/2025/09/mlperf-tiny-v1-3-tech/
- https://arxiv.org/pdf/1912.01231 (HI-MIA)
- https://arxiv.org/pdf/2206.15400 (LibriPhrase)
- https://arxiv.org/abs/2505.14814 (GraphemeAug)
- https://arxiv.org/abs/2505.22995 (LLM-Synth4KWS)
- https://arxiv.org/abs/2511.07821 (SynTTS-Commands)
- https://arxiv.org/pdf/1909.13447 (DiPCo)
- https://arxiv.org/abs/2006.02774 (room simulation)
- https://arxiv.org/pdf/2008.09606 (Howl)
- https://moniotrlab.khoury.northeastern.edu/smart-speakers-study-preliminary
- https://arxiv.org/html/2506.11169 (Small-footprint KWS review 2025)
