# Wake word / KWS engine landscape (September 2026)

Scope: engines a developer can integrate and benchmark locally. "Batch eval" = can you feed WAV files from Python and get scores. Dates from GitHub API (pushed_at) queried 2026-09-25 unless noted.

## 1. Summary table

| Engine | License | Custom wake word | Languages (EN / DE) | Runtimes | Python batch-eval on WAV | Output | Last activity |
|---|---|---|---|---|---|---|---|
| **openWakeWord** (dscripka) | Apache-2.0 code; bundled models CC-BY-NC-SA 4.0 | Yes, training required (synthetic Piper TTS, Colab/local notebook, ~1 h); free | EN only officially; community pipelines do DE via Piper de_DE voices; German "Hey Rona" model on HF (DocCheck) | Python (ONNX/tflite), community C++/Rust ports, Wyoming server for HA | Yes: `Model.predict(frame)`, `Model.predict_clip("x.wav")` (16 kHz/16-bit) | Per-80 ms-frame score 0–1, default threshold 0.5, optional debounce | Last release v0.6.0 (2024-02-11); last commit 2025-12-30. Training deps aging (issue #317, Feb 2026) |
| **microWakeWord** (OHF-Voice/ESPHome) | Apache-2.0 | Yes, training required (Piper TTS + augmentation; author calls it "very difficult"); community Colab wrappers | Official models EN (okay_nabu, hey_jarvis, hey_mycroft, alexa, stop); DE possible by training with German Piper voices | ESP32/ESP32-S3 via TFLite-Micro (ESPHome), Android (HA Companion app since 2026.3), Python via `pymicro-wakeword` | Yes: `pymicro-wakeword` (`process_streaming_prob()`; CLI `python -m pymicro_wakeword --model okay_nabu *.wav`) | Per-10 ms-step probabilities; detection when sliding window (e.g. 5) exceeds cutoff (e.g. 0.97) | Trainer pushed 2026-07-06; pymicro-wakeword 2.5.0 released 2026-09-17 |
| **Picovoice Porcupine** | SDK Apache-2.0, engine needs AccessKey; free plan non-commercial, limited custom models per month; paid = contact sales | Yes, text-only: type phrase in Picovoice Console, `.ppn` trains "in seconds"; platform-specific models | EN, DE, FR, IT, ES, PT, JA, KO, ZH | Python, Java, .NET, Node, C, Android, iOS, Flutter, RN, Web/WASM, Rust, RPi, Arduino/Cortex-M | Yes: `handle.process(frame)` on 512-sample 16 kHz frames → keyword index or -1; no raw score, only `sensitivity` 0–1 | Boolean per frame; no per-frame probabilities | v4.0.0 (2025-12-11); repo pushed 2026-09-10 |
| **Mycroft Precise** | Apache-2.0 | Yes, training on recorded samples; GRU model | Language-agnostic | Python/Linux/RPi; OVOS plugins (precise-lite, precise-onnx since Nov 2025) | Yes: `precise-test` on dirs of WAVs | Per-chunk prob 0–1 | Dead upstream: last push 2023-11; kept alive by OpenVoiceOS |
| **Snowboy** (Kitt.AI) | Apache-2.0 | Training service offline since 2021; seasalt-ai fork adds local personal models | Language-agnostic personal models | Python, C++, Android; Snowman (C++ rewrite) | Yes (existing models only) | Detection index / sensitivity | Legacy only (2021/2022) |
| **Sensory TrulyHandsfree / VoiceHub** | Commercial, closed. VoiceHub: free self-service models for prototyping; production needs license | Yes, text-only via VoiceHub portal (~1 h) | 15+ languages incl. EN, DE | Android, iOS, Linux, Windows, QNX, MCUs; C/Java APIs — no Python | No official Python API | Detection events | SDK 7.6.0 Oct 2025 |
| **Rhasspy Raven** | MIT | 3+ WAV templates, DTW matching | Language-agnostic | Python | Yes | Per-window DTW probability | Archived 2025-10-06. Dead. |
| **Vosk keyword mode** | Apache-2.0 | Text-only grammar `'["hey computer","[unk]"]'`; no training | 20+ langs incl. EN, DE | Python, Java, C#, Node, Android, iOS | Yes (ASR over WAV) | Text with per-word confidence; latency 0.2–3 s | repo pushed 2026-08-09 |
| **sherpa-onnx open-vocabulary KWS** (k2-fsa) | Apache-2.0 | Text-only keywords file; per-keyword boosting and threshold | Pretrained zh-en, wenetspeech (zh), gigaspeech (EN). No DE model | Python, C/C++, Java, JS/WASM, Kotlin, Swift, Go, Rust, Android, iOS | Yes: `sherpa_onnx.KeywordSpotter` | Keyword hit events; tunable threshold | PyPI 1.13.8 (2026-09-10) |
| **Whisper-based** (whisper.cpp `command`, Trigger-Talk) | MIT | Text-only match on transcript | 99 langs incl. DE | Python/C++; too heavy for MCU | Yes | Text; high latency/CPU | Trigger-Talk pushed 2025-07 |
| **DaVoice** | Sample repos MIT, engine closed | By e-mail request → `.onnx` per phrase; "50+ languages" | 50+ incl. DE (claimed) | Python, Android, iOS, RN, Flutter, Web, Unity, .NET, Go, Rust, C/C++ | Yes: `KeywordDetection`, `start_keyword_detection_from_file(path)` | Callback with `phrase`, `threshold_scores` | Python repo pushed 2026-09-10 |
| **Speechly** | — | — | — | — | — | — | Discontinued (archived 2025-01) |
| **LiveKit wakeword** (2026) | Apache-2.0 | Training required, one command: synthetic TTS + augmentation; Conv-attention head on openWakeWord-style embeddings | EN production quality; 30+ langs incl. DE via VoxCPM2 TTS with "lower accuracy" | Python 3.11+, Rust crate, Swift; ESP32 in development | Yes: `WakeWordModel.predict(frame)`; `run_eval()` gives AUT, FPPH, recall, DET | Per-frame confidence 0–1 | PyPI 0.2.1 (2026-05-20); pushed 2026-08-01 |
| **ViolaWake** (2026) | Apache-2.0 | Training: ~200 recorded positives + TTS positives + confusable negatives; `violawake-train` | EN demonstrated | Python, browser via onnxruntime-web | Yes: `WakeDetector.process(chunk)`, `violawake-eval`, `violawake-streaming-eval` (FA/h) | Per-chunk score 0–1 | Pre-release beta; pushed 2026-08-05 |
| **VoxRT** (2026) | SDK Apache-2.0; runtime closed, free for commercial use; custom phrases = paid | On request | Any (on request) | Android, iOS, browser WASM, Linux; Rust runtime, no Python | No Python | Threshold-crossing events | pushed 2026-09-15 |
| **onnx-wakeword / Voicute** (2026) | No license file; models via voicute.com | Online service (~30 min/keyword) | ZH, EN, JA, FR, DE | Python, Android, ESP32, Web; models <130 KB | Callback with probability; file eval not documented | Keyword + probability | pushed 2026-09-21 |
| **Outspoken** (2026) | Hosted openWakeWord training | Text-only web UI, ~45 min → ONNX | EN, NL, DE, FR | Anything running OWW ONNX | Yes (standard OWW API) | OWW scores | updated 2026-09-13 |
| **Hey Buddy** (painebenjamin) | Apache-2.0 | Training required (100k Piper TTS positives) | EN | Browser (ONNX Runtime Web), Python | Yes | Per-frame score | pushed 2025-07-25 |
| **EfficientWord-Net** | Apache-2.0 | Few-shot: 3–4 recordings → embedding reference | Language-agnostic | Python 3.10–3.14 | Yes (`HotwordDetector.scoreFrame`) | Similarity score, threshold ~0.7 | PyPI 1.0.5 (2026-01-05) |
| **local-wake** | Open source | No training: Google speech-embedding + DTW vs 3–4 reference clips | Language-agnostic | Python/RPi | Yes | DTW distance/score | Active 2025–26 |
| **WeKWS** (WeNet) | Apache-2.0 | Training required (recipes hi_xiaowen, Hey Snips, Speech Commands) | data-driven | PyTorch, ONNX | Yes (research toolkit) | Per-frame posteriors | pushed 2026-07-23 |
| **NVIDIA NeMo MatchboxNet/MarbleNet** | Apache-2.0 | Closed-set classification; retrain | data-driven | PyTorch/NeMo | Yes | Class posteriors | no dedicated wake-word product |
| **Azure Custom Keyword** | Free Basic/Advanced `.table` models in Speech Studio; free on-device use | Text-only (Speech Studio) | Multiple incl. DE (verify) | Speech SDK: C#, C++, Python, Java, JS, Android, iOS; `KeywordRecognizer` offline | Yes via Speech SDK | Recognized events | Docs 2026-02-25 |
| **Apple / Google / Amazon** | Apple: no third-party wake-word API. Google: Assistant SDK deprecated; EU DMA decision (2026-07-16) forces open Android wake-word access by 2027-08-01. Amazon: AVS Device SDK archived 2024-01. None usable for local benchmarking. |

## 2. Published accuracy / false-accept numbers

| Engine | Claim | Data / method |
|---|---|---|
| Picovoice Porcupine | 97.3% detection at 1 FA / 10 h, 10 dB SNR; Snowboy 68.1%, PocketSphinx 48.0% | Own benchmark: 300+ recordings of 6 keywords from 50+ speakers, LibriSpeech background, DEMAND noise |
| openWakeWord | Target "<5% FRR at <0.5 FA/h" | Picovoice positives; DiPCo negatives (~5.5 h) |
| microWakeWord | No formal numbers | Real-world background sets |
| LiveKit wakeword | 0.08 FP/h vs 8.50 for openWakeWord; recall 86% vs 69% | Internal "hey livekit" set: 15,000 pos, 45,084 neg, 25 h |
| ViolaWake | EER 5.49% vs OWW alexa EER 8.24%; production model claims 0.8% EER | Own benchmark_v2 |
| VoxRT | ROC AUC 0.9966 | Own held-out data |
| onnx-wakeword | Recall 90.3–100% | Own Common Voice/music/household mix |
| DaVoice | 99.25% / 97.65% detection, 0 false positives; openWakeWord 62–69% | Customer reports, not reproducible |
| Sensory, Snowboy, Precise, Raven, Vosk, sherpa-onnx | No public numbers | — |

## 3. Practical takeaways for a local benchmark

- Best Python batch-eval ergonomics with raw per-frame scores: openWakeWord, LiveKit wakeword (built-in `run_eval` with FPPH/AUT), ViolaWake, pymicro-wakeword (`process_streaming_prob`), EfficientWord-Net.
- Porcupine only exposes a boolean per frame plus a sensitivity knob, so ROC/DET curves require sweeping sensitivity; Sensory and VoxRT have no Python API.
- Text-only custom wake words without training: Porcupine (seconds), Sensory VoiceHub (~1 h), Azure Custom Keyword, Outspoken (~45 min, OWW-compatible), Voicute (~30 min), sherpa-onnx KWS and Vosk (grammar).
- German: first-class in Porcupine, Sensory, Azure, Outspoken, Voicute, Vosk, DaVoice (claimed); trainable via German Piper voices in openWakeWord/microWakeWord/LiveKit (lower quality); not available in sherpa-onnx pretrained KWS.
- ESP32/MCU: microWakeWord (production in HA Voice PE), Porcupine, Edge Impulse, onnx-wakeword; LiveKit ESP32 in development.
- Browser: Porcupine WASM, Hey Buddy, VoxRT WASM, ViolaWake, Voicute web.
- Dead or frozen: Snowboy, Mycroft Precise upstream, Raven, Speechly, AVS Device SDK, Google Assistant SDK. Kyutai has no wake-word component.
- Home Assistant: openWakeWord via Wyoming add-on, microWakeWord on Voice PE, ESPHome and Android Companion app (HA 2026.3).

## 4. Sources

- openWakeWord: https://github.com/dscripka/openWakeWord ; https://huggingface.co/davidscripka/openwakeword ; https://huggingface.co/DocCheck/wakeword-hey-rona ; https://github.com/dscripka/openWakeWord/discussions/52
- microWakeWord: https://github.com/OHF-Voice/micro-wake-word ; https://github.com/esphome/micro-wake-word-models ; https://github.com/OHF-Voice/pymicro-wakeword ; https://esphome.io/components/micro_wake_word/ ; https://www.home-assistant.io/blog/2026/03/04/release-20263/
- Porcupine: https://github.com/Picovoice/porcupine ; https://picovoice.ai/docs/porcupine/ ; https://picovoice.ai/pricing/
- Mycroft Precise: https://github.com/MycroftAI/mycroft-precise ; https://github.com/OpenVoiceOS/ovos-ww-plugin-precise-onnx
- Snowboy: https://github.com/Kitt-AI/snowboy ; https://github.com/seasalt-ai/snowboy
- Sensory: https://sensory.com/product/voicehub/ ; https://doc.sensory.com/tnl/7.7/
- Raven: https://github.com/rhasspy/rhasspy-wake-raven
- Vosk: https://github.com/alphacep/vosk-api ; https://github.com/OpenVoiceOS/ovos-ww-plugin-vosk
- sherpa-onnx KWS: https://k2-fsa.github.io/sherpa/onnx/kws/index.html
- Whisper-based: https://github.com/ManiAm/Trigger-Talk
- DaVoice: https://davoice.io/ ; https://github.com/frymanofer/Python_WakeWordDetection
- LiveKit: https://livekit.com/blog/livekit-wakeword ; https://github.com/livekit/livekit-wakeword
- ViolaWake: https://github.com/GeeIHadAGoodTime/ViolaWake
- VoxRT: https://github.com/VoxRT
- onnx-wakeword/Voicute: https://github.com/voicute/onnx-wakeword
- Outspoken: https://outspoken.cloud/blog/best-wake-word-tools
- Hey Buddy: https://github.com/painebenjamin/hey-buddy
- EfficientWord-Net: https://github.com/Ant-Brain/EfficientWord-Net
- local-wake: https://github.com/st-matskevich/local-wake
- WeKWS: https://github.com/wenet-e2e/wekws
- Azure Custom Keyword: https://learn.microsoft.com/en-us/azure/ai-services/speech-service/keyword-recognition-overview
- Wyoming openWakeWord: https://github.com/rhasspy/wyoming-openwakeword
