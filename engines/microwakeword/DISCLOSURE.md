# Disclosure: microWakeWord

Filled from public documentation (OHF-Voice/micro-wake-word README, model release
notes, kahrendt/microwakeword feature dataset card). To be confirmed by the engine
maintainers.

## Training data

- Positives: Piper TTS synthetic samples with augmentation; the "okay nabu" v2 model
  additionally used real recordings from the Home Assistant Wake Word Collective.
- Negatives / ambient: spectrogram features from CHiME-6, DiPCo, FMA, FSD50K, WHAM!,
  LibriSpeech-other and VOiCES (published as features, CC BY-NC 4.0).

## Benchmark overlap

| Benchmark source | Used in training? | Notes |
|---|---|---|
| common_voice | not documented as used | |
| voxpopuli | no | |
| icsi, hackerpublicradio, kuechenradio, media_ccc_de, libre_a_vous, kde_espana, espika_fm | no | |
| (future) DiPCo, CHiME-6 | yes | any DiPCo/CHiME-6 negatives in the benchmark are flagged |

## Commercial status

- Runtime and models: Apache-2.0. Commercial product: no (Nabu Casa ships it in HA Voice PE).
- On-device: yes (ESP32-S3 via TFLite-Micro; CPU via pymicro-wakeword).
- Custom wake words: training required.
