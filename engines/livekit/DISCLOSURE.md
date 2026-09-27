# Disclosure: LiveKit wakeword

Filled from the public repository and blog post (livekit-wakeword 0.2.1). LiveKit has
not reviewed this entry.

## Training data

- Positives: synthetic speech generated with Piper TTS (English) or VoxCPM2 (30+
  languages), augmented with noise and room impulse responses.
- Negatives: synthetic confusable phrases plus real background audio sets selected by
  the training configuration (the defaults reference public noise and speech corpora;
  the exact list depends on the submitted config).
- Backbone: openWakeWord-style Google speech-embedding model (trained by Google on
  undisclosed data) with a convolutional-attention classifier head.

## Benchmark overlap

| Benchmark source | Used in training? | Notes |
|---|---|---|
| common_voice | depends on the training config | the default negative sets include Common Voice; submitters must state it |
| other sources | not documented | |

## Commercial status

- Open source (Apache-2.0). On-device: yes (CPU via ONNX Runtime; Rust and Swift
  runtimes exist; ESP32 in development).
- Custom wake words: training required (one command, synthetic positives); no text-only
  path. English production quality; other languages marked lower accuracy upstream.
- Score semantics: continuous confidence per 2 s window evaluated every 80 ms frame;
  the shipped listener uses threshold 0.5 and a 2 s debounce, which the harness replaces
  with its own detection semantics.
