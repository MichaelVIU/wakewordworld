# Disclosure: openWakeWord

Filled from public documentation (README and training notebook of dscripka/openWakeWord,
v0.6.0). To be confirmed by the maintainer of the engine when they review this entry.

## Training data

- Positives: synthetic speech generated with Piper TTS (many voices), augmented with
  room impulse responses and noise.
- Negatives: real audio features (not raw audio) from ACAV100M, Mozilla Common Voice
  and the Free Music Archive; ambient "false-activation" validation set built from
  DiPCo, the Santa Barbara Corpus and MUSDB.
- Embedding backbone: Google speech-embedding model (trained by Google on undisclosed
  data).

## Benchmark overlap

| Benchmark source | Used in training? | Notes |
|---|---|---|
| common_voice | yes (features) | negative examples; results on this source are flagged |
| voxpopuli | not documented | |
| icsi, hackerpublicradio, kuechenradio, media_ccc_de, libre_a_vous, kde_espana, espika_fm | no | |

## Commercial status

- Runtime licence: Apache-2.0. Bundled models: CC BY-NC-SA 4.0.
- Commercial product: no. On-device: yes (CPU, ONNX/TFLite).
- Custom wake words: training required (synthetic positives), no text-only path.
