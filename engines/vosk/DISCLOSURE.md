# Disclosure: Vosk

Filled from public documentation (alphacephei.com/vosk/models). Vosk is a general
speech recogniser used here as an ASR baseline with a keyword grammar.

## Training data

- Small models (en-us 0.15, de 0.15, fr 0.22, es 0.42): Kaldi acoustic models trained on
  public corpora listed per model on the model page, including LibriSpeech, Common Voice,
  TED-LIUM, VoxForge, M-AILABS and others depending on language.

## Benchmark overlap

| Benchmark source | Used in training? | Notes |
|---|---|---|
| common_voice | yes (several languages) | results on this source are flagged |
| voxpopuli | not documented | |
| icsi, hackerpublicradio, kuechenradio, media_ccc_de, libre_a_vous, kde_espana, espika_fm | no | |

## Commercial status

- Runtime and models: Apache-2.0. Commercial product: no. On-device: yes (CPU).
- Custom wake words: any text phrase via grammar, no training.
