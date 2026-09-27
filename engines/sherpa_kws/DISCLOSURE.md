# Disclosure: sherpa-onnx keyword spotting

Filled from the k2-fsa documentation and icefall recipes (sherpa-onnx 1.13.8). The
authors have not reviewed this entry.

## Training data

- English model `sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01`: zipformer
  transducer trained with icefall on GigaSpeech (10,000 h of audiobooks, podcasts and
  YouTube; corpus under a non-commercial research licence, model weights Apache-2.0).
- Chinese models: WenetSpeech; bilingual: a zh-en mix. Not used here.

## Benchmark overlap

| Benchmark source | Used in training? | Notes |
|---|---|---|
| hackerpublicradio, media_ccc_en, wikimania | possible | GigaSpeech contains podcasts and YouTube audio; no item-level list is published, so overlap cannot be excluded |
| other sources | not documented | |

## Commercial status

- Open source (Apache-2.0 runtime and weights); the GigaSpeech corpus licence restricts
  commercial use of the *data*, not of the released model.
- On-device: yes (CPU via ONNX Runtime; also Android, iOS, embedded Linux).
- Custom wake words: text-only, any phrase spellable with the model's tokens; no
  retraining. Boolean hit events with a per-keyword threshold that the benchmark sweeps.
