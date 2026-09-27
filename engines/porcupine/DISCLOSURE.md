# Disclosure: Picovoice Porcupine

Filled from public documentation (picovoice.ai docs, pvporcupine 4.0.3). Picovoice has
not reviewed this entry.

## Training data

- Proprietary. Picovoice states that custom keywords are trained "from text" with its
  own data pipeline; the training corpus is not disclosed.
- Built-in keywords (alexa, computer, jarvis, porcupine, hey google, hey siri, ...) are
  shipped as pre-trained `.ppn` files.

## Benchmark overlap

| Benchmark source | Used in training? | Notes |
|---|---|---|
| all sources | not documented | training data is not disclosed, so overlap cannot be excluded |

## Commercial status

- Commercial product. The free tier requires an access key from the Picovoice Console,
  is limited to non-commercial use and to a small number of custom keywords per month;
  keys are held by the maintainers and never stored in the repository or the image.
- On-device: yes (CPU; Linux, macOS, Windows, Raspberry Pi, Android, iOS, web, MCUs).
- Custom wake words: text-only, generated in the Picovoice Console within seconds,
  per language and platform.
- Score semantics: boolean per 512-sample frame with a construction-time sensitivity;
  the benchmark sweeps sensitivity 0.1-0.9 to obtain a curve. Versions before 2.x are
  not comparable with current models.
