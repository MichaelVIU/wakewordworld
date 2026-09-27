# Disclosure: EfficientWord-Net

Filled from the public repository and paper (EfficientWord-Net 1.0.5). The authors have
not reviewed this entry.

## Training data

- Embedding backbone: ResNet-50 trained with an ArcFace loss on spoken-word data
  described in the project's paper (Google Speech Commands and synthetic/collected
  word recordings); exact corpus not itemised upstream.
- Wake words themselves are not trained: 3-4 enrolment recordings per word produce a
  reference embedding file.

## Benchmark overlap

| Benchmark source | Used in training? | Notes |
|---|---|---|
| all sources | no (backbone) | Speech Commands is not a benchmark source |
| enrolment clips | never from the benchmark | the maintainer records or sources them outside the pool and hashes the reference file |

## Commercial status

- Open source (Apache-2.0). On-device: yes (CPU, TFLite/ONNX).
- Custom wake words: enrolment from a few recordings, no training, no text-only path.
- Score semantics: cosine-style similarity on a 1.5 s window sliding by 0.75 s; the
  library's default threshold is about 0.7, replaced by the harness sweep.
