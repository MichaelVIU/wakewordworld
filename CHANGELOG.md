# Changelog

All notable changes to the harness are documented here. Dataset releases have their own
changelog under `manifests/`.

## Unreleased

- M0: repository skeleton, licence tiers, source spec schema, manifest schema, CLI,
  governance documents, CI.
- M1: fetchers (RSS, PeerTube, Internet Archive, media.ccc.de, Wikimedia Commons,
  Hugging Face Hub, HTTP archives), ingest pipeline, transcription (faster-whisper,
  parakeet-mlx, MMS alignment, reference parsers), word index with name lexicons and
  phonetic near-miss index, manifest build/freeze/validate; 21 source specs.
- M2: evaluation harness (streaming protocol, detection semantics, metrics with
  cluster bootstrap), adapters for openWakeWord, microWakeWord, Vosk, Porcupine,
  LiveKit wakeword, EfficientWord-Net, sherpa-onnx KWS.
- M3-M5 tooling: static report and JSON leaderboard, dataset card, Hugging Face publish
  and Zenodo clients, release/submission/weekly workflows, HF Jobs scripts, runbooks.
- M4: augmentation lane (SNR mixing with real noise, RIR convolution, speed perturbation).
- M6: on-device measurement script and self-hosted runner workflow.
