# On-device lane

Accuracy is measured on cloud or workstation CPUs; it does not depend on the device.
Cost does. The on-device lane measures real-time factor, CPU share and peak memory on
reference hardware, and (later) power on microcontrollers.

## Reference devices

| Class | Device | Role |
|---|---|---|
| Single-board computer | Raspberry Pi 5 (8 GB), Raspberry Pi OS 64-bit | GitHub Actions self-hosted runner; runs `scripts/measure_device.py` |
| Microcontroller | ESP32-S3 (e.g. Home Assistant Voice PE class board) | Device under test flashed from the Pi; microWakeWord and Porcupine only |

## Raspberry Pi runner

1. Install Raspberry Pi OS 64-bit, `sudo apt install ffmpeg libsndfile1`.
2. Register a GitHub Actions runner with labels `self-hosted, linux, ARM64, pi5` as an
   *ephemeral, just-in-time* runner (`./config.sh --ephemeral`), so each job starts
   from a clean process. Run it under a dedicated user with no access to secrets.
3. Copy the tier A chunks for the current manifest to `/srv/wakewordworld/data`
   (`chunks/<source_id>/<chunk_id>.flac`, plus `index/`). The Pi never holds tier B
   audio.
4. Trigger `.github/workflows/on-device.yml` manually. The workflow refuses to run on
   forks and never runs for pull requests.

The measurement script reports wall and CPU real-time factor, CPU percent of one core,
and peak RSS. Results land in `results/device/<device>-<engine>.json` and are joined
into the report by engine id and version.

## ESP32-S3 device under test

Planned procedure (M6, not yet automated):

1. Build the engine firmware (ESPHome `micro_wake_word` component, or the Picovoice
   ESP-IDF demo) with the same model files whose hashes appear in `engine.yaml`.
2. Play the evaluation chunks from the Pi through a measured loudspeaker at 1 m in a
   quiet room; capture detections over serial with timestamps synchronised to the
   playback clock.
3. Score detections with `wakewordworld.eval.detect` exactly like streaming scores
   (rising edge, debounce, hit window); report FRR/FA per hour and detection latency
   for the acoustic path, plus current draw measured with an inline USB power meter.

Acoustic replay adds the loudspeaker and room to the chain; those numbers are reported
in their own table and never mixed with the direct-feed results.
