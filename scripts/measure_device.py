"""Measure on-device cost of an engine: real-time factor, CPU time, peak memory.

Runs the harness scoring loop on a fixed set of chunks and records resource usage with
the ``resource`` module (portable to Linux and macOS). Intended for the self-hosted
Raspberry Pi lane; works on any machine for comparison.

Example::

    uv run python scripts/measure_device.py openwakeword \
        --manifest manifests/0.1.0 --limit 20 --out results/device/pi5-openwakeword.json
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import sys
import time
from pathlib import Path


def _peak_rss_mb() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF)
    # ru_maxrss is bytes on macOS and kilobytes on Linux.
    return ru.ru_maxrss / (1024 * 1024) if sys.platform == "darwin" else ru.ru_maxrss / 1024


def main() -> int:
    """Entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("engine_id")
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--config", default="{}", help="engine config JSON")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--data-root", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--label", default=platform.node(), help="device label, e.g. pi5")
    args = ap.parse_args()

    from wakewordworld.engines.registry import load_engine
    from wakewordworld.eval.protocol import stream_chunk
    from wakewordworld.eval.run import load_manifest_rows
    from wakewordworld.util.paths import DataRoot

    data_root = DataRoot.resolve(args.data_root)
    rows = load_manifest_rows(args.manifest)[: args.limit]
    engine = load_engine(args.engine_id, **json.loads(args.config))
    per_chunk = []
    cpu0 = time.process_time()
    wall0 = time.perf_counter()
    try:
        for row in rows:
            audio = data_root.chunks / row.source_id / f"{row.chunk_id}.flac"
            if not audio.exists():
                continue
            _, st = stream_chunk(engine, audio, chunk_id=row.chunk_id)
            per_chunk.append({"chunk_id": row.chunk_id, "audio_s": st.audio_s, "wall_s": st.wall_s})
    finally:
        engine.close()
    cpu = time.process_time() - cpu0
    wall = time.perf_counter() - wall0
    audio_s = sum(c["audio_s"] for c in per_chunk)
    report = {
        "device": args.label,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "engine_id": engine.info.engine_id,
        "engine_version": engine.info.version,
        "model_hashes": dict(engine.info.model_hashes),
        "n_chunks": len(per_chunk),
        "audio_s": audio_s,
        "wall_s": wall,
        "cpu_s": cpu,
        "rtf_wall": wall / audio_s if audio_s else None,
        "rtf_cpu": cpu / audio_s if audio_s else None,
        "cpu_percent_of_one_core": 100.0 * cpu / audio_s if audio_s else None,
        "peak_rss_mb": _peak_rss_mb(),
        "per_chunk": per_chunk,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "per_chunk"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
