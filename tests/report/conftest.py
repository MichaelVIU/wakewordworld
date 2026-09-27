from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from wakewordworld.eval.results import RunMeta, SummaryRow, write_run

BUDGETS = ("0_1", "0_5", "1", "3")


def _row(
    wake_word: str,
    slice_type: str,
    slice_value: str,
    *,
    aut: float,
    spread: float,
    n_positives: int = 400,
    n_units: int = 120,
) -> SummaryRow:
    kwargs: dict[str, object] = {}
    for i, b in enumerate(BUDGETS):
        v = min(1.0, aut + 0.05 * (3 - i))
        kwargs[f"frr_at_{b}"] = v
        kwargs[f"frr_at_{b}_lo"] = max(0.0, v - spread)
        kwargs[f"frr_at_{b}_hi"] = min(1.0, v + spread)
    return SummaryRow(
        wake_word=wake_word,
        slice_type=slice_type,
        slice_value=slice_value,
        n_units=n_units,
        n_chunks=n_units * 3,
        n_positives=n_positives,
        n_confusable=50,
        negative_hours=42.0,
        threshold_at_0_5=0.42,
        eer=aut,
        aut=aut,
        aut_lo=max(0.0, aut - spread),
        aut_hi=min(1.0, aut + spread),
        confusable_fa_per_1000=12.5,
        latency_p50_s=0.31,
        latency_p90_s=0.55,
        latency_p99_s=0.9,
        rtf_median=0.02,
        **kwargs,  # type: ignore[arg-type]
    )


def _curve(wake_word: str, slice_type: str, slice_value: str, offset: float) -> pl.DataFrame:
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    fa = [8.0, 2.0, 0.6, 0.2, 0.0]
    frr = [offset, offset + 0.05, offset + 0.1, offset + 0.2, offset + 0.4]
    return pl.DataFrame(
        {
            "wake_word": [wake_word] * 5,
            "slice_type": [slice_type] * 5,
            "slice_value": [slice_value] * 5,
            "threshold": thresholds,
            "fa_per_hour": fa,
            "frr": [min(1.0, f) for f in frr],
        }
    )


def make_run(
    root: Path,
    *,
    engine_id: str,
    version: str,
    aut: float,
    spread: float,
    prelim_word: bool = False,
    verified: bool = False,
) -> Path:
    run_id = f"{engine_id}-{version}"
    summary = [
        _row("michael", "all", "all", aut=aut, spread=spread),
        _row("michael", "language", "de", aut=aut + 0.02, spread=spread),
        _row("michael", "language", "en", aut=aut - 0.01, spread=spread),
        _row("michael", "domain", "podcast", aut=aut, spread=spread),
        _row("michael", "sealed", "no", aut=aut, spread=spread),
        _row("michael", "sealed", "yes", aut=aut + 0.03, spread=spread),
        _row(
            "computer",
            "all",
            "all",
            aut=aut,
            spread=spread,
            n_positives=20 if prelim_word else 300,
            n_units=10 if prelim_word else 80,
        ),
    ]
    curves = pl.concat(
        [
            _curve("michael", "all", "all", aut),
            _curve("michael", "language", "de", aut + 0.02),
            _curve("michael", "language", "en", aut - 0.01),
            _curve("computer", "all", "all", aut),
        ]
    )
    chunks = pl.DataFrame({"chunk_id": ["c1"], "rtf": [0.02]})
    meta = RunMeta.new(
        run_id=run_id,
        manifest_version="0.1.0",
        manifest_dir="manifests/0.1.0",
        engine_id=engine_id,
        engine_version=version,
        model_hashes={"model.onnx": "a" * 64},
        engine_config={},
        wake_words={"michael": "michael", "computer": "computer"},
        detection={"debounce_s": 1.0, "pre_s": 0.5, "post_s": 1.0},
        n_chunks=3,
        audio_hours=1.5,
        languages=["de", "en"],
    )
    out = root / run_id
    write_run(out, meta, summary, curves, chunks)
    if verified:
        d = json.loads((out / "run.json").read_text())
        d["verified"] = True
        (out / "run.json").write_text(json.dumps(d))
    return out


@pytest.fixture
def results_root(tmp_path: Path) -> Path:
    root = tmp_path / "results"
    # "good" has the better point estimate but a wide interval; "steady" wins on aut_hi.
    make_run(root, engine_id="good", version="1.0", aut=0.10, spread=0.15, verified=True)
    make_run(root, engine_id="steady", version="2.1", aut=0.14, spread=0.02, prelim_word=True)
    return root
