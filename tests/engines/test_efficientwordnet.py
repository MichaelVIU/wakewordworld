from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

from wakewordworld.engines.base import FRAME_SAMPLES, EngineUnavailableError
from wakewordworld.engines.efficientwordnet import WINDOW_SAMPLES, EfficientWordNetEngine


def _install_fake_ewn(fake_modules) -> types.SimpleNamespace:
    state = types.SimpleNamespace(init=None, frames=[], base=0)
    pkg = types.ModuleType("eff_word_net")
    engine = types.ModuleType("eff_word_net.engine")
    ap = types.ModuleType("eff_word_net.audio_processing")

    class Resnet50_Arc_loss:  # noqa: N801 - mirrors the library's class name
        def __init__(self):
            state.base += 1

    class HotwordDetector:
        def __init__(self, hotword, model, reference_file, threshold):
            state.init = {"hotword": hotword, "reference_file": reference_file}

        def scoreFrame(self, inp):  # noqa: N802 - library method name
            arr = np.asarray(inp)
            state.frames.append(arr.copy())
            assert arr.shape == (WINDOW_SAMPLES,)
            if not np.any(arr):
                return None  # silence gate
            return {"match": True, "confidence": 1.5 if arr.dtype == np.int16 else 0.42}

    ap.Resnet50_Arc_loss = Resnet50_Arc_loss
    engine.HotwordDetector = HotwordDetector
    pkg.engine = engine
    pkg.audio_processing = ap
    install = fake_modules["__install__"]
    install("eff_word_net", pkg)
    install("eff_word_net.engine", engine)
    install("eff_word_net.audio_processing", ap)
    return state


def test_scores_stride_and_silence(fake_modules, tmp_path: Path, frame) -> None:
    state = _install_fake_ewn(fake_modules)
    ref = tmp_path / "ref.json"
    ref.write_text("{}")
    eng = EfficientWordNetEngine(hotword="hey michael", reference_file=str(ref), stride_frames=3)
    assert state.init == {"hotword": "hey michael", "reference_file": str(ref)}
    assert eng.info.wake_words == ("hey_michael",)
    assert eng.info.capabilities.custom_words == "enrolment"
    assert "reference" in eng.info.model_hashes
    need = -(-WINDOW_SAMPLES // FRAME_SAMPLES)  # 19 frames to cover 1.5 s
    for _ in range(need - 1):
        assert eng.process(frame) == {"hey_michael": 0.0}
    assert state.frames == []
    loud = np.full(FRAME_SAMPLES, 500, dtype=np.int16)
    for _ in range(2):
        assert eng.process(loud) == {"hey_michael": 0.0}  # since < stride
    out = eng.process(loud)  # third frame after full window -> score
    assert len(state.frames) == 1
    assert state.frames[0].dtype == np.int16
    assert out == {"hey_michael": 1.0}  # clamped from 1.5
    assert eng.process(frame) == {"hey_michael": 1.0}  # held
    eng.reset()
    assert eng.process(frame) == {"hey_michael": 0.0}


def test_float_format_and_silence_gate(fake_modules, tmp_path: Path, frame) -> None:
    state = _install_fake_ewn(fake_modules)
    ref = tmp_path / "ref.json"
    ref.write_text("{}")
    eng = EfficientWordNetEngine(
        hotword="x", reference_file=str(ref), stride_frames=1, sample_format="float32"
    )
    for _ in range(19):
        out = eng.process(frame)
    assert out == {"x": 0.0}  # silence -> None -> 0
    assert state.frames[-1].dtype == np.float32
    loud = np.full(FRAME_SAMPLES, 500, dtype=np.int16)
    assert eng.process(loud) == {"x": pytest.approx(0.42)}


def test_validation(fake_modules, tmp_path: Path) -> None:
    _install_fake_ewn(fake_modules)
    with pytest.raises(ValueError, match="reference_file"):
        EfficientWordNetEngine(hotword="x", reference_file=None)
    with pytest.raises(FileNotFoundError):
        EfficientWordNetEngine(hotword="x", reference_file=str(tmp_path / "nope.json"))
    ref = tmp_path / "ref.json"
    ref.write_text("{}")
    with pytest.raises(ValueError, match="unknown EfficientWord-Net model"):
        EfficientWordNetEngine(hotword="x", reference_file=str(ref), model="nope")


def test_unavailable(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setitem(sys.modules, "eff_word_net", None)
    ref = tmp_path / "ref.json"
    ref.write_text("{}")
    with pytest.raises(EngineUnavailableError, match="engines-ewn"):
        EfficientWordNetEngine(hotword="x", reference_file=str(ref))
