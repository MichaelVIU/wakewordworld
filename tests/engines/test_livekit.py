from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

from wakewordworld.engines.base import FRAME_SAMPLES, EngineUnavailableError
from wakewordworld.engines.livekit import WINDOW_FRAMES, LiveKitEngine


def _install_fake_livekit(fake_modules, tmp_path: Path) -> types.SimpleNamespace:
    state = types.SimpleNamespace(models=None, calls=[])
    mel = tmp_path / "melspectrogram.onnx"
    emb = tmp_path / "embedding_model.onnx"
    mel.write_bytes(b"mel")
    emb.write_bytes(b"emb")

    class WakeWordModel:
        def __init__(self, models=None):
            state.models = list(models or [])

        def predict(self, chunk):
            state.calls.append(np.asarray(chunk).copy())
            # score = fraction of non-zero samples, scaled, for the first model; 2.0 for the second
            frac = float(np.count_nonzero(chunk)) / chunk.shape[0]
            return {"hey_livekit": frac, "other": 2.0}

    pkg = types.ModuleType("livekit")
    wake = types.ModuleType("livekit.wakeword")
    inf = types.ModuleType("livekit.wakeword.inference")
    model_mod = types.ModuleType("livekit.wakeword.inference.model")
    model_mod.WakeWordModel = WakeWordModel
    model_mod.get_mel_model_path = lambda: mel
    model_mod.get_embedding_model_path = lambda: emb
    inf.model = model_mod
    wake.inference = inf
    pkg.wakeword = wake
    install = fake_modules["__install__"]
    install("livekit", pkg)
    install("livekit.wakeword", wake)
    install("livekit.wakeword.inference", inf)
    install("livekit.wakeword.inference.model", model_mod)
    return state


def test_window_scoring_and_hold(fake_modules, tmp_path: Path, frame) -> None:
    state = _install_fake_livekit(fake_modules, tmp_path)
    clf = tmp_path / "hey_livekit.onnx"
    clf.write_bytes(b"clf")
    eng = LiveKitEngine(models=[str(clf), str(tmp_path / "other.onnx")], stride_frames=2)
    assert state.models == [str(clf), str(tmp_path / "other.onnx")]
    assert eng.info.engine_id == "livekit"
    assert eng.info.wake_words == ("hey_livekit", "other")
    assert set(eng.info.model_hashes) == {"hey_livekit", "melspectrogram", "embedding_model"}
    assert eng.info.capabilities.continuous_scores is True
    assert eng.info.capabilities.custom_words == "training"
    # Before the 2 s window is full: zeros and no predict calls.
    for _ in range(WINDOW_FRAMES - 1):
        assert eng.process(frame) == {"hey_livekit": 0.0, "other": 0.0}
    assert state.calls == []
    loud = np.full(FRAME_SAMPLES, 1000, dtype=np.int16)
    out = eng.process(loud)  # window full, since=1 < stride -> still held zeros
    assert out == {"hey_livekit": 0.0, "other": 0.0}
    out = eng.process(loud)  # since=2 -> predict on 25 frames (2 loud of 25)
    assert len(state.calls) == 1
    assert state.calls[0].shape == (WINDOW_FRAMES * FRAME_SAMPLES,)
    assert out["hey_livekit"] == pytest.approx(2 / WINDOW_FRAMES)
    assert out["other"] == 1.0  # clamped
    held = eng.process(frame)  # held until next stride
    assert held == out
    eng.reset()
    assert eng.process(frame) == {"hey_livekit": 0.0, "other": 0.0}
    assert len(state.calls) == 1


def test_requires_models(fake_modules, tmp_path: Path) -> None:
    _install_fake_livekit(fake_modules, tmp_path)
    with pytest.raises(ValueError, match="classifier"):
        LiveKitEngine(models=[])


def test_frame_size_enforced(fake_modules, tmp_path: Path) -> None:
    _install_fake_livekit(fake_modules, tmp_path)
    eng = LiveKitEngine(models=[str(tmp_path / "w.onnx")])
    with pytest.raises(ValueError, match="expected 1280"):
        eng.process(np.zeros(100, dtype=np.int16))


def test_unavailable(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "livekit", None)
    monkeypatch.setitem(sys.modules, "livekit.wakeword", None)
    monkeypatch.setitem(sys.modules, "livekit.wakeword.inference", None)
    monkeypatch.setitem(sys.modules, "livekit.wakeword.inference.model", None)
    with pytest.raises(EngineUnavailableError, match="engines-livekit"):
        LiveKitEngine(models=["x.onnx"])
