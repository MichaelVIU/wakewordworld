from __future__ import annotations

import enum
import sys
import types

import numpy as np
import pytest

from wakewordworld.engines.base import EngineUnavailableError
from wakewordworld.engines.microwakeword import MicroWakeWordEngine


def _install_fake_mww(
    fake_modules, probs_per_call: list[float], n_windows: int = 3
) -> types.SimpleNamespace:
    state = types.SimpleNamespace(resets=0, feature_resets=0, closed=0, calls=0)

    class Model(enum.StrEnum):
        OKAY_NABU = "okay_nabu"
        HEY_JARVIS = "hey_jarvis"

    class MicroWakeWord:
        def __init__(self, ident: str):
            self.id = ident
            self.wake_word = ident.replace("_", " ").title()
            self.probability_cutoff = 0.85
            self.sliding_window_size = 5
            self.stride = 3
            self.trained_languages = ["en", "de"]
            self.model_path = f"/nonexistent/{ident}.tflite"
            self._probabilities: list[float] = []

        @classmethod
        def from_builtin(cls, model, models_dir=None):
            return cls(model.value)

        @classmethod
        def from_config(cls, path):
            return cls("custom")

        def process_streaming_prob(self, feats):
            p = probs_per_call[state.calls % len(probs_per_call)]
            state.calls += 1
            self._probabilities.append(p * 2)  # "raw" value differs from windowed
            return p

        def process_streaming(self, feats):
            return self.process_streaming_prob(feats) > self.probability_cutoff

        def reset(self):
            state.resets += 1

        def close(self):
            state.closed += 1

    class MicroWakeWordFeatures:
        def process_streaming(self, audio_bytes: bytes):
            assert len(audio_bytes) == 1280 * 2
            for _ in range(n_windows):
                yield np.zeros((1, 1, 40), dtype=np.float32)

        def reset(self):
            state.feature_resets += 1

    mod = types.ModuleType("pymicro_wakeword")
    mod.__version__ = "2.5.0"
    mod.Model = Model
    mod.MicroWakeWord = MicroWakeWord
    mod.MicroWakeWordFeatures = MicroWakeWordFeatures
    fake_modules["__install__"]("pymicro_wakeword", mod)
    return state


def test_max_over_windows_and_info(fake_modules, frame) -> None:
    state = _install_fake_mww(fake_modules, probs_per_call=[0.1, 0.4, 0.2])
    eng = MicroWakeWordEngine(models=["okay_nabu"])
    assert eng.info.wake_words == ("okay_nabu",)
    assert eng.info.capabilities.languages == ("de", "en")
    assert eng.info.config["cutoffs"] == {"okay_nabu": 0.85}
    out = eng.process(frame)
    assert out == {"okay_nabu": pytest.approx(0.4)}  # max over the 3 windows
    eng.reset()
    assert state.resets == 1
    assert state.feature_resets == 1
    eng.close()
    assert state.closed == 1


def test_raw_and_native_modes(fake_modules, frame) -> None:
    _install_fake_mww(fake_modules, probs_per_call=[0.3, 0.9, 0.2])
    raw = MicroWakeWordEngine(models=["okay_nabu"], score="raw")
    assert raw.process(frame)["okay_nabu"] == 1.0  # raw = 2 * 0.9 clamped
    native = MicroWakeWordEngine(models=["okay_nabu"], native_detection=True)
    assert native.info.capabilities.continuous_scores is False
    assert native.process(frame)["okay_nabu"] == 1.0  # 0.9 > cutoff


def test_multiple_models_each_scored(fake_modules, frame) -> None:
    _install_fake_mww(fake_modules, probs_per_call=[0.5], n_windows=1)
    eng = MicroWakeWordEngine(models=["okay_nabu", "hey_jarvis"])
    out = eng.process(frame)
    assert set(out) == {"okay_nabu", "hey_jarvis"}
    assert all(v == pytest.approx(0.5) for v in out.values())


def test_bad_score_mode(fake_modules) -> None:
    _install_fake_mww(fake_modules, probs_per_call=[0.0])
    with pytest.raises(ValueError, match="score must be"):
        MicroWakeWordEngine(models=["okay_nabu"], score="mean")


def test_unavailable(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "pymicro_wakeword", None)
    with pytest.raises(EngineUnavailableError, match="engines-mww"):
        MicroWakeWordEngine(models=["okay_nabu"])
