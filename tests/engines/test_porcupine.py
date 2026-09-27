from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from wakewordworld.engines.base import FRAME_SAMPLES, EngineUnavailableError
from wakewordworld.engines.porcupine import PorcupineEngine, _wake_word_name

MARK = 12345  # a sample value that makes the fake engine fire


def _install_fake_pv(fake_modules, *, fire_on_mark: bool = True) -> types.SimpleNamespace:
    state = types.SimpleNamespace(created=None, frames=[], deleted=0)
    pv = types.ModuleType("pvporcupine")

    class PorcupineError(Exception):
        pass

    class PorcupineActivationError(PorcupineError):
        pass

    class Handle:
        frame_length = 512
        sample_rate = 16000
        version = "4.0.3-fake"

        def __init__(self, n_keywords: int) -> None:
            self.n = n_keywords

        def process(self, pcm):
            arr = np.asarray(pcm)
            assert arr.shape == (512,)
            state.frames.append(arr.copy())
            if fire_on_mark and MARK in arr:
                return 1 if self.n > 1 else 0
            return -1

        def delete(self):
            state.deleted += 1

    def create(**kwargs):
        state.created = kwargs
        if kwargs.get("access_key") == "bad":
            raise PorcupineActivationError("AccessKey is invalid")
        n = len(kwargs.get("keywords", []) or []) + len(kwargs.get("keyword_paths", []) or [])
        return Handle(n)

    pv.create = create
    pv.PorcupineError = PorcupineError
    pv.PorcupineActivationError = PorcupineActivationError
    pv.KEYWORD_PATHS = {"alexa": "/nonexistent/alexa.ppn", "jarvis": "/nonexistent/jarvis.ppn"}
    fake_modules["__install__"]("pvporcupine", pv)
    return state


def test_requires_access_key(fake_modules, monkeypatch) -> None:
    _install_fake_pv(fake_modules)
    monkeypatch.delenv("PICOVOICE_ACCESS_KEY", raising=False)
    with pytest.raises(EngineUnavailableError, match="access key"):
        PorcupineEngine(keywords=["alexa"])


def test_activation_error_is_unavailable(fake_modules) -> None:
    _install_fake_pv(fake_modules)
    with pytest.raises(EngineUnavailableError, match="could not start"):
        PorcupineEngine(keywords=["alexa"], access_key="bad")


def test_info_and_sweep(fake_modules, monkeypatch) -> None:
    state = _install_fake_pv(fake_modules)
    monkeypatch.setenv("PICOVOICE_ACCESS_KEY", "k")
    eng = PorcupineEngine(
        keywords=["alexa", "jarvis"],
        keyword_paths=["/tmp/Hey-Michael_de_mac_v3_0_0.ppn"],
        sensitivity=0.7,
        model_path="/tmp/porcupine_params_de.pv",
    )
    assert state.created["access_key"] == "k"
    assert state.created["sensitivities"] == [0.7, 0.7, 0.7]
    assert eng.info.engine_id == "porcupine"
    assert eng.info.version == "4.0.3-fake"
    assert eng.info.wake_words == ("alexa", "jarvis", "hey_michael")
    caps = eng.info.capabilities
    assert caps.continuous_scores is False
    assert caps.sweep_param == "sensitivity"
    assert 0.1 in caps.sweep_values
    assert caps.custom_words == "text"
    assert caps.languages == ("de",)
    eng.close()
    assert state.deleted == 1


def test_buffering_alignment_and_remainder(fake_modules, monkeypatch) -> None:
    state = _install_fake_pv(fake_modules)
    monkeypatch.setenv("PICOVOICE_ACCESS_KEY", "k")
    eng = PorcupineEngine(keywords=["alexa", "jarvis"])
    f1 = np.arange(FRAME_SAMPLES, dtype=np.int16)  # 0..1279
    out1 = eng.process(f1)
    assert out1 == {"alexa": 0.0, "jarvis": 0.0}
    # 1280 = 2 * 512 + 256 -> two sub-frames processed, 256 carried
    assert len(state.frames) == 2
    assert state.frames[0][0] == 0
    assert state.frames[1][0] == 512
    f2 = np.full(FRAME_SAMPLES, 7, dtype=np.int16)
    f2[10] = MARK  # lands in the sub-frame that starts with the carried remainder
    out2 = eng.process(f2)
    assert len(state.frames) == 5  # 256 + 1280 = 1536 = 3 * 512, remainder 0
    assert state.frames[2][:256].tolist() == list(range(1024, 1280))
    assert state.frames[2][256 + 10] == MARK
    assert out2 == {"alexa": 0.0, "jarvis": 1.0}
    f3 = np.zeros(FRAME_SAMPLES, dtype=np.int16)
    eng.process(f3)
    assert len(state.frames) == 7  # 0 + 1280 -> 2 sub-frames, remainder 256
    eng.reset()
    eng.process(f3)
    assert len(state.frames) == 9  # remainder dropped by reset: again 2 sub-frames
    assert state.frames[7][0] == 0


def test_frame_size_enforced(fake_modules, monkeypatch) -> None:
    _install_fake_pv(fake_modules)
    monkeypatch.setenv("PICOVOICE_ACCESS_KEY", "k")
    eng = PorcupineEngine(keywords=["alexa"])
    with pytest.raises(ValueError, match="expected 1280"):
        eng.process(np.zeros(512, dtype=np.int16))


def test_wake_word_names() -> None:
    assert _wake_word_name("alexa") == "alexa"
    assert _wake_word_name("Hey-Michael_en_linux_v3_0_0.ppn") == "hey_michael"
    assert _wake_word_name("/x/Okay Nabu_de_raspberry-pi_v3_0_0.ppn") == "okay_nabu"


def test_unavailable(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "pvporcupine", None)
    with pytest.raises(EngineUnavailableError, match="engines-porcupine"):
        PorcupineEngine(keywords=["alexa"], access_key="k")
