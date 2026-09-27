from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from wakewordworld.engines.base import EngineUnavailableError
from wakewordworld.engines.openwakeword import OpenWakeWordEngine


def _install_fake_oww(
    fake_modules, scores: dict[str, float], reported_keys: list[str]
) -> types.SimpleNamespace:
    state = types.SimpleNamespace(resets=0, downloads=[], predicted=[])
    oww = types.ModuleType("openwakeword")
    oww.__version__ = "9.9.9"
    oww.__file__ = "/nonexistent/openwakeword/__init__.py"

    class Model:
        def __init__(
            self,
            wakeword_models,
            inference_framework,
            vad_threshold,
            enable_speex_noise_suppression,
        ):
            state.init = {
                "wakeword_models": wakeword_models,
                "inference_framework": inference_framework,
            }
            self.models = dict.fromkeys(reported_keys)

        def predict(self, x):
            state.predicted.append(x)
            return dict(scores)

        def reset(self):
            state.resets += 1

    model_mod = types.ModuleType("openwakeword.model")
    model_mod.Model = Model
    utils_mod = types.ModuleType("openwakeword.utils")
    utils_mod.download_models = lambda model_names: state.downloads.append(list(model_names))
    oww.model = model_mod
    oww.utils = utils_mod
    install = fake_modules["__install__"]
    install("openwakeword", oww)
    install("openwakeword.model", model_mod)
    install("openwakeword.utils", utils_mod)
    return state


def test_info_scores_and_key_normalisation(fake_modules, frame) -> None:
    state = _install_fake_oww(
        fake_modules,
        scores={"alexa": 0.3, "hey_jarvis_v0.1": 1.7, "unrelated": 0.9},
        reported_keys=["alexa", "hey_jarvis_v0.1"],
    )
    eng = OpenWakeWordEngine(models=["alexa", "hey_jarvis", "/tmp/custom_word.onnx"])
    assert eng.info.engine_id == "openwakeword"
    assert eng.info.version == "9.9.9"
    assert eng.info.wake_words == ("alexa", "hey_jarvis", "custom_word")
    assert state.downloads == [["alexa", "hey_jarvis"]]  # custom paths are not downloaded
    assert state.init["wakeword_models"] == ["alexa", "hey_jarvis", "/tmp/custom_word.onnx"]
    out = eng.process(frame)
    assert set(out) == {"alexa", "hey_jarvis", "custom_word"}
    assert out["alexa"] == pytest.approx(0.3)
    assert out["hey_jarvis"] == 1.0  # clamped, key normalised by prefix
    assert out["custom_word"] == 0.0  # not reported -> 0
    assert eng.info.capabilities.continuous_scores is True
    eng.reset()
    assert state.resets == 1
    assert len(state.predicted) == 1


def test_frame_size_enforced(fake_modules) -> None:
    _install_fake_oww(fake_modules, scores={}, reported_keys=["alexa"])
    eng = OpenWakeWordEngine(models=["alexa"])
    with pytest.raises(ValueError, match="expected 1280"):
        eng.process(np.zeros(512, dtype=np.int16))


def test_unavailable(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "openwakeword", None)
    with pytest.raises(EngineUnavailableError, match="engines-oww"):
        OpenWakeWordEngine(models=["alexa"])
