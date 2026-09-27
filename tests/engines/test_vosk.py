from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from wakewordworld.engines.base import EngineUnavailableError
from wakewordworld.engines.vosk import VoskEngine, _contains


def _install_fake_vosk(fake_modules, script: list[tuple[bool, dict]]) -> types.SimpleNamespace:
    """``script`` is a list of (accepted, payload) consumed one per AcceptWaveform call."""
    state = types.SimpleNamespace(recognisers=0, grammar=None, log_level=None, words_set=0)

    class Model:
        def __init__(self, model_path=None, model_name=None, lang=None):
            state.model_path = model_path

    class KaldiRecognizer:
        def __init__(self, model, rate, grammar):
            state.recognisers += 1
            state.grammar = json.loads(grammar)
            state.raw_grammar = grammar
            self._i = 0
            self._last = None

        def SetWords(self, flag):  # noqa: N802
            state.words_set += 1

        def AcceptWaveform(self, data):  # noqa: N802
            assert len(data) == 1280 * 2
            accepted, payload = script[self._i]
            self._i += 1
            self._last = payload
            return accepted

        def Result(self):  # noqa: N802
            return json.dumps(self._last)

        def PartialResult(self):  # noqa: N802
            return json.dumps(self._last)

    mod = types.ModuleType("vosk")
    mod.Model = Model
    mod.KaldiRecognizer = KaldiRecognizer
    mod.SetLogLevel = lambda lvl: setattr(state, "log_level", lvl)
    fake_modules["__install__"]("vosk", mod)
    return state


def test_partial_then_final_fires_once(fake_modules, frame, tmp_path: Path) -> None:
    (tmp_path / "vosk-model-small-en-us-0.15").mkdir()
    (tmp_path / "vosk-model-small-en-us-0.15" / "am").write_bytes(b"x")
    script = [
        (False, {"partial": "hey"}),
        (False, {"partial": "hey computer"}),
        (False, {"partial": "hey computer please"}),
        (
            True,
            {
                "text": "hey computer please",
                "result": [
                    {"word": "hey", "conf": 0.9},
                    {"word": "computer", "conf": 0.6},
                    {"word": "please", "conf": 0.5},
                ],
            },
        ),
        (False, {"partial": "hey computer"}),  # new utterance -> fires again
    ]
    state = _install_fake_vosk(fake_modules, script)
    eng = VoskEngine(
        wake_words=["hey computer", "alexa"],
        model_path=str(tmp_path / "vosk-model-small-en-us-0.15"),
    )
    assert state.grammar == ["hey computer", "alexa", "[unk]"]
    assert state.log_level == -1
    assert eng.info.capabilities.continuous_scores is False
    assert eng.info.capabilities.languages == ("en",)
    assert eng.info.wake_words == ("hey computer", "alexa")
    outs = [eng.process(frame) for _ in range(5)]
    assert [o["hey computer"] for o in outs] == [0.0, 1.0, 0.0, 0.0, 1.0]
    assert all(o["alexa"] == 0.0 for o in outs)


def test_final_only_uses_min_confidence(fake_modules, frame, tmp_path: Path) -> None:
    (tmp_path / "m").mkdir()
    script = [
        (True, {"text": "alexa", "result": [{"word": "alexa", "conf": 0.42}]}),
    ]
    _install_fake_vosk(fake_modules, script)
    eng = VoskEngine(wake_words=["alexa"], model_path=str(tmp_path / "m"))
    assert eng.process(frame)["alexa"] == pytest.approx(0.42)


def test_reset_makes_new_recogniser(fake_modules, frame, tmp_path: Path) -> None:
    (tmp_path / "m").mkdir()
    state = _install_fake_vosk(fake_modules, [(False, {"partial": ""})])
    eng = VoskEngine(wake_words=["alexa"], model_path=str(tmp_path / "m"))
    assert state.recognisers == 1
    eng.reset()
    assert state.recognisers == 2
    assert state.words_set == 2


def test_requires_wake_words(fake_modules, tmp_path: Path) -> None:
    _install_fake_vosk(fake_modules, [])
    with pytest.raises(ValueError, match="at least one wake word"):
        VoskEngine(wake_words=[], model_path=str(tmp_path))


def test_contains() -> None:
    assert _contains(["a", "hey", "computer", "b"], ["hey", "computer"]) == 1
    assert _contains(["hey", "there", "computer"], ["hey", "computer"]) is None


def test_unavailable(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "vosk", None)
    with pytest.raises(EngineUnavailableError, match="engines-vosk"):
        VoskEngine(wake_words=["alexa"], model_path="/tmp/x")


def test_grammar_keeps_non_ascii_phrases(fake_modules) -> None:  # type: ignore[no-untyped-def]
    """Accented wake words must reach Vosk unescaped (Vosk does not decode JSON escapes)."""
    state = _install_fake_vosk(fake_modules, [(False, {"partial": ""})])
    from wakewordworld.engines.vosk import VoskEngine

    VoskEngine(wake_words=["víctor"], model_path="/tmp/model")
    assert "víctor" in state.raw_grammar
    assert "\\u00ed" not in state.raw_grammar
