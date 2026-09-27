from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

from wakewordworld.engines.base import FRAME_SAMPLES, EngineUnavailableError
from wakewordworld.engines.sherpa_kws import SWEEP_THRESHOLDS, SherpaKwsEngine, _labels_from_file


def _model_dir(tmp_path: Path) -> Path:
    d = tmp_path / "sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01"
    d.mkdir()
    for n in ("tokens.txt", "bpe.model"):
        (d / n).write_bytes(n.encode())
    for p in ("encoder", "decoder", "joiner"):
        (d / f"{p}-x.onnx").write_bytes(b"fp32")
        (d / f"{p}-x.int8.onnx").write_bytes(b"int8")
    return d


def _install_fake_sherpa(fake_modules, *, fire_at_call: int = 3, result: str = "hey_jarvis"):
    state = types.SimpleNamespace(spotter=None, streams=0, decodes=0, resets=0, waveforms=[])
    mod = types.ModuleType("sherpa_onnx")

    class Stream:
        def __init__(self):
            self.pending = 0

        def accept_waveform(self, sr, samples):
            assert sr == 16000
            assert samples.dtype == np.float32
            state.waveforms.append(samples.copy())
            self.pending = 2

    class KeywordSpotter:
        def __init__(self, **kwargs):
            state.spotter = kwargs

        def create_stream(self, keywords=None):
            state.streams += 1
            return Stream()

        def is_ready(self, s):
            return s.pending > 0

        def decode_stream(self, s):
            s.pending -= 1
            state.decodes += 1

        def get_result(self, s):
            return result if state.decodes == fire_at_call else ""

        def reset_stream(self, s):
            state.resets += 1

    def text2token(texts, tokens, tokens_type, bpe_model):
        assert tokens_type == "bpe"
        return [["▁" + t for t in x.split()] for x in texts]

    mod.KeywordSpotter = KeywordSpotter
    mod.text2token = text2token
    fake_modules["__install__"]("sherpa_onnx", mod)
    return state


def test_phrases_to_keywords_and_hits(fake_modules, tmp_path: Path, monkeypatch, frame) -> None:
    monkeypatch.setenv("WWW_DATA_ROOT", str(tmp_path / "data"))
    state = _install_fake_sherpa(fake_modules, fire_at_call=3)
    eng = SherpaKwsEngine(
        keywords=["hey jarvis", "alexa"],
        model_dir=str(_model_dir(tmp_path)),
        keywords_threshold=0.3,
    )
    assert eng.info.engine_id == "sherpa_kws"
    assert eng.info.wake_words == ("hey_jarvis", "alexa")
    assert state.spotter["keywords_threshold"] == 0.3
    assert state.spotter["encoder"].endswith(".int8.onnx")
    kw = Path(state.spotter["keywords_file"]).read_text()
    assert "▁HEY ▁JARVIS :1.0 #0.3 @hey_jarvis" in kw
    assert "▁ALEXA :1.0 #0.3 @alexa" in kw
    caps = eng.info.capabilities
    assert caps.continuous_scores is False
    assert caps.sweep_param == "keywords_threshold"
    assert caps.sweep_values == SWEEP_THRESHOLDS
    assert caps.languages == ("en",)
    assert eng.process(frame) == {"hey_jarvis": 0.0, "alexa": 0.0}  # decodes 1,2
    assert eng.process(frame) == {"hey_jarvis": 1.0, "alexa": 0.0}  # decode 3 fires
    assert state.resets == 1
    assert eng.process(frame) == {"hey_jarvis": 0.0, "alexa": 0.0}
    assert state.waveforms[0].shape == (FRAME_SAMPLES,)
    eng.reset()
    assert state.streams == 2


def test_keywords_file_labels(tmp_path: Path) -> None:
    p = tmp_path / "kw.txt"
    p.write_text("▁HE Y ▁S I RI :1.0 #0.25 @hey_siri\n▁A LE X A\n\n")
    assert _labels_from_file(p) == ["hey_siri", "▁a_le_x_a"]


def test_fp32_selection_and_validation(fake_modules, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("WWW_DATA_ROOT", str(tmp_path / "data"))
    state = _install_fake_sherpa(fake_modules)
    eng = SherpaKwsEngine(keywords=["alexa"], model_dir=str(_model_dir(tmp_path)), int8=False)
    assert not state.spotter["encoder"].endswith(".int8.onnx")
    assert eng.info.config["int8"] is False
    with pytest.raises(ValueError, match="keywords"):
        SherpaKwsEngine(model_dir=str(tmp_path))


def test_unavailable(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "sherpa_onnx", None)
    with pytest.raises(EngineUnavailableError, match="engines-sherpa"):
        SherpaKwsEngine(keywords=["alexa"], model_dir="/tmp")
