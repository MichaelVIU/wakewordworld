"""Vosk keyword-grammar adapter (ASR baseline, https://alphacephei.com/vosk/).

Vosk is a full speech recogniser; constraining it with a grammar of the wake phrases
plus ``[unk]`` turns it into an open-vocabulary keyword spotter. It exposes no graded
score, so this is a boolean-style engine: a wake phrase scores ``1.0`` on the frame
where it first appears in the recogniser's partial or final text for the current
utterance and ``0.0`` otherwise. When the phrase first appears in a *final* result the
score is the minimum per-word confidence Vosk reports for the phrase's words.

Models are downloaded from ``https://alphacephei.com/vosk/models/<name>.zip`` into the
benchmark cache and their archive checksum is recorded in ``info.model_hashes``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from wakewordworld.engines.base import (
    FRAME_SAMPLES,
    SAMPLE_RATE,
    Capabilities,
    Engine,
    EngineInfo,
    EngineUnavailableError,
    engine_registry,
)
from wakewordworld.engines.models import download_file, hash_models, models_dir, unzip_model
from wakewordworld.util.paths import DataRoot

__all__ = ["VoskEngine", "create", "ensure_vosk_model"]

_INSTALL_HINT = "install with: uv sync --extra engines-vosk  (package: vosk)"
MODEL_BASE_URL = "https://alphacephei.com/vosk/models/"
_LANG_RE = re.compile(r"vosk-model-(?:small-)?([a-z]{2})(?:-[a-z]{2})?-")


def ensure_vosk_model(
    model_name: str, *, data_root: DataRoot | None = None, expected_sha256: str | None = None
) -> tuple[Path, str]:
    """Download and extract a Vosk model by name; returns (model dir, archive sha256)."""
    target = models_dir("vosk", data_root)
    folder = target / model_name
    archive = target / f"{model_name}.zip"
    if folder.is_dir() and any(folder.iterdir()) and not archive.exists():
        return folder, "unpinned-preinstalled"
    digest = download_file(
        f"{MODEL_BASE_URL}{model_name}.zip", archive, expected_sha256=expected_sha256
    )
    unzip_model(archive, target)
    return folder, digest


def _phrase_tokens(phrase: str) -> list[str]:
    return phrase.lower().split()


def _contains(tokens: list[str], phrase: list[str]) -> int | None:
    """Index where ``phrase`` occurs as consecutive tokens, or None."""
    n = len(phrase)
    for i in range(len(tokens) - n + 1):
        if tokens[i : i + n] == phrase:
            return i
    return None


class VoskEngine:
    """Streaming adapter around ``vosk.KaldiRecognizer`` with a keyword grammar."""

    def __init__(
        self,
        wake_words: list[str] | None = None,
        model_name: str | None = "vosk-model-small-en-us-0.15",
        model_path: str | None = None,
        model_sha256: str | None = None,
        log_level: int = -1,
    ) -> None:
        try:
            import vosk
        except ImportError as exc:  # pragma: no cover - exercised via fake module tests
            raise EngineUnavailableError(f"vosk not importable; {_INSTALL_HINT}") from exc
        if not wake_words:
            msg = "vosk engine needs at least one wake word phrase"
            raise ValueError(msg)
        set_log = getattr(vosk, "SetLogLevel", None)
        if callable(set_log):
            set_log(log_level)

        archive_hash = "unpinned"
        if model_path:
            path = Path(model_path)
            name = path.name
        else:
            assert model_name is not None
            path, archive_hash = ensure_vosk_model(model_name, expected_sha256=model_sha256)
            name = model_name
        self._model = vosk.Model(model_path=str(path))
        self._phrases = tuple(wake_words)
        self._tokens = {p: _phrase_tokens(p) for p in self._phrases}
        # Vosk does not decode JSON escapes, so non-ASCII phrases ("víctor") must be
        # passed verbatim or they are silently dropped as out-of-vocabulary.
        grammar = json.dumps(
            [*[" ".join(t) for t in self._tokens.values()], "[unk]"], ensure_ascii=False
        )
        self._grammar = grammar
        self._vosk = vosk
        self._rec = vosk.KaldiRecognizer(self._model, SAMPLE_RATE, grammar)
        self._rec.SetWords(True)
        self._fired: set[str] = set()
        lang_m = _LANG_RE.match(name)
        hashes = {name: archive_hash}
        hashes.update({f"{name}/dir": v for v in hash_models({name: path}).values()})
        self._info = EngineInfo(
            engine_id="vosk",
            version=_dist_version("vosk"),
            wake_words=self._phrases,
            model_hashes=hashes,
            config={"model": name, "grammar": grammar},
            capabilities=Capabilities(
                continuous_scores=False,
                custom_words="text",
                languages=(lang_m.group(1),) if lang_m else (),
            ),
        )

    @property
    def info(self) -> EngineInfo:
        """Identity and capabilities."""
        return self._info

    def reset(self) -> None:
        """Start a fresh recogniser state."""
        self._rec = self._vosk.KaldiRecognizer(self._model, SAMPLE_RATE, self._grammar)
        self._rec.SetWords(True)
        self._fired.clear()

    def process(self, frame: NDArray[np.int16]) -> Mapping[str, float]:
        """Feed one frame; fire a phrase the first time it appears in the utterance text."""
        if frame.shape[0] != FRAME_SAMPLES:
            msg = f"expected {FRAME_SAMPLES} samples, got {frame.shape[0]}"
            raise ValueError(msg)
        out = dict.fromkeys(self._phrases, 0.0)
        if self._rec.AcceptWaveform(frame.astype(np.int16).tobytes()):
            res = json.loads(self._rec.Result() or "{}")
            words = res.get("result") or []
            tokens = [str(w.get("word", "")).lower() for w in words] or str(
                res.get("text", "")
            ).lower().split()
            for phrase, ptoks in self._tokens.items():
                if phrase in self._fired:
                    continue
                idx = _contains(tokens, ptoks)
                if idx is None:
                    continue
                confs = [float(w.get("conf", 1.0)) for w in words[idx : idx + len(ptoks)]]
                out[phrase] = min(confs) if confs else 1.0
            self._fired.clear()
        else:
            partial = json.loads(self._rec.PartialResult() or "{}")
            tokens = str(partial.get("partial", "")).lower().split()
            for phrase, ptoks in self._tokens.items():
                if phrase not in self._fired and _contains(tokens, ptoks) is not None:
                    out[phrase] = 1.0
                    self._fired.add(phrase)
        return {k: min(1.0, max(0.0, v)) for k, v in out.items()}

    def close(self) -> None:
        """Drop the recogniser."""
        self._rec = None


def _dist_version(name: str) -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


@engine_registry.register("vosk")
def create(**config: Any) -> Engine:
    """Factory used by the registry."""
    return VoskEngine(**config)
