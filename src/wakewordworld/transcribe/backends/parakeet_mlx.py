"""NVIDIA Parakeet-TDT via ``parakeet-mlx`` (Apple Silicon only).

API assumptions (parakeet-mlx 0.3.x, checked from its README, not against a local
install at implementation time): ``from parakeet_mlx import from_pretrained``;
``model.transcribe(path, chunk_duration=..., overlap_duration=...)`` returns an
``AlignedResult`` with ``.sentences``; each sentence has ``.start``, ``.end``, ``.text``
and ``.tokens``; tokens carry ``.text``, ``.start``, ``.end`` and are sub-word pieces
that start with a space at word boundaries (SentencePiece style). If the installed
package deviates, :class:`BackendUnavailable` is raised with the observed shape instead
of guessing.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from wakewordworld.sources.spec import Language
from wakewordworld.transcribe.backends.base import BackendUnavailable
from wakewordworld.transcribe.schema import Segment, Transcript, Word

__all__ = ["ParakeetMlxBackend"]

_INSTALL_HINT = "install with: uv sync --extra transcribe-mlx  (package: parakeet-mlx, macOS only)"


class ParakeetMlxBackend:
    """Parakeet-TDT 0.6B v3 (25 European languages) on MLX."""

    def __init__(
        self,
        model: str = "mlx-community/parakeet-tdt-0.6b-v3",
        chunk_duration: float = 120.0,
        overlap_duration: float = 15.0,
    ) -> None:
        if sys.platform != "darwin":
            msg = "parakeet-mlx runs on macOS (Apple Silicon) only"
            raise BackendUnavailable(msg)
        try:
            from parakeet_mlx import from_pretrained
        except ImportError as exc:
            msg = f"parakeet-mlx is not installed; {_INSTALL_HINT}"
            raise BackendUnavailable(msg) from exc
        self.name = f"parakeet-mlx:{model.rsplit('/', 1)[-1]}"
        self._chunk = chunk_duration
        self._overlap = overlap_duration
        try:
            self._model = from_pretrained(model)
        except Exception as exc:
            msg = f"could not load parakeet-mlx model {model!r}: {exc}"
            raise BackendUnavailable(msg) from exc

    def transcribe(self, audio_path: Path, language: Language | None) -> Transcript:
        """Run recognition; the model is multilingual and ignores ``language``."""
        try:
            result = self._model.transcribe(
                str(audio_path),
                chunk_duration=self._chunk,
                overlap_duration=self._overlap,
            )
        except TypeError:
            # Older versions without chunking keyword arguments.
            result = self._model.transcribe(str(audio_path))
        sentences = getattr(result, "sentences", None)
        if sentences is None:
            msg = (
                "parakeet-mlx returned an unexpected result type "
                f"{type(result).__name__} without .sentences; refusing to guess"
            )
            raise BackendUnavailable(msg)
        segments = [_sentence_to_segment(s) for s in sentences]
        segments = [s for s in segments if s.text]
        duration = segments[-1].end if segments else 0.0
        return Transcript(
            file_id="",
            source_id="",
            language=language or Language.EN,
            backend=self.name,
            origin="asr",
            duration_s=duration,
            segments=segments,
        )


def _sentence_to_segment(sentence: Any) -> Segment:
    tokens = list(getattr(sentence, "tokens", []) or [])
    words: list[Word] = []
    cur_text = ""
    cur_start = 0.0
    cur_end = 0.0
    for tok in tokens:
        text = str(getattr(tok, "text", ""))
        t_start = float(getattr(tok, "start", 0.0))
        t_end = float(getattr(tok, "end", t_start))
        if text.startswith(" ") or not cur_text:
            if cur_text.strip():
                words.append(
                    Word(text=cur_text.strip(), start=cur_start, end=max(cur_end, cur_start))
                )
            cur_text = text
            cur_start = t_start
        else:
            cur_text += text
        cur_end = t_end
    if cur_text.strip():
        words.append(Word(text=cur_text.strip(), start=cur_start, end=max(cur_end, cur_start)))
    start = float(getattr(sentence, "start", words[0].start if words else 0.0))
    end = float(getattr(sentence, "end", words[-1].end if words else start))
    return Segment(
        start=start,
        end=max(end, start),
        text=str(getattr(sentence, "text", "")).strip(),
        words=words,
    )
