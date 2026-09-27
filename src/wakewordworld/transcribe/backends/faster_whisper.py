"""faster-whisper (CTranslate2) backend.

Verified against the faster-whisper 1.x API: ``WhisperModel(model, device, compute_type)``
and ``model.transcribe(path, word_timestamps=True, vad_filter=True, ...)`` returning an
iterator of segments with ``.start``, ``.end``, ``.text``, ``.words`` (each with
``.word``, ``.start``, ``.end``, ``.probability``). The iterator decodes lazily, so
multi-hour files are processed window by window without loading all segments first.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from wakewordworld.sources.spec import Language
from wakewordworld.transcribe.backends.base import BackendUnavailable
from wakewordworld.transcribe.schema import Segment, Transcript, Word

__all__ = ["FasterWhisperBackend"]

_INSTALL_HINT = "install with: uv sync --extra transcribe  (package: faster-whisper)"


class FasterWhisperBackend:
    """Whisper via CTranslate2 with word timestamps."""

    def __init__(
        self,
        model: str = "large-v3-turbo",
        device: str = "auto",
        compute_type: str = "auto",
        beam_size: int = 5,
        download_root: Path | None = None,
    ) -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            msg = f"faster-whisper is not installed; {_INSTALL_HINT}"
            raise BackendUnavailable(msg) from exc
        self.name = f"faster-whisper:{model}"
        self._beam_size = beam_size
        try:
            self._model = WhisperModel(
                model,
                device=device,
                compute_type=compute_type,
                download_root=str(download_root) if download_root else None,
            )
        except Exception as exc:
            msg = f"could not load faster-whisper model {model!r}: {exc}"
            raise BackendUnavailable(msg) from exc

    def transcribe(self, audio_path: Path, language: Language | None) -> Transcript:
        """Run recognition on a whole file."""
        segments_iter, info = self._model.transcribe(
            str(audio_path),
            language=str(language) if language else None,
            beam_size=self._beam_size,
            word_timestamps=True,
            vad_filter=True,
            condition_on_previous_text=False,
        )
        segments: list[Segment] = []
        for seg in segments_iter:
            words = [
                Word(
                    text=w.word.strip(),
                    start=float(w.start),
                    end=max(float(w.end), float(w.start)),
                    confidence=_clamp(getattr(w, "probability", None)),
                )
                for w in (seg.words or [])
                if w.word.strip()
            ]
            segments.append(
                Segment(
                    start=float(seg.start),
                    end=max(float(seg.end), float(seg.start)),
                    text=seg.text.strip(),
                    words=words,
                )
            )
        detected = language or _language_from_code(getattr(info, "language", None))
        duration = float(getattr(info, "duration", 0.0) or 0.0)
        if segments:
            duration = max(duration, segments[-1].end)
        return Transcript(
            file_id="",
            source_id="",
            language=detected or Language.EN,
            backend=self.name,
            origin="asr",
            duration_s=duration,
            segments=segments,
        )


def _clamp(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return None


def _language_from_code(code: str | None) -> Language | None:
    if not code:
        return None
    try:
        return Language(code.lower()[:2])
    except ValueError:
        return None
