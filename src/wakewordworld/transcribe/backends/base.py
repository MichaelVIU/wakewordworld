"""Backend protocol and factory.

Heavy dependencies (CTranslate2, MLX, PyTorch) are imported lazily inside the
backends so that the package imports and the CLI starts without them installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from wakewordworld.sources.spec import Language
from wakewordworld.transcribe.schema import Transcript

__all__ = ["BACKEND_NAMES", "AsrBackend", "BackendUnavailable", "check_backends", "get_backend"]

BACKEND_NAMES: tuple[str, ...] = ("faster-whisper", "parakeet-mlx")


class BackendUnavailable(RuntimeError):  # noqa: N818 - name fixed by the module contract
    """Raised when a backend's optional dependency is missing or its API is unexpected."""


@runtime_checkable
class AsrBackend(Protocol):
    """Speech recogniser that returns word timings."""

    name: str

    def transcribe(self, audio_path: Path, language: Language | None) -> Transcript:
        """Transcribe one 16 kHz mono file.

        The returned transcript carries ``file_id``/``source_id`` as empty strings; the
        pipeline fills them in. ``origin`` must be ``"asr"``.
        """
        ...


def get_backend(name: str, **opts: Any) -> AsrBackend:
    """Instantiate a backend by name.

    Args:
        name: One of :data:`BACKEND_NAMES`.
        **opts: Backend options (``model``, ``device``, ``compute_type`` ...).

    Raises:
        BackendUnavailable: When the backend's dependency is missing.
        ValueError: For an unknown backend name.
    """
    if name == "faster-whisper":
        from wakewordworld.transcribe.backends.faster_whisper import FasterWhisperBackend

        return FasterWhisperBackend(**opts)
    if name == "parakeet-mlx":
        from wakewordworld.transcribe.backends.parakeet_mlx import ParakeetMlxBackend

        return ParakeetMlxBackend(**opts)
    msg = f"unknown ASR backend {name!r}; choose from {', '.join(BACKEND_NAMES)}"
    raise ValueError(msg)


def check_backends() -> dict[str, str | None]:
    """Try to construct each backend; map name to ``None`` (ok) or the failure reason."""
    result: dict[str, str | None] = {}
    for name in BACKEND_NAMES:
        try:
            get_backend(name)
        except BackendUnavailable as exc:
            result[name] = str(exc)
        except Exception as exc:
            result[name] = f"{type(exc).__name__}: {exc}"
        else:
            result[name] = None
    return result
