"""LiveKit wakeword adapter (https://github.com/livekit/livekit-wakeword).

``livekit.wakeword.WakeWordModel`` is stateless: it scores a complete ~2 s window
(25 harness frames of 1280 samples) each time. The shipped listener keeps a sliding
deque of the last 25 frames and calls ``predict`` after every new frame; the adapter
reproduces exactly that. Frames received before the window is full score ``0.0``.

The package bundles only the mel front end and the speech-embedding backbone; wake
word classifiers (``.onnx``) are produced by ``livekit-wakeword`` training or obtained
from third parties and passed as ``models``. Wake word keys are the classifier file
stems.

Config:
    ``models``: list of classifier ``.onnx`` paths (required).
    ``stride_frames``: run ``predict`` every N frames (default 1 = the listener's
    behaviour); the last score is held on frames in between.
"""

from __future__ import annotations

import contextlib
from collections import deque
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from wakewordworld.engines.base import (
    FRAME_SAMPLES,
    Capabilities,
    Engine,
    EngineInfo,
    EngineUnavailableError,
    engine_registry,
)
from wakewordworld.engines.models import hash_models

__all__ = ["LiveKitEngine", "create"]

_INSTALL_HINT = "install with: uv sync --extra engines-livekit  (package: livekit-wakeword)"
WINDOW_FRAMES = 25  # CHUNK_FRAMES in livekit.wakeword.inference.listener (2.0 s)


class LiveKitEngine:
    """Streaming adapter around ``livekit.wakeword.WakeWordModel``."""

    def __init__(self, models: list[str] | None = None, stride_frames: int = 1) -> None:
        try:
            from livekit.wakeword.inference import model as lk_model
        except ImportError as exc:  # pragma: no cover - exercised via fake module tests
            raise EngineUnavailableError(
                f"livekit.wakeword not importable; {_INSTALL_HINT}"
            ) from exc
        if not models:
            msg = (
                "livekit engine needs at least one classifier .onnx in `models`; the "
                "package ships no wake word classifiers"
            )
            raise ValueError(msg)
        if stride_frames < 1:
            msg = "stride_frames must be >= 1"
            raise ValueError(msg)
        paths = [Path(m) for m in models]
        self._wake_words = tuple(p.stem for p in paths)
        self._model = lk_model.WakeWordModel(models=[str(p) for p in paths])
        self._window: deque[NDArray[np.int16]] = deque(maxlen=WINDOW_FRAMES)
        self._stride = int(stride_frames)
        self._since = 0
        self._last = dict.fromkeys(self._wake_words, 0.0)

        model_paths = {p.stem: p for p in paths}
        for getter, name in (
            (getattr(lk_model, "get_mel_model_path", None), "melspectrogram"),
            (getattr(lk_model, "get_embedding_model_path", None), "embedding_model"),
        ):
            if callable(getter):
                with contextlib.suppress(OSError, TypeError, ValueError):
                    model_paths[name] = Path(getter())
        self._info = EngineInfo(
            engine_id="livekit",
            version=_dist_version("livekit-wakeword"),
            wake_words=self._wake_words,
            model_hashes=hash_models(model_paths),
            config={"models": [str(p) for p in paths], "stride_frames": self._stride},
            capabilities=Capabilities(
                continuous_scores=True, custom_words="training", languages=("en",)
            ),
        )

    @property
    def info(self) -> EngineInfo:
        """Identity and capabilities."""
        return self._info

    def reset(self) -> None:
        """Clear the sliding window and held scores."""
        self._window.clear()
        self._since = 0
        self._last = dict.fromkeys(self._wake_words, 0.0)

    def process(self, frame: NDArray[np.int16]) -> Mapping[str, float]:
        """Append the frame; score the last 2 s window every ``stride_frames`` frames."""
        if frame.shape[0] != FRAME_SAMPLES:
            msg = f"expected {FRAME_SAMPLES} samples, got {frame.shape[0]}"
            raise ValueError(msg)
        self._window.append(np.asarray(frame, dtype=np.int16))
        if len(self._window) < WINDOW_FRAMES:
            return dict(self._last)
        self._since += 1
        if self._since >= self._stride:
            self._since = 0
            chunk = np.concatenate(list(self._window))
            raw: dict[str, float] = self._model.predict(chunk)
            self._last = {w: min(1.0, max(0.0, float(raw.get(w, 0.0)))) for w in self._wake_words}
        return dict(self._last)

    def close(self) -> None:
        """Nothing to release; kept for protocol symmetry."""
        return None


def _dist_version(name: str) -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


@engine_registry.register("livekit")
def create(**config: Any) -> Engine:
    """Factory used by the registry."""
    return LiveKitEngine(**config)
