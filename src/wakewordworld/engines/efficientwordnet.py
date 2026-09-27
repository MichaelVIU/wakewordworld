"""EfficientWord-Net adapter (https://github.com/Ant-Brain/EfficientWord-Net).

EfficientWord-Net is a few-shot engine: a ResNet-50 ArcFace embedding is compared
against reference embeddings enrolled from three or four recordings of the word. It
scores a 1.5 s window (24 000 samples) and the reference stream slides it by 0.75 s.
The adapter keeps a ring buffer of the last 1.5 s, calls ``scoreFrame`` every
``stride_frames`` frames (default 9 = 0.72 s, the closest multiple of 80 ms to the
library's 0.75 s hop) and holds the latest confidence on the frames in between.
``scoreFrame`` returns ``None`` when the library's silence gate rejects the window;
that is reported as ``0.0``.

The enrolment reference (``reference_file``) must be built by the maintainer from
recordings that are not part of the benchmark (see the engine's DISCLOSURE).
"""

from __future__ import annotations

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

__all__ = ["EfficientWordNetEngine", "create"]

_INSTALL_HINT = "install with: uv sync --extra engines-ewn  (package: EfficientWord-Net)"
WINDOW_SAMPLES = 24_000  # 1.5 s at 16 kHz, the library's window_length_secs
DEFAULT_STRIDE_FRAMES = 9  # 0.72 s, closest to the library's 0.75 s sliding window


class EfficientWordNetEngine:
    """Streaming adapter around ``eff_word_net.engine.HotwordDetector``."""

    def __init__(
        self,
        hotword: str | None = None,
        reference_file: str | None = None,
        model: str = "resnet_50_arc",
        stride_frames: int = DEFAULT_STRIDE_FRAMES,
        sample_format: str = "int16",
    ) -> None:
        try:
            from eff_word_net import audio_processing, engine
        except ImportError as exc:  # pragma: no cover - exercised via fake module tests
            raise EngineUnavailableError(f"eff_word_net not importable; {_INSTALL_HINT}") from exc
        if not hotword or not reference_file:
            msg = "efficientwordnet engine needs `hotword` and an enrolled `reference_file`"
            raise ValueError(msg)
        if stride_frames < 1:
            msg = "stride_frames must be >= 1"
            raise ValueError(msg)
        if sample_format not in {"int16", "float32"}:
            msg = "sample_format must be 'int16' or 'float32'"
            raise ValueError(msg)
        ref = Path(reference_file)
        if not ref.exists():
            msg = f"reference file not found: {ref}"
            raise FileNotFoundError(msg)
        factories = {
            "resnet_50_arc": getattr(audio_processing, "Resnet50_Arc_loss", None),
            "first_iteration_siamese": getattr(audio_processing, "First_Iteration_Siamese", None),
        }
        factory = factories.get(model)
        if factory is None:
            msg = f"unknown EfficientWord-Net model {model!r}; known: {sorted(factories)}"
            raise ValueError(msg)
        self._base_model = factory()
        self._detector = engine.HotwordDetector(
            hotword=hotword,
            model=self._base_model,
            reference_file=str(ref),
            threshold=0.0,  # never used: the harness thresholds the confidence itself
        )
        self._key = hotword.strip().lower().replace(" ", "_")
        self._stride = int(stride_frames)
        self._format = sample_format
        self._window: deque[NDArray[np.int16]] = deque(maxlen=WINDOW_SAMPLES // FRAME_SAMPLES + 1)
        self._since = 0
        self._last = 0.0
        self._info = EngineInfo(
            engine_id="efficientwordnet",
            version=_dist_version("EfficientWord-Net"),
            wake_words=(self._key,),
            model_hashes=hash_models({"reference": ref}),
            config={
                "hotword": hotword,
                "reference_file": str(ref),
                "model": model,
                "stride_frames": self._stride,
                "sample_format": sample_format,
            },
            capabilities=Capabilities(
                continuous_scores=True, custom_words="enrolment", languages=()
            ),
        )

    @property
    def info(self) -> EngineInfo:
        """Identity and capabilities."""
        return self._info

    def reset(self) -> None:
        """Clear the ring buffer and held score."""
        self._window.clear()
        self._since = 0
        self._last = 0.0

    def process(self, frame: NDArray[np.int16]) -> Mapping[str, float]:
        """Append the frame; score the last 1.5 s every ``stride_frames`` frames."""
        if frame.shape[0] != FRAME_SAMPLES:
            msg = f"expected {FRAME_SAMPLES} samples, got {frame.shape[0]}"
            raise ValueError(msg)
        self._window.append(np.asarray(frame, dtype=np.int16))
        buffered = len(self._window) * FRAME_SAMPLES
        if buffered < WINDOW_SAMPLES:
            return {self._key: self._last}
        self._since += 1
        if self._since >= self._stride:
            self._since = 0
            audio = np.concatenate(list(self._window))[-WINDOW_SAMPLES:]
            inp: NDArray[Any] = (
                audio.astype(np.float32) / 32768.0 if self._format == "float32" else audio
            )
            result = self._detector.scoreFrame(inp)
            if result is None:
                self._last = 0.0
            else:
                self._last = min(1.0, max(0.0, float(result.get("confidence", 0.0))))
        return {self._key: self._last}

    def close(self) -> None:
        """Nothing to release; kept for protocol symmetry."""
        return None


def _dist_version(name: str) -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


@engine_registry.register("efficientwordnet")
def create(**config: Any) -> Engine:
    """Factory used by the registry."""
    return EfficientWordNetEngine(**config)
