"""The engine adapter contract.

Every engine is driven the same way: 16 kHz mono int16 audio is fed in fixed 80 ms
frames (1280 samples) in file order with no lookahead. After each frame the adapter
returns one score per wake word in ``[0, 1]``. Engines that only expose a boolean
detection return ``1.0`` on the frame where they fire and ``0.0`` otherwise; their
curves are obtained by sweeping the engine's sensitivity parameter instead of a score
threshold (``Capabilities.sweep_param``).

Detection semantics (rising edge, debounce, hit window) are *not* implemented by
adapters. They live in :mod:`wakewordworld.eval.detect` and are applied identically to
all engines.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "FRAME_SAMPLES",
    "SAMPLE_RATE",
    "Capabilities",
    "Engine",
    "EngineInfo",
    "EngineRegistry",
    "EngineUnavailableError",
    "FrameResult",
    "engine_registry",
]

SAMPLE_RATE = 16_000
FRAME_MS = 80
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000  # 1280


class EngineUnavailableError(RuntimeError):
    """Raised when an engine's runtime or model is not installed or not licensed."""


@dataclass(frozen=True)
class Capabilities:
    """What an engine can do, so the harness can pick the right evaluation path."""

    continuous_scores: bool
    """True when scores are graded per frame; False for boolean-only engines."""

    sweep_param: str | None = None
    """Name of a construction-time sensitivity parameter to sweep for boolean engines."""

    sweep_values: tuple[float, ...] = ()
    """Values of ``sweep_param`` to evaluate (boolean engines only)."""

    custom_words: str = "none"
    """How custom wake words are made: none | text | training | enrolment."""

    languages: tuple[str, ...] = ()
    """Languages the loaded models were built for (ISO 639-1), informational."""

    on_device: bool = True
    """False for cloud/API engines; such results are flagged in reports."""


@dataclass(frozen=True)
class EngineInfo:
    """Identity of a loaded engine for provenance in results."""

    engine_id: str
    version: str
    wake_words: tuple[str, ...]
    model_hashes: Mapping[str, str] = field(default_factory=dict)
    config: Mapping[str, object] = field(default_factory=dict)
    capabilities: Capabilities = field(default_factory=lambda: Capabilities(continuous_scores=True))


FrameResult = Mapping[str, float]
"""Scores after one frame, keyed by wake word (as given in ``EngineInfo.wake_words``)."""


@runtime_checkable
class Engine(Protocol):
    """Streaming wake word engine."""

    @property
    def info(self) -> EngineInfo:
        """Identity and capabilities."""
        ...

    def reset(self) -> None:
        """Clear internal state between independent audio streams (chunks)."""
        ...

    def process(self, frame: NDArray[np.int16]) -> FrameResult:
        """Consume exactly ``FRAME_SAMPLES`` int16 samples and return scores.

        Adapters that need other frame sizes buffer internally; the harness never sends
        a partial frame except that the last frame of a stream is zero-padded.
        """
        ...

    def close(self) -> None:
        """Release resources."""
        ...


EngineFactory = Callable[..., Engine]


class EngineRegistry:
    """Maps engine ids to factories; adapters register at import time."""

    def __init__(self) -> None:
        self._factories: dict[str, EngineFactory] = {}

    def register(self, engine_id: str) -> Callable[[EngineFactory], EngineFactory]:
        """Decorator registering an engine factory."""

        def deco(factory: EngineFactory) -> EngineFactory:
            if engine_id in self._factories:
                msg = f"engine {engine_id!r} already registered"
                raise ValueError(msg)
            self._factories[engine_id] = factory
            return factory

        return deco

    def create(self, engine_id: str, **config: object) -> Engine:
        """Instantiate an engine by id."""
        try:
            factory = self._factories[engine_id]
        except KeyError as exc:
            msg = f"unknown engine {engine_id!r}; known: {', '.join(self.ids()) or 'none'}"
            raise LookupError(msg) from exc
        return factory(**config)

    def ids(self) -> Sequence[str]:
        """Registered engine ids."""
        return sorted(self._factories)


engine_registry = EngineRegistry()


def pad_frame(frame: NDArray[np.int16]) -> NDArray[np.int16]:
    """Zero-pad a short final frame to ``FRAME_SAMPLES``."""
    if frame.shape[0] == FRAME_SAMPLES:
        return frame
    out = np.zeros(FRAME_SAMPLES, dtype=np.int16)
    out[: frame.shape[0]] = frame
    return out
