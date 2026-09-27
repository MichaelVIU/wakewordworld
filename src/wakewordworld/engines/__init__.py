"""Engine adapters: a uniform streaming interface over wake word engines."""

from wakewordworld.engines.base import (
    FRAME_SAMPLES,
    SAMPLE_RATE,
    Capabilities,
    Engine,
    EngineInfo,
    EngineUnavailableError,
    FrameResult,
    engine_registry,
)

__all__ = [
    "FRAME_SAMPLES",
    "SAMPLE_RATE",
    "Capabilities",
    "Engine",
    "EngineInfo",
    "EngineUnavailableError",
    "FrameResult",
    "engine_registry",
]
