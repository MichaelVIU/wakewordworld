"""microWakeWord adapter via ``pymicro-wakeword`` (https://github.com/OHF-Voice/pymicro-wakeword).

The library computes 40-dim spectrogram features every 10 ms with a streaming front end
and runs the model every ``stride`` (typically 3) feature windows. ``process_streaming_prob``
returns the *sliding-window mean* over the last ``sliding_window_size`` (typically 5)
model outputs, and ``process_streaming`` compares that mean with the model's
``probability_cutoff``. Those are the semantics shipped in ESPHome.

For each 80 ms harness frame the adapter feeds the audio to the feature front end,
runs the model on every window it yields, and returns the **maximum** windowed
probability observed within the frame. Taking the maximum keeps rising edges visible
at the coarser 80 ms resolution; a mean would smear short peaks below threshold.

Config:
    ``models``: built-in ids (okay_nabu, hey_jarvis, hey_mycroft, alexa) or paths to
    model ``.json`` manifests.
    ``score``: ``"windowed"`` (default; the library's sliding mean) or ``"raw"`` (the
    latest single model output, read from the library's probability buffer).
    ``native_detection``: when True, return 1.0/0.0 from the library's own
    ``process_streaming`` (mean > cutoff) instead of a graded score, to measure the
    engine exactly as shipped.
"""

from __future__ import annotations

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

__all__ = ["MicroWakeWordEngine", "create"]

_INSTALL_HINT = "install with: uv sync --extra engines-mww  (package: pymicro-wakeword)"


class MicroWakeWordEngine:
    """Streaming adapter around ``pymicro_wakeword.MicroWakeWord``."""

    def __init__(
        self,
        models: list[str] | None = None,
        score: str = "windowed",
        native_detection: bool = False,
    ) -> None:
        try:
            import pymicro_wakeword as pw
            from pymicro_wakeword import MicroWakeWord, MicroWakeWordFeatures
        except ImportError as exc:  # pragma: no cover - exercised via fake module tests
            raise EngineUnavailableError(
                f"pymicro_wakeword not importable; {_INSTALL_HINT}"
            ) from exc
        if score not in {"windowed", "raw"}:
            msg = f"score must be 'windowed' or 'raw', got {score!r}"
            raise ValueError(msg)

        models = list(models or ["okay_nabu", "hey_jarvis", "hey_mycroft", "alexa"])
        self._features = MicroWakeWordFeatures()
        self._models: dict[str, Any] = {}
        model_paths: dict[str, Path] = {}
        languages: set[str] = set()
        for m in models:
            if m.endswith(".json"):
                mww = MicroWakeWord.from_config(m)
                name = Path(m).stem
                model_paths[name] = Path(m)
            else:
                mww = MicroWakeWord.from_builtin(pw.Model(m))
                name = m
                mp = getattr(mww, "model_path", None)
                if isinstance(mp, bytes):
                    mp = mp.decode()
                if mp:
                    model_paths[name] = Path(mp)
            self._models[name] = mww
            languages.update(getattr(mww, "trained_languages", []) or [])
        self._score_mode = score
        self._native = native_detection
        version = str(getattr(pw, "__version__", None) or _dist_version("pymicro-wakeword"))
        self._info = EngineInfo(
            engine_id="microwakeword",
            version=version,
            wake_words=tuple(self._models),
            model_hashes=hash_models(model_paths),
            config={
                "models": models,
                "score": score,
                "native_detection": native_detection,
                "cutoffs": {
                    n: getattr(m, "probability_cutoff", None) for n, m in self._models.items()
                },
                "sliding_window_size": {
                    n: getattr(m, "sliding_window_size", None) for n, m in self._models.items()
                },
            },
            capabilities=Capabilities(
                continuous_scores=not native_detection,
                custom_words="training",
                languages=tuple(sorted(languages)) or ("en",),
            ),
        )

    @property
    def info(self) -> EngineInfo:
        """Identity and capabilities."""
        return self._info

    def reset(self) -> None:
        """Reset the feature front end and every model's window buffers."""
        self._features.reset()
        for m in self._models.values():
            m.reset()

    def _score_one(self, mww: Any, feats: NDArray[np.floating]) -> float:
        if self._native:
            return 1.0 if mww.process_streaming(feats) else 0.0
        prob = mww.process_streaming_prob(feats)
        if self._score_mode == "raw":
            buf = getattr(mww, "_probabilities", None)
            if buf:
                prob = buf[-1]
        return 0.0 if prob is None else float(prob)

    def process(self, frame: NDArray[np.int16]) -> Mapping[str, float]:
        """Feed one 80 ms frame; return the max windowed probability per model."""
        if frame.shape[0] != FRAME_SAMPLES:
            msg = f"expected {FRAME_SAMPLES} samples, got {frame.shape[0]}"
            raise ValueError(msg)
        out = dict.fromkeys(self._models, 0.0)
        for feats in self._features.process_streaming(frame.astype(np.int16).tobytes()):
            for name, mww in self._models.items():
                s = self._score_one(mww, feats)
                out[name] = max(out[name], min(1.0, max(0.0, s)))
        return out

    def close(self) -> None:
        """Release TFLite interpreters."""
        for m in self._models.values():
            close = getattr(m, "close", None)
            if callable(close):
                close()


def _dist_version(name: str) -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


@engine_registry.register("microwakeword")
def create(**config: Any) -> Engine:
    """Factory used by the registry."""
    return MicroWakeWordEngine(**config)
