"""openWakeWord adapter (https://github.com/dscripka/openWakeWord).

openWakeWord scores exactly one 80 ms frame (1280 samples) per call, which matches the
harness frame, so frames are passed through unchanged. Scores are the model's raw
per-frame outputs; openWakeWord's optional ``patience``/``debounce_time`` arguments are
never used because the harness applies its own detection semantics.

Built-in models are downloaded by openWakeWord into its own package directory (the
library does not support another target for ``Model``), so the container build
pre-downloads them; custom models are given as paths.
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

__all__ = ["OpenWakeWordEngine", "create"]

_INSTALL_HINT = "install with: uv sync --extra engines-oww  (package: openwakeword)"


def _wake_word_name(model: str) -> str:
    """Requested model string -> wake word key ('alexa', 'hey_jarvis', custom file stem)."""
    return Path(model).stem if model.endswith((".onnx", ".tflite")) else model


class OpenWakeWordEngine:
    """Streaming adapter around ``openwakeword.model.Model``."""

    def __init__(
        self,
        models: list[str] | None = None,
        inference_framework: str = "onnx",
        vad_threshold: float = 0.0,
        enable_speex_noise_suppression: bool = False,
    ) -> None:
        try:
            import openwakeword
            from openwakeword.model import Model
            from openwakeword.utils import download_models
        except ImportError as exc:  # pragma: no cover - exercised via fake module tests
            raise EngineUnavailableError(f"openwakeword not importable; {_INSTALL_HINT}") from exc

        models = list(models or ["alexa", "hey_jarvis", "hey_mycroft"])
        builtin = [m for m in models if not m.endswith((".onnx", ".tflite"))]
        if builtin:
            download_models(model_names=builtin)
        # openWakeWord rewrites the list it is given in place (names -> file paths), so
        # derive the wake word names first and hand it a copy.
        self._requested = tuple(_wake_word_name(m) for m in models)
        self._model = Model(
            wakeword_models=list(models),
            inference_framework=inference_framework,
            vad_threshold=vad_threshold,
            enable_speex_noise_suppression=enable_speex_noise_suppression,
        )
        self._key_map = self._build_key_map(list(self._model.models.keys()))
        version = str(getattr(openwakeword, "__version__", None) or _dist_version("openwakeword"))
        model_paths: dict[str, Path] = {}
        for m in models:
            p = Path(m)
            if p.suffix in {".onnx", ".tflite"} and p.exists():
                model_paths[_wake_word_name(m)] = p
        # Built-in models live inside the package; hash them too when locatable.
        pkg_models = Path(openwakeword.__file__).parent / "resources" / "models"
        for name in builtin:
            for cand in pkg_models.glob(f"{name}*.{inference_framework}"):
                model_paths.setdefault(name, cand)
        self._info = EngineInfo(
            engine_id="openwakeword",
            version=version,
            wake_words=self._requested,
            model_hashes=hash_models(model_paths),
            config={
                "models": models,
                "inference_framework": inference_framework,
                "vad_threshold": vad_threshold,
                "enable_speex_noise_suppression": enable_speex_noise_suppression,
            },
            capabilities=Capabilities(
                continuous_scores=True, custom_words="training", languages=("en",)
            ),
        )

    def _build_key_map(self, reported: list[str]) -> dict[str, str]:
        """Map openWakeWord's reported model keys onto the requested wake word names."""
        mapping: dict[str, str] = {}
        for key in reported:
            if key in self._requested:
                mapping[key] = key
                continue
            matches = [w for w in self._requested if key.startswith(w)]
            if matches:
                mapping[key] = max(matches, key=len)
        return mapping

    @property
    def info(self) -> EngineInfo:
        """Identity and capabilities."""
        return self._info

    def reset(self) -> None:
        """Clear openWakeWord's internal audio and prediction buffers."""
        self._model.reset()

    def process(self, frame: NDArray[np.int16]) -> Mapping[str, float]:
        """Score one 80 ms frame."""
        if frame.shape[0] != FRAME_SAMPLES:
            msg = f"expected {FRAME_SAMPLES} samples, got {frame.shape[0]}"
            raise ValueError(msg)
        raw: dict[str, Any] = self._model.predict(frame)
        out = dict.fromkeys(self._requested, 0.0)
        for key, score in raw.items():
            name = self._key_map.get(key)
            if name is not None:
                out[name] = max(out[name], min(1.0, max(0.0, float(score))))
        return out

    def close(self) -> None:
        """Nothing to release; kept for protocol symmetry."""
        return None


def _dist_version(name: str) -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


@engine_registry.register("openwakeword")
def create(**config: Any) -> Engine:
    """Factory used by the registry."""
    return OpenWakeWordEngine(**config)
