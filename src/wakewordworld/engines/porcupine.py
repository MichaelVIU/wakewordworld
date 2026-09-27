"""Picovoice Porcupine adapter (https://picovoice.ai/platform/porcupine/).

Porcupine is a closed-source engine driven through the ``pvporcupine`` SDK. It consumes
512-sample frames and returns a keyword index (or ``-1``); there is no graded score.
The adapter therefore reports ``1.0`` for a keyword on the 80 ms harness frame during
which any of its 512-sample sub-frames fired, ``0.0`` otherwise, and declares a
``sensitivity`` sweep so the harness can trace a trade-off curve by re-running.

The 1280-sample harness frame is not a multiple of 512: two full sub-frames are
processed per call and the 256-sample remainder is carried into the next call, so no
audio is dropped and alignment drifts by design exactly as it would in production.

An access key is required (``PICOVOICE_ACCESS_KEY`` or ``access_key``); the free tier
is limited to non-commercial use and a small number of custom keywords per month.
"""

from __future__ import annotations

import os
import re
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

__all__ = ["PorcupineEngine", "create"]

_INSTALL_HINT = "install with: uv sync --extra engines-porcupine  (package: pvporcupine)"
_KEY_HINT = (
    "Porcupine needs an access key: set PICOVOICE_ACCESS_KEY or pass access_key "
    "(free keys at https://console.picovoice.ai/, non-commercial use only)"
)
SWEEP_SENSITIVITIES: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
_MODEL_LANG_RE = re.compile(r"porcupine_params_([a-z]{2})\.pv$")


def _wake_word_name(keyword_or_path: str) -> str:
    """Built-in keyword name or the stem of a custom ``.ppn`` file, underscore-joined."""
    name = Path(keyword_or_path).stem if keyword_or_path.endswith(".ppn") else keyword_or_path
    # custom files are usually named like "Hey-Michael_en_mac_v3_0_0.ppn"
    name = re.sub(r"_(?:[a-z]{2})_(?:mac|linux|windows|raspberry-pi|android|ios|web).*$", "", name)
    return re.sub(r"[\s\-]+", "_", name.strip().lower())


class PorcupineEngine:
    """Streaming adapter around ``pvporcupine.Porcupine``."""

    def __init__(
        self,
        keywords: list[str] | None = None,
        keyword_paths: list[str] | None = None,
        sensitivity: float = 0.5,
        access_key: str | None = None,
        model_path: str | None = None,
        library_path: str | None = None,
    ) -> None:
        try:
            import pvporcupine
        except ImportError as exc:  # pragma: no cover - exercised via fake module tests
            raise EngineUnavailableError(f"pvporcupine not importable; {_INSTALL_HINT}") from exc
        key = access_key or os.environ.get("PICOVOICE_ACCESS_KEY", "")
        if not key:
            raise EngineUnavailableError(_KEY_HINT)
        if not keywords and not keyword_paths:
            keywords = ["porcupine"]
        names = [*(keywords or []), *(keyword_paths or [])]
        self._wake_words = tuple(_wake_word_name(n) for n in names)
        sens = [float(sensitivity)] * len(names)
        kwargs: dict[str, Any] = {"access_key": key, "sensitivities": sens}
        if keywords:
            kwargs["keywords"] = list(keywords)
        if keyword_paths:
            kwargs["keyword_paths"] = list(keyword_paths)
        if model_path:
            kwargs["model_path"] = model_path
        if library_path:
            kwargs["library_path"] = library_path
        base_error = getattr(pvporcupine, "PorcupineError", Exception)
        try:
            self._handle = pvporcupine.create(**kwargs)
        except base_error as exc:
            # Activation, key and platform errors are environment problems, not bugs.
            raise EngineUnavailableError(f"Porcupine could not start: {exc}") from exc
        self._frame_length = int(getattr(self._handle, "frame_length", 512))
        self._remainder: NDArray[np.int16] = np.zeros(0, dtype=np.int16)

        model_paths: dict[str, Path] = {}
        builtin_paths = getattr(pvporcupine, "KEYWORD_PATHS", {})
        for k in keywords or []:
            p = builtin_paths.get(k)
            if p:
                model_paths[_wake_word_name(k)] = Path(p)
        for kp in keyword_paths or []:
            model_paths[_wake_word_name(kp)] = Path(kp)
        if model_path:
            model_paths["params"] = Path(model_path)
        lang = "en"
        if model_path:
            m = _MODEL_LANG_RE.search(Path(model_path).name)
            if m:
                lang = m.group(1)
        self._info = EngineInfo(
            engine_id="porcupine",
            version=str(getattr(self._handle, "version", None) or _dist_version("pvporcupine")),
            wake_words=self._wake_words,
            model_hashes=hash_models(model_paths),
            config={
                "keywords": list(keywords or []),
                "keyword_paths": list(keyword_paths or []),
                "sensitivity": float(sensitivity),
                "model_path": model_path,
            },
            capabilities=Capabilities(
                continuous_scores=False,
                sweep_param="sensitivity",
                sweep_values=SWEEP_SENSITIVITIES,
                custom_words="text",
                languages=(lang,),
            ),
        )

    @property
    def info(self) -> EngineInfo:
        """Identity and capabilities."""
        return self._info

    def reset(self) -> None:
        """Drop buffered samples (Porcupine itself keeps no long-term state)."""
        self._remainder = np.zeros(0, dtype=np.int16)

    def process(self, frame: NDArray[np.int16]) -> Mapping[str, float]:
        """Feed one harness frame as 512-sample sub-frames; 1.0 where a keyword fired."""
        if frame.shape[0] != FRAME_SAMPLES:
            msg = f"expected {FRAME_SAMPLES} samples, got {frame.shape[0]}"
            raise ValueError(msg)
        buf = np.concatenate([self._remainder, frame.astype(np.int16)])
        out = dict.fromkeys(self._wake_words, 0.0)
        n_full = buf.shape[0] // self._frame_length
        for i in range(n_full):
            sub = buf[i * self._frame_length : (i + 1) * self._frame_length]
            idx = int(self._handle.process(sub.tolist()))
            if 0 <= idx < len(self._wake_words):
                out[self._wake_words[idx]] = 1.0
        self._remainder = buf[n_full * self._frame_length :].copy()
        return out

    def close(self) -> None:
        """Release the native handle."""
        handle = getattr(self, "_handle", None)
        if handle is not None:
            handle.delete()
            self._handle = None


def _dist_version(name: str) -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


@engine_registry.register("porcupine")
def create(**config: Any) -> Engine:
    """Factory used by the registry."""
    return PorcupineEngine(**config)
