from __future__ import annotations

import sys
import types
from collections.abc import Iterator

import numpy as np
import pytest

from wakewordworld.engines.base import FRAME_SAMPLES


@pytest.fixture
def frame() -> np.ndarray:
    return np.zeros(FRAME_SAMPLES, dtype=np.int16)


@pytest.fixture
def fake_modules() -> Iterator[dict[str, types.ModuleType]]:
    """Inject fake SDK modules into sys.modules for the duration of a test."""
    injected: dict[str, types.ModuleType] = {}
    saved = {}

    def install(name: str, module: types.ModuleType) -> None:
        saved[name] = sys.modules.get(name)
        sys.modules[name] = module
        injected[name] = module

    injected["__install__"] = install  # type: ignore[assignment]
    yield injected
    for name, old in saved.items():
        if old is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = old
