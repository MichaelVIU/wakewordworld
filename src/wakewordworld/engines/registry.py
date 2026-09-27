"""Import all adapter modules so they register, and expose lookup helpers.

Importing this module does not import any engine SDK; adapters import their SDK lazily
when instantiated and raise :class:`EngineUnavailableError` when it is missing.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from wakewordworld.engines import microwakeword, openwakeword, vosk  # noqa: F401 - registration
from wakewordworld.engines.base import Engine, engine_registry

__all__ = ["list_engines", "load_engine"]


def load_engine(engine_id: str, **config: Any) -> Engine:
    """Instantiate a registered engine with a configuration mapping."""
    return engine_registry.create(engine_id, **config)


def list_engines() -> Sequence[str]:
    """Ids of all registered engines."""
    return engine_registry.ids()
