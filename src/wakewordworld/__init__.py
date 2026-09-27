"""WakeWordWorld: an independent benchmark for wake word engines on real multilingual speech."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("wakewordworld")
except PackageNotFoundError:  # pragma: no cover - only when running from a source tree
    __version__ = "0.0.0"

__all__ = ["__version__"]
