"""First-name lexicons per language.

The seed lists live in ``wakewordworld/resources/names/<lang>.txt`` (one lower-case
name per line, ``#`` comments allowed). A user may extend a language with
``<data_root>/cache/names_extra_<lang>.txt``.
"""

from __future__ import annotations

from functools import cache
from importlib import resources
from pathlib import Path

from wakewordworld.sources.spec import Language
from wakewordworld.util.paths import DataRoot

__all__ = ["NameLexicon", "load_names"]


def _parse(text: str) -> frozenset[str]:
    names: set[str] = set()
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip().casefold()
        if line:
            names.add(line)
    return frozenset(names)


@cache
def load_names(language: Language) -> frozenset[str]:
    """Load the shipped first-name list for a language."""
    ref = resources.files("wakewordworld.resources").joinpath("names", f"{language.value}.txt")
    return _parse(ref.read_text(encoding="utf-8"))


class NameLexicon:
    """Membership test for first names, with optional per-data-root extensions."""

    def __init__(self, data_root: DataRoot | None = None) -> None:
        self._extra: dict[Language, frozenset[str]] = {}
        if data_root is not None:
            for lang in Language:
                p: Path = data_root.cache / f"names_extra_{lang.value}.txt"
                if p.exists():
                    self._extra[lang] = _parse(p.read_text(encoding="utf-8"))

    def names(self, language: Language) -> frozenset[str]:
        """All names known for a language (shipped plus extensions)."""
        return load_names(language) | self._extra.get(language, frozenset())

    def is_name(self, token: str, language: Language) -> bool:
        """Whether a normalised token is a known first name in that language."""
        return token in self.names(language)
