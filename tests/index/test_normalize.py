# ruff: noqa: RUF001 - IPA symbols and apostrophe variants are the point of these tests
from __future__ import annotations

import pytest

from wakewordworld.index.normalize import normalise_text, normalise_token, split_elision
from wakewordworld.sources.spec import Language


@pytest.mark.parametrize(
    ("raw", "lang", "expected"),
    [
        ("Jane,", Language.EN, "jane"),
        ("Morgan?", Language.EN, "morgan"),
        ('"Michael"', Language.EN, "michael"),
        ("(Liz)", Language.EN, "liz"),
        ("Straße", Language.DE, "straße"),
        ("GROSS", Language.DE, "gross"),
        ("Müller", Language.DE, "müller"),
        ("¿Qué?", Language.ES, "qué"),
        ("¡Hola!", Language.ES, "hola"),
        ("l'école", Language.FR, "l'école"),
        ("d’Anne", Language.FR, "d'anne"),
        ("jean-pierre", Language.FR, "jean-pierre"),
        ("don't", Language.EN, "don't"),
        ("...", Language.EN, None),
        ("--", Language.EN, None),
        ("", Language.EN, None),
        ("42", Language.EN, "42"),
        ("'tis", Language.EN, "tis"),
    ],
)
def test_normalise_token(raw: str, lang: Language, expected: str | None) -> None:
    assert normalise_token(raw, lang) == expected


def test_split_elision() -> None:
    assert split_elision("l'école", Language.FR) == ["école"]
    assert split_elision("d'anne", Language.FR) == ["anne"]
    assert split_elision("jusqu'au", Language.FR) == ["au"]
    assert split_elision("aujourd'hui", Language.FR) == []
    assert split_elision("don't", Language.EN) == []
    assert split_elision("école", Language.FR) == []


def test_normalise_text() -> None:
    assert normalise_text("Yes, Michael — I do!", Language.EN) == ["yes", "michael", "i", "do"]
    assert normalise_text("  ", Language.DE) == []
