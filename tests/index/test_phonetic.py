# ruff: noqa: RUF001 - IPA symbols and apostrophe variants are the point of these tests
from __future__ import annotations

import pytest

from wakewordworld.index.phonetic import (
    EspeakMissingError,
    espeak_binary,
    near_misses,
    phoneme_distance,
    phonemes,
)
from wakewordworld.sources.spec import Language
from wakewordworld.util.paths import DataRoot

try:
    espeak_binary()
    HAVE_ESPEAK = True
except EspeakMissingError:
    HAVE_ESPEAK = False

needs_espeak = pytest.mark.skipif(not HAVE_ESPEAK, reason="espeak-ng not installed")


def test_phoneme_distance_pure() -> None:
    assert phoneme_distance("maɪkəl", "maɪkəl") == 0.0
    assert phoneme_distance("mˈa\u200dɪkəl", "maɪːkəl") == 0.0  # stress/length/joiner ignored
    assert phoneme_distance("bɔ̃ʒuʁ", "bɔʒuʁ") == 0.0  # combining nasal mark ignored
    assert 0 < phoneme_distance("maɪkəl", "mɪkeɪlə") < 1
    assert phoneme_distance("", "") == 0.0
    assert phoneme_distance("a", "") == 1.0


@needs_espeak
def test_phonemes_and_cache(tmp_path) -> None:
    root = DataRoot(tmp_path)
    root.ensure()
    ipa = phonemes("michael", Language.EN, data_root=root)
    assert ipa
    assert "m" in ipa
    assert (root.cache / "phonemes_en.json").exists()
    assert phonemes("michael", Language.EN, data_root=root) == ipa


@needs_espeak
@pytest.mark.parametrize(
    ("lang", "target", "close", "far"),
    [
        (Language.EN, "michael", ["mikael", "mike"], ["banana", "computer"]),
        (Language.DE, "michael", ["michaela", "michel"], ["kartoffel", "haus"]),
        (Language.FR, "michel", ["michèle", "mickaël"], ["bonjour", "voiture"]),
    ],
)
def test_near_misses(lang: Language, target: str, close: list[str], far: list[str]) -> None:
    vocab = [target, *close, *far, "xyzzy"]
    hits = near_misses(target, vocab, lang, max_distance=0.5)
    words = [w for w, _ in hits]
    assert target not in words
    for c in close:
        assert c in words, (c, hits)
    for f in far:
        assert f not in words, (f, hits)
    distances = [d for _, d in hits]
    assert distances == sorted(distances)
