"""Phonetic near-miss search using espeak-ng grapheme-to-phoneme conversion.

Used to build the confusable-word negative set for each candidate wake word (for
example *michaela*, *michel* and *mikael* for *michael*).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import unicodedata
from collections.abc import Iterable
from pathlib import Path

from wakewordworld.sources.spec import Language
from wakewordworld.util.paths import DataRoot

__all__ = [
    "EspeakMissingError",
    "espeak_binary",
    "near_misses",
    "phoneme_distance",
    "phonemes",
]

_VOICES: dict[Language, str] = {
    Language.EN: "en-us",
    Language.DE: "de",
    Language.FR: "fr-fr",
    Language.ES: "es",
}
# Stress, length, tie and joiner marks that do not count as phonemes.
_IGNORED = frozenset(
    {"\u02c8", "\u02cc", "\u02d0", "\u02d1", "\u203f", "\u0361", "\u200d", "\u200c", " ", "_"}
)
_CACHE: dict[tuple[str, str], str] = {}


class EspeakMissingError(RuntimeError):
    """Raised when espeak-ng is not installed."""


def espeak_binary() -> str:
    """Path of the espeak-ng executable."""
    for candidate in ("/opt/homebrew/bin/espeak-ng", shutil.which("espeak-ng") or ""):
        if candidate and Path(candidate).exists():
            return candidate
    msg = "espeak-ng not found; install it (brew install espeak-ng / apt install espeak-ng)"
    raise EspeakMissingError(msg)


def _disk_cache_path(data_root: DataRoot, language: Language) -> Path:
    return data_root.cache / f"phonemes_{language.value}.json"


def _load_disk_cache(data_root: DataRoot, language: Language) -> None:
    p = _disk_cache_path(data_root, language)
    if p.exists():
        for k, v in json.loads(p.read_text(encoding="utf-8")).items():
            _CACHE.setdefault((language.value, k), v)


def _save_disk_cache(data_root: DataRoot, language: Language) -> None:
    p = _disk_cache_path(data_root, language)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = {k[1]: v for k, v in _CACHE.items() if k[0] == language.value}
    p.write_text(json.dumps(data, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def _espeak_batch(words: list[str], language: Language) -> list[str]:
    binary = espeak_binary()
    proc = subprocess.run(
        [binary, "-q", "--ipa=3", "-v", _VOICES[language]],
        input="\n".join(words) + "\n",
        capture_output=True,
        text=True,
        check=True,
    )
    lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    if len(lines) != len(words):
        # Fall back to one call per word when espeak merges or drops lines.
        return [_espeak_batch([w], language)[0] if len(words) > 1 else "" for w in words]
    return lines


def phonemes(word: str, language: Language, *, data_root: DataRoot | None = None) -> str:
    """IPA transcription of ``word`` as produced by espeak-ng."""
    key = (language.value, word)
    if key not in _CACHE and data_root is not None:
        _load_disk_cache(data_root, language)
    if key not in _CACHE:
        _CACHE[key] = _espeak_batch([word], language)[0]
        if data_root is not None:
            _save_disk_cache(data_root, language)
    return _CACHE[key]


def _symbols(ipa: str) -> list[str]:
    """Split an IPA string into comparable symbols (one base character each).

    espeak-ng's ``--ipa`` output does not reliably separate phonemes, so the
    comparison works on base characters: stress, length, tie and joiner marks and
    combining diacritics (nasalisation, syllabicity) are dropped. Diphthongs thus
    count as two symbols, which is acceptable for a near-miss ranking.
    """
    return [
        ch
        for ch in unicodedata.normalize("NFD", ipa)
        if ch not in _IGNORED and not unicodedata.combining(ch)
    ]


def phoneme_distance(a: str, b: str) -> float:
    """Normalised Levenshtein distance between two IPA strings (0 = identical)."""
    xs, ys = _symbols(a), _symbols(b)
    if not xs and not ys:
        return 0.0
    prev = list(range(len(ys) + 1))
    for i, x in enumerate(xs, start=1):
        cur = [i]
        for j, y in enumerate(ys, start=1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1] / max(len(xs), len(ys))


def near_misses(
    target: str,
    vocabulary: Iterable[str],
    language: Language,
    *,
    max_distance: float = 0.34,
    top: int = 50,
    data_root: DataRoot | None = None,
) -> list[tuple[str, float]]:
    """Vocabulary words phonetically close to ``target`` (excluding the target itself)."""
    words = sorted({w for w in vocabulary if w and w != target})
    if data_root is not None:
        _load_disk_cache(data_root, language)
    missing = [w for w in words if (language.value, w) not in _CACHE]
    for i in range(0, len(missing), 500):
        batch = missing[i : i + 500]
        for w, ipa in zip(batch, _espeak_batch(batch, language), strict=True):
            _CACHE[(language.value, w)] = ipa
    if data_root is not None and missing:
        _save_disk_cache(data_root, language)
    t_ipa = phonemes(target, language, data_root=data_root)
    scored = [(w, phoneme_distance(t_ipa, _CACHE[(language.value, w)])) for w in words]
    hits = [(w, d) for w, d in scored if d <= max_distance]
    hits.sort(key=lambda x: (x[1], x[0]))
    return hits[:top]
