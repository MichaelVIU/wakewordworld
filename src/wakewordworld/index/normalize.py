"""Token normalisation for the word index.

The word index stores a normalised form of every token so that lookups such as
"all occurrences of *michael*" are exact string matches. Normalisation is
deliberately conservative: case folding, Unicode NFC, and stripping of
surrounding punctuation. Diacritics are kept (``é`` stays ``é``), German ``ß`` is
not folded, and internal apostrophes and hyphens are preserved so that
``d'accord`` or ``jean-pierre`` stay one token.

French and Spanish elisions (``l'école``, ``d'Anne``) are kept as one token by
:func:`normalise_token`; :func:`split_elision` additionally exposes the part after
the apostrophe so the index can carry both forms.
"""

from __future__ import annotations

import re
import unicodedata

from wakewordworld.sources.spec import Language

__all__ = ["normalise_text", "normalise_token", "split_elision"]

# Characters treated as apostrophes in source text.
_APOSTROPHES = "'\u2019\u02bc\u2018`\u00b4"
_APOS_RE = re.compile(f"[{re.escape(_APOSTROPHES)}]")
# Anything that is neither a letter, a digit, an apostrophe nor a hyphen.
_EDGE_STRIP_RE = re.compile(r"^[^\w'-]+|[^\w'-]+$", re.UNICODE)
_HAS_LETTER_OR_DIGIT_RE = re.compile(r"[^\W_]", re.UNICODE)
_SPLIT_RE = re.compile(r"\s+")

# Elision prefixes per language (lower-case, without apostrophe).
_ELISION_PREFIXES: dict[Language, frozenset[str]] = {
    Language.FR: frozenset(
        {"l", "d", "j", "m", "n", "s", "t", "c", "qu", "jusqu", "lorsqu", "puisqu", "quoiqu"}
    ),
    Language.ES: frozenset({"l", "d"}),  # rare, mostly Catalan/Italian loans
    Language.EN: frozenset(),
    Language.DE: frozenset(),
}


def normalise_token(text: str, language: Language) -> str | None:
    """Normalise one raw token; return ``None`` when nothing usable remains.

    Args:
        text: Raw token as found in a transcript.
        language: Language of the transcript (drives elision rules only).

    Returns:
        Lower-case NFC token with surrounding punctuation stripped, or ``None`` if
        the token contains no letter or digit.
    """
    del language  # normalisation itself is language independent for now
    token = unicodedata.normalize("NFC", text).strip()
    token = _APOS_RE.sub("'", token)
    token = _EDGE_STRIP_RE.sub("", token)
    token = token.strip("'-_")
    if not token or not _HAS_LETTER_OR_DIGIT_RE.search(token):
        return None
    return token.casefold() if "ß" not in token else token.lower()


def split_elision(token: str, language: Language) -> list[str]:
    """Return additional index forms for an elided token.

    ``l'école`` yields ``["école"]``; ``jusqu'au`` yields ``["au"]``; a token without
    a recognised elision prefix yields ``[]``.
    """
    prefixes = _ELISION_PREFIXES.get(language, frozenset())
    if not prefixes or "'" not in token:
        return []
    head, _, tail = token.partition("'")
    if head in prefixes and tail and _HAS_LETTER_OR_DIGIT_RE.search(tail):
        return [tail]
    return []


def normalise_text(text: str, language: Language) -> list[str]:
    """Split free text on whitespace and normalise every token (elisions kept whole)."""
    out: list[str] = []
    for raw in _SPLIT_RE.split(text):
        tok = normalise_token(raw, language)
        if tok is not None:
            out.append(tok)
    return out
