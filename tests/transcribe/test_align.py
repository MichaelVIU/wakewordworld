from __future__ import annotations

import builtins
import sys
from pathlib import Path

import pytest

from wakewordworld.transcribe.align import (
    align_reference,
    align_uniform,
    romanise,
    split_long_segments,
    tokenize,
)
from wakewordworld.transcribe.backends.base import BackendUnavailable
from wakewordworld.transcribe.schema import Segment


def test_tokenize_keeps_apostrophes_and_hyphens() -> None:
    assert tokenize("Liz said she'd send them, Mm-hmm.") == [
        "Liz",
        "said",
        "she'd",
        "send",
        "them",
        "Mm-hmm",
    ]
    assert tokenize("l\u2019été à Paris !") == ["l\u2019été", "à", "Paris"]


@pytest.mark.parametrize(
    ("word", "expected"),
    [
        ("Michael", "michael"),
        ("Größe", "grosse"),
        ("Frédéric", "frederic"),
        ("l\u2019été", "l'ete"),
        ("¿Qué?", "que"),
        ("—", ""),
    ],
)
def test_romanise(word: str, expected: str) -> None:
    assert romanise(word) == expected


def test_align_uniform_spreads_words() -> None:
    seg = Segment(start=10.0, end=14.0, text="Jane are we recording", speaker="A")
    out = align_uniform([seg])
    words = out[0].words
    assert [w.text for w in words] == ["Jane", "are", "we", "recording"]
    assert words[0].start == pytest.approx(10.0)
    assert words[-1].end == pytest.approx(14.0)
    assert all(w.confidence is None and w.speaker == "A" for w in words)
    assert align_uniform([Segment(start=0, end=1, text="...")])[0].words == []


def test_split_long_segments_proportional() -> None:
    seg = Segment(
        start=0.0,
        end=120.0,
        text="First sentence here. Second one is longer than the first! Third?",
    )
    parts = split_long_segments([seg], max_s=60)
    assert len(parts) == 3
    assert parts[0].start == 0.0
    assert parts[-1].end == pytest.approx(120.0)
    assert all(p.end - p.start > 0 for p in parts)
    short = Segment(start=0.0, end=10.0, text="short. one.")
    assert split_long_segments([short]) == [short]


def test_align_reference_needs_torch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import wakewordworld.transcribe.align as align_mod

    monkeypatch.setattr(align_mod, "_MMS", None)
    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name in ("torch", "torchaudio") or name.startswith("torchaudio."):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", fake_import)
    sys.modules.pop("torch", None)
    with pytest.raises(BackendUnavailable, match="torch"):
        align_reference(tmp_path / "x.flac", [Segment(start=0, end=1, text="hi")])
