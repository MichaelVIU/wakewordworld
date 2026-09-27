from __future__ import annotations

from pathlib import Path

import numpy as np

from wakewordworld.transcribe.music import (
    is_music_marker,
    looks_like_music,
    music_features,
    tag_music,
)
from wakewordworld.transcribe.schema import Segment

from .conftest import SR, speechlike, tone, write_flac


def test_markers() -> None:
    assert is_music_marker("")
    assert is_music_marker(" [Musik] ")
    assert is_music_marker("♪♪")
    assert is_music_marker("[music]")
    assert not is_music_marker("Musik ist schön")


def test_tone_is_music_speechlike_is_not() -> None:
    t = music_features(tone(3.0), SR)
    s = music_features(speechlike(3.0), SR)
    assert looks_like_music(t)
    assert not looks_like_music(s)
    assert t.zcr_cv < s.zcr_cv


def test_tag_music_on_file(tmp_path: Path) -> None:
    audio = np.concatenate([speechlike(3.0), tone(3.0, 660.0), speechlike(2.0, seed=3)])
    path = write_flac(tmp_path / "a.flac", audio)
    segs = [
        Segment(start=0.0, end=3.0, text="hello there Jane"),
        Segment(start=3.0, end=6.0, text="la la la"),
        Segment(start=6.0, end=8.0, text="back to talking"),
        Segment(start=7.0, end=7.3, text="[Musik]"),
        Segment(start=7.0, end=7.3, text="tiny"),
    ]
    tag_music(path, segs)
    assert [s.is_music for s in segs] == [False, True, False, True, False]
