from __future__ import annotations

from pathlib import Path

import pytest

from wakewordworld.transcribe.reference import (
    icsi_mrt,
    parse_reference,
    plain,
    sniff_format,
    webvtt,
)


def test_icsi_mrt_parses_segments_and_strips_children(fixtures_dir: Path) -> None:
    segs = icsi_mrt.parse(fixtures_dir / "sample.mrt")
    texts = [s.text for s in segs]
    assert texts == [
        "Okay , so Jane , are we recording ?",
        "Yeah , we are .",
        "Great . So Morgan wanted the numbers by Friday right ?",
        "Mm-hmm . Liz said she'd send them .",
        "Uh , Dan , can you hear us in the back ?",
        "Yes .",
    ]
    assert segs[0].speaker == "me011"
    assert segs[0].start == pytest.approx(1.25)
    assert segs[0].end == pytest.approx(3.9)
    assert segs[-1].speaker == "me013"
    assert all(s.end >= s.start for s in segs)
    assert "breath" not in " ".join(texts)
    assert "door slam" not in " ".join(texts)


def test_vtt_rolling_captions_merge_and_speakers(fixtures_dir: Path) -> None:
    segs = webvtt.parse(fixtures_dir / "sample.vtt")
    texts = [s.text for s in segs]
    assert texts[0] == "Hello everyone, welcome back."
    assert segs[0].speaker == "Tim"
    assert segs[0].start == pytest.approx(1.0)
    assert segs[0].end == pytest.approx(5.0)  # merged with identical repeat
    assert "Today Linus is here." in texts
    assert texts.count("Today Linus is here.") == 1
    linus = next(s for s in segs if s.speaker == "Linus")
    assert linus.text == "Hi Tim & thanks for having me."
    assert linus.start == pytest.approx(10.0)
    assert texts[-1] == "[Musik]"
    assert segs[-1].start == pytest.approx(62.0)


def test_srt_parses_multiline_and_skips_empty(fixtures_dir: Path) -> None:
    segs = webvtt.parse(fixtures_dir / "sample.srt")
    assert [s.text for s in segs] == [
        "Bonjour à tous,",
        "et bienvenue dans Libre à vous ! Aujourd'hui Frédéric est avec nous.",
        "♪",
    ]
    assert segs[1].start == pytest.approx(2.1)
    assert segs[1].end == pytest.approx(4.9)


def test_plain_single_segment(tmp_path: Path) -> None:
    p = tmp_path / "ref.txt"
    p.write_text("  Michael   kam  gestern .\n", encoding="utf-8")
    segs = plain.parse(p, 3.2)
    assert len(segs) == 1
    assert segs[0].text == "Michael kam gestern ."
    assert (segs[0].start, segs[0].end) == (0.0, 3.2)
    assert plain.parse_text("   ", 1.0) == []


def test_parse_reference_sniffs_and_validates(fixtures_dir: Path, tmp_path: Path) -> None:
    assert sniff_format(Path("x.MRT")) == "icsi_mrt"
    assert sniff_format(Path("x.unknown")) is None
    assert len(parse_reference(fixtures_dir / "sample.vtt")) == len(
        webvtt.parse(fixtures_dir / "sample.vtt")
    )
    with pytest.raises(ValueError, match="cannot determine"):
        parse_reference(tmp_path / "x.unknown")
    p = tmp_path / "t.txt"
    p.write_text("hi", encoding="utf-8")
    with pytest.raises(ValueError, match="duration_s"):
        parse_reference(p)
    assert parse_reference(p, duration_s=2.0)[0].end == 2.0
