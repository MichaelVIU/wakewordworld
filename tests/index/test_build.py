from __future__ import annotations

import json

import polars as pl
import pytest

from tests.index.conftest import make_chunk, make_spec, write_source
from wakewordworld.index.build import build_index, load_index
from wakewordworld.sources.spec import Language
from wakewordworld.transcribe.schema import Segment, Transcript, Word
from wakewordworld.util.paths import DataRoot


def test_build_index_chunk_boundaries_and_music(
    data_root: DataRoot, sample_transcript: Transcript
) -> None:
    spec = make_spec("src_en", "en")
    chunks = [
        make_chunk("f1", spec.id, "c1", 0.0, 30.0),
        make_chunk("f1", spec.id, "c2", 30.0, 30.0),
    ]
    write_source(data_root, spec, transcript=sample_transcript, chunks=chunks)
    out = build_index(data_root, spec)
    df = pl.read_parquet(out)

    # Music segment word skipped; "Michael" at 29.8 belongs to c1 by start time and is clipped.
    assert df.filter(pl.col("word") == "michael").height == 1
    michael = df.filter(pl.col("word") == "michael").row(0, named=True)
    assert michael["chunk_id"] == "c1"
    assert michael["start_s"] == 29.8
    assert michael["end_s"] == 30.0  # clipped to the chunk end
    assert michael["speaker"] == "B"
    assert not michael["utterance_start"]
    assert not michael["utterance_end"]

    # "I" and "do" start after 30 s and go to c2 with chunk-relative times.
    c2 = df.filter(pl.col("chunk_id") == "c2").sort("start_s")
    assert c2["word"].to_list() == ["i", "do"]
    assert c2["start_s"].to_list() == pytest.approx([1.0, 1.1])
    assert c2["utterance_end"].to_list() == [False, True]

    # Utterance flags and normalisation.
    jane = df.filter(pl.col("word") == "jane").row(0, named=True)
    assert jane["utterance_start"]
    assert not jane["utterance_end"]
    assert jane["raw"] == "Jane,"
    assert set(df["language"].to_list()) == {"en"}
    assert set(df["source_id"].to_list()) == {"src_en"}
    assert set(df["backend"].to_list()) == {"test:1"}

    meta = json.loads(out.with_suffix(".meta.json").read_text())
    assert meta["n_chunks"] == 2
    assert meta["n_chunks_indexed"] == 2
    assert meta["n_files_missing_transcript"] == 0
    assert meta["hours_indexed"] == round(60 / 3600, 4)
    assert meta["n_words"] == df.height == 9


def test_build_index_missing_transcript(data_root: DataRoot) -> None:
    spec = make_spec("src_de", "de")
    write_source(
        data_root, spec, transcript=None, chunks=[make_chunk("f1", spec.id, "c1", 0.0, 10.0)]
    )
    out = build_index(data_root, spec)
    df = pl.read_parquet(out)
    assert df.height == 0
    meta = json.loads(out.with_suffix(".meta.json").read_text())
    assert meta["n_files_missing_transcript"] == 1
    assert meta["n_chunks_indexed"] == 0


def test_elision_emits_extra_form(data_root: DataRoot) -> None:
    spec = make_spec("src_fr", "fr")
    t = Transcript(
        file_id="f1",
        source_id=spec.id,
        language=Language.FR,
        backend="test:1",
        origin="asr",
        duration_s=10.0,
        segments=[
            Segment(
                start=0,
                end=3,
                text="c'est d'Anne",
                words=[
                    Word(text="c'est", start=0.1, end=0.4),
                    Word(text="d'Anne", start=0.5, end=0.9),
                ],
            )
        ],
    )
    write_source(data_root, spec, transcript=t, chunks=[make_chunk("f1", spec.id, "c1", 0.0, 10.0)])
    df = pl.read_parquet(build_index(data_root, spec))
    assert sorted(df["word"].to_list()) == ["anne", "c'est", "d'anne", "est"]


def test_load_index_concatenates(data_root: DataRoot, sample_transcript: Transcript) -> None:
    spec = make_spec("src_en", "en")
    write_source(
        data_root,
        spec,
        transcript=sample_transcript,
        chunks=[make_chunk("f1", spec.id, "c1", 0.0, 60.0)],
    )
    build_index(data_root, spec)
    assert load_index(data_root).height == load_index(data_root, ["src_en"]).height == 9
    assert load_index(data_root, ["nope"]).height == 0
