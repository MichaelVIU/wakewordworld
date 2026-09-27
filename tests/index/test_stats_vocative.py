from __future__ import annotations

import polars as pl
import pytest

from tests.index.conftest import make_chunk, make_spec, write_source
from wakewordworld.index.build import INDEX_COLUMNS, build_index, load_index
from wakewordworld.index.lexicon import NameLexicon, load_names
from wakewordworld.index.stats import candidate_wake_words, chunk_file_map, name_table, word_query
from wakewordworld.index.vocative import flag_vocatives
from wakewordworld.sources.spec import Language
from wakewordworld.transcribe.schema import Transcript
from wakewordworld.util.paths import DataRoot


def _rows(*items: tuple[str, float, float, bool, bool]) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "chunk_id": "c",
                "word": w,
                "raw": w,
                "start_s": s,
                "end_s": e,
                "confidence": None,
                "speaker": None,
                "utterance_start": us,
                "utterance_end": ue,
                "backend": "t",
                "source_id": "s",
                "language": "en",
            }
            for w, s, e, us, ue in items
        ],
        schema=INDEX_COLUMNS,
    )


def test_lexicons_load() -> None:
    for lang in Language:
        names = load_names(lang)
        assert len(names) >= 150, lang
        assert "michael" in names
    assert "michaela" in load_names(Language.DE)
    assert "michel" in load_names(Language.FR)
    assert "miguel" in load_names(Language.ES)


def test_extension_file(data_root: DataRoot) -> None:
    (data_root.cache / "names_extra_en.txt").write_text("zaphod\n# comment\n", encoding="utf-8")
    lex = NameLexicon(data_root)
    assert lex.is_name("zaphod", Language.EN)
    assert not NameLexicon().is_name("zaphod", Language.EN)


def test_flag_vocatives_rules() -> None:
    df = _rows(
        ("jane", 0.5, 0.9, True, False),  # utterance start -> vocative
        ("can", 1.0, 1.2, False, False),
        ("you", 1.2, 1.4, False, False),
        ("michael", 1.4, 1.9, False, False),  # tightly embedded -> not vocative
        ("hear", 1.9, 2.2, False, False),
        ("morgan", 3.0, 3.4, False, False),  # gap before >= 0.3 -> vocative
        ("ok", 3.45, 3.6, False, True),  # not a name
    )
    out = flag_vocatives(df)
    got = dict(zip(out["word"].to_list(), out["vocative"].to_list(), strict=True))
    assert got == {
        "jane": True,
        "can": False,
        "you": False,
        "michael": False,
        "hear": False,
        "morgan": True,
        "ok": False,
    }
    assert out["is_name"].sum() == 3
    # original row order preserved
    assert out["word"].to_list() == df["word"].to_list()


def test_flag_vocatives_empty() -> None:
    out = flag_vocatives(pl.DataFrame(schema=INDEX_COLUMNS))
    assert "vocative" in out.columns
    assert out.height == 0


def test_name_table_and_query(data_root: DataRoot, sample_transcript: Transcript) -> None:
    spec = make_spec("src_en", "en")
    chunks = [
        make_chunk("f1", spec.id, "c1", 0.0, 30.0),
        make_chunk("f1", spec.id, "c2", 30.0, 30.0),
    ]
    write_source(data_root, spec, transcript=sample_transcript, chunks=chunks)
    build_index(data_root, spec)
    df = load_index(data_root)
    table = name_table(df, Language.EN, NameLexicon(), chunk_files=chunk_file_map(data_root))
    assert table.columns == [
        "name",
        "occurrences",
        "vocatives",
        "distinct_chunks",
        "distinct_files",
        "distinct_sources",
    ]
    by_name = {r["name"]: r for r in table.iter_rows(named=True)}
    assert by_name["jane"]["occurrences"] == 1
    assert by_name["jane"]["vocatives"] == 1
    assert by_name["michael"]["occurrences"] == 1
    assert by_name["michael"]["vocatives"] == 1  # 1.0 s pause before "Michael"
    assert by_name["michael"]["distinct_files"] == 1
    assert name_table(df, Language.DE).height == 0

    q = word_query(df, "michael")
    assert q.height == 1
    assert q.row(0, named=True)["chunk_id"] == "c1"
    assert word_query(df, "michael", language=Language.FR).height == 0

    cands = candidate_wake_words(table, min_occurrences=1, min_files=1)
    assert set(cands["name"].to_list()) == {"jane", "michael"}
    assert candidate_wake_words(table).height == 0


@pytest.mark.parametrize("word", ["yes", "do"])
def test_non_names_absent_from_table(
    data_root: DataRoot, sample_transcript: Transcript, word: str
) -> None:
    spec = make_spec("src_en", "en")
    write_source(
        data_root,
        spec,
        transcript=sample_transcript,
        chunks=[make_chunk("f1", spec.id, "c1", 0.0, 60.0)],
    )
    build_index(data_root, spec)
    table = name_table(load_index(data_root), Language.EN)
    assert word not in table["name"].to_list()
