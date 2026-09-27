from __future__ import annotations

import json
from pathlib import Path

from wakewordworld.report.build import (
    export_json,
    leaderboard,
    load_runs,
    per_domain,
    per_language,
    sealed_gap,
)


def test_load_runs(results_root: Path) -> None:
    runs = load_runs(results_root)
    assert [r.engine_id for r in runs] == ["good", "steady"]
    assert runs[0].verified is True
    assert runs[1].verified is False


def test_leaderboard_ranks_by_aut_hi(results_root: Path) -> None:
    board = leaderboard(load_runs(results_root))
    michael = board.filter(board["wake_word"] == "michael")
    assert michael.get_column("engine").to_list() == ["steady 2.1", "good 1.0"]
    assert michael.get_column("rank").to_list() == [1, 2]
    assert michael.get_column("preliminary").to_list() == [False, False]
    assert "frr_budget" in board.columns


def test_preliminary_rows_are_unranked(results_root: Path) -> None:
    board = leaderboard(load_runs(results_root))
    computer = board.filter(board["wake_word"] == "computer")
    by_engine = {r["engine"]: r for r in computer.iter_rows(named=True)}
    assert by_engine["steady 2.1"]["preliminary"] is True
    assert by_engine["steady 2.1"]["rank"] is None
    assert by_engine["good 1.0"]["preliminary"] is False
    assert by_engine["good 1.0"]["rank"] == 1
    # preliminary rows sort after ranked ones
    assert computer.get_column("engine").to_list()[0] == "good 1.0"


def test_pivots_and_gap(results_root: Path) -> None:
    runs = load_runs(results_root)
    lang = per_language(runs)
    assert set(lang.columns) >= {"engine", "wake_word", "de", "en"}
    assert lang.height == 2
    dom = per_domain(runs)
    assert "podcast" in dom.columns
    gap = sealed_gap(runs)
    assert gap.height == 2
    assert all(abs(g - 0.03) < 1e-9 for g in gap.get_column("gap").to_list())


def test_export_json(results_root: Path, tmp_path: Path) -> None:
    out = tmp_path / "lb.json"
    export_json(load_runs(results_root), out)
    payload = json.loads(out.read_text())
    assert payload["schema"] == "wakewordworld-leaderboard/1"
    assert {r["engine_id"] for r in payload["runs"]} == {"good", "steady"}
    assert len(payload["leaderboard"]) == 4
    first = payload["leaderboard"][0]
    for key in (
        "engine",
        "wake_word",
        "aut",
        "aut_lo",
        "aut_hi",
        "preliminary",
        "verified",
        "run_id",
    ):
        assert key in first
    assert payload["per_language"]


def test_empty_root(tmp_path: Path) -> None:
    runs = load_runs(tmp_path)
    assert runs == []
    assert leaderboard(runs).is_empty()
    assert per_language(runs).is_empty()


def test_false_accept_table_and_pool_status(results_root: Path) -> None:
    from wakewordworld.report.build import false_accept_table, load_runs, pool_status

    runs = load_runs(results_root)
    fa = false_accept_table(runs)
    assert not fa.is_empty()
    assert {"engine", "wake_word", "language", "fa_at_0.5", "n_positives"} <= set(fa.columns)
    assert (fa.get_column("language") == "all").any()
    st = pool_status(runs)
    assert st["audio_hours"] >= 0
    assert isinstance(st["positives"], dict)
