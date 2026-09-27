from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from wakewordworld.cli import app

runner = CliRunner()


def test_status_all(tmp_path: Path) -> None:
    result = runner.invoke(app, ["ingest", "status", "--all", "--data-root", str(tmp_path)])
    assert result.exit_code == 0, result.stdout
    assert "Ingest status" in result.stdout


def test_unknown_source_id(tmp_path: Path) -> None:
    result = runner.invoke(app, ["ingest", "status", "nope", "--data-root", str(tmp_path)])
    assert result.exit_code == 2


def test_no_sources_given(tmp_path: Path) -> None:
    result = runner.invoke(app, ["ingest", "fetch", "--data-root", str(tmp_path)])
    assert result.exit_code == 2
