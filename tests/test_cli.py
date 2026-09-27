from __future__ import annotations

from typer.testing import CliRunner

from wakewordworld import __version__
from wakewordworld.cli import app

runner = CliRunner()


def test_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_sources_validate_repo() -> None:
    result = runner.invoke(app, ["sources", "validate"])
    assert result.exit_code == 0, result.stdout
    assert "valid" in result.stdout


def test_sources_list_repo() -> None:
    result = runner.invoke(app, ["sources", "list"])
    assert result.exit_code == 0, result.stdout
