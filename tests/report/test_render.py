from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from wakewordworld.cli import app
from wakewordworld.report.build import load_runs
from wakewordworld.report.render import render_report
from wakewordworld.report.svg import det_svg, interval_bar_svg

runner = CliRunner()


def test_render_report_structure(results_root: Path, tmp_path: Path) -> None:
    out = render_report(load_runs(results_root), title="Test report", out_path=tmp_path / "r.html")
    html = out.read_text(encoding="utf-8")
    assert html.startswith("<!DOCTYPE html>")
    assert "<title>Test report</title>" in html
    assert "Wake word “michael”" in html
    assert "Wake word “computer”" in html
    assert 'id="per-language"' in html
    assert 'id="per-domain"' in html
    assert 'id="sealed"' in html
    assert 'id="provenance"' in html
    assert "<svg" in html
    assert 'class="det"' in html
    assert 'class="tag verified"' in html
    assert 'class="tag prelim"' in html
    assert 'class="preliminary"' in html
    assert "0.140 [0.120, 0.160]" in html  # CI formatting for steady's AUT
    # ranking order in the michael table: steady first
    michael = html[html.index("Wake word “michael”") :]
    assert michael.index("steady 2.1") < michael.index("good 1.0")
    assert "<script" not in html


def test_cli_build_and_list(results_root: Path, tmp_path: Path) -> None:
    out = tmp_path / "site" / "index.html"
    js = tmp_path / "site" / "leaderboard.json"
    result = runner.invoke(
        app,
        ["report", "build", "--results", str(results_root), "--out", str(out), "--json", str(js)],
    )
    assert result.exit_code == 0, result.stdout
    assert out.exists()
    assert json.loads(js.read_text())["leaderboard"]
    result = runner.invoke(app, ["report", "list", "--results", str(results_root)])
    assert result.exit_code == 0, result.stdout
    assert "steady" in result.stdout


def test_det_svg_handles_zero_fa_and_orders_steps() -> None:
    svg = det_svg(
        [
            ("a", [(0.0, 0.5), (0.3, 0.2), (5.0, 0.1)]),
            ("b", [(float("nan"), 0.1), (1.0, 0.3)]),
        ],
        title="x",
    )
    assert svg.startswith("<svg")
    assert svg.count("<path") == 2
    assert "<title>x</title>" in svg
    for budget in ("0.1", "0.5", "1", "3"):
        assert f">{budget}</text>" in svg
    assert "det-legend" in svg


def test_interval_bar_svg() -> None:
    assert "<circle" in interval_bar_svg(0.2, 0.1, 0.3)
    assert "<circle" not in interval_bar_svg(None, None, None)
    assert "0.200 [0.100, 0.300]" in interval_bar_svg(0.2, 0.1, 0.3)
