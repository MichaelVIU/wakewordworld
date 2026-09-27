"""``wakewordworld eval`` commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import polars as pl
import typer
from rich.console import Console
from rich.table import Table

from wakewordworld.eval.detect import DetectionConfig
from wakewordworld.eval.run import EvalPlan, run_evaluation, target_phrase
from wakewordworld.util.paths import DataRoot, repo_root

eval_app = typer.Typer(help="Run engines against the benchmark.", no_args_is_help=True)
console = Console()


def _parse_kv(pairs: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for p in pairs:
        if "=" not in p:
            msg = f"expected key=value, got {p!r}"
            raise typer.BadParameter(msg)
        k, v = p.split("=", 1)
        out[k.strip()] = v.strip()
    return out


@eval_app.command("engines")
def list_engines() -> None:
    """List registered engine adapters."""
    from wakewordworld.engines.registry import list_engines as _list

    for eid in _list():
        console.print(eid)


@eval_app.command("run")
def run(
    engine_id: Annotated[str, typer.Argument(help="Engine id, see `eval engines`.")],
    manifest_dir: Annotated[
        Path, typer.Option("--manifest", help="Manifest directory (manifests/<version>).")
    ],
    engine_config: Annotated[
        str | None, typer.Option("--config", help="Engine config as JSON string.")
    ] = None,
    target: Annotated[
        list[str] | None,
        typer.Option("--target", help="engine_key=index phrase, e.g. hey_jarvis='hey jarvis'."),
    ] = None,
    language: Annotated[list[str] | None, typer.Option("--language")] = None,
    limit: Annotated[int | None, typer.Option("--limit", help="Only the first N chunks.")] = None,
    n_boot: Annotated[int, typer.Option("--n-boot")] = 1000,
    force: Annotated[bool, typer.Option("--force", help="Re-score cached chunks.")] = False,
    debounce_s: Annotated[float, typer.Option("--debounce-s")] = 1.0,
    pre_s: Annotated[float, typer.Option("--pre-s")] = 0.5,
    post_s: Annotated[float, typer.Option("--post-s")] = 1.0,
    out: Annotated[
        Path | None, typer.Option("--out", help="Results root (default results/).")
    ] = None,
    data_root: Annotated[Path | None, typer.Option("--data-root", envvar="WWW_DATA_ROOT")] = None,
) -> None:
    """Score all chunks of a manifest with one engine and write results."""
    from wakewordworld.engines.registry import load_engine

    cfg = json.loads(engine_config) if engine_config else {}
    engine = load_engine(engine_id, **cfg)
    try:
        keys = engine.info.wake_words
        targets = {k: target_phrase(k) for k in keys}
        targets.update(_parse_kv(target or []))
        plan = EvalPlan(
            manifest_dir=manifest_dir,
            engine_id=engine_id,
            engine_config=cfg,
            wake_words=targets,
            detection=DetectionConfig(debounce_s=debounce_s, pre_s=pre_s, post_s=post_s),
            languages=tuple(language or ()),
            n_boot=n_boot,
            force_rescore=force,
            limit_chunks=limit,
        )
        run_dir = run_evaluation(
            DataRoot.resolve(data_root), engine, plan, out or (repo_root() / "results")
        )
    finally:
        engine.close()
    console.print(f"results written to [bold]{run_dir}[/bold]")
    show(run_dir)


@eval_app.command("show")
def show(run_dir: Annotated[Path, typer.Argument(help="A results/<run_id> directory.")]) -> None:
    """Print the headline table of a run."""
    df = pl.read_parquet(run_dir / "summary.parquet").filter(
        pl.col("slice_type").is_in(["all", "language"])
    )
    meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    table = Table(
        title=f"{meta['engine_id']} {meta['engine_version']} on {meta['manifest_version']}"
    )
    for col in (
        "wake word",
        "slice",
        "units",
        "pos",
        "neg h",
        "FRR@0.5",
        "95% CI",
        "FRR@1",
        "AUT",
        "lat p50",
    ):
        table.add_column(col)

    def fmt(v: object) -> str:
        return "-" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))

    for r in df.iter_rows(named=True):
        table.add_row(
            r["wake_word"],
            f"{r['slice_type']}={r['slice_value']}" if r["slice_type"] != "all" else "all",
            str(r["n_units"]),
            str(r["n_positives"]),
            f"{r['negative_hours']:.1f}",
            fmt(r["frr_at_0_5"]),
            f"[{fmt(r['frr_at_0_5_lo'])}, {fmt(r['frr_at_0_5_hi'])}]",
            fmt(r["frr_at_1"]),
            f"{r['aut']:.3f} [{r['aut_lo']:.3f}, {r['aut_hi']:.3f}]",
            fmt(r["latency_p50_s"]),
        )
    console.print(table)
