"""``wakewordworld transcribe`` commands."""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from wakewordworld.sources.spec import SourceSpec, load_source_specs
from wakewordworld.transcribe.backends import BackendUnavailable, check_backends, get_backend
from wakewordworld.transcribe.pipeline import (
    AlignBackend,
    iter_file_records,
    transcribe_source,
    transcript_path,
)
from wakewordworld.transcribe.schema import Transcript
from wakewordworld.util.paths import DataRoot, repo_root

transcribe_app = typer.Typer(help="Word-level transcription and alignment.", no_args_is_help=True)
console = Console()

DataRootOpt = Annotated[
    Path | None,
    typer.Option("--data-root", envvar="WWW_DATA_ROOT", help="Data directory (default ./data)."),
]
SourcesDirOpt = Annotated[
    Path | None, typer.Option("--sources-dir", help="Directory with source YAML files.")
]


def _select_specs(
    source_ids: list[str], all_sources: bool, sources_dir: Path | None
) -> list[SourceSpec]:
    specs = load_source_specs(sources_dir or repo_root() / "sources")
    if all_sources:
        return specs
    if not source_ids:
        console.print("[red]give at least one SOURCE_ID or --all[/red]")
        raise typer.Exit(code=2)
    by_id = {s.id: s for s in specs}
    missing = [s for s in source_ids if s not in by_id]
    if missing:
        console.print(f"[red]unknown source id(s): {', '.join(missing)}[/red]")
        raise typer.Exit(code=2)
    return [by_id[s] for s in source_ids]


@transcribe_app.command("run")
def run(
    source_ids: Annotated[list[str] | None, typer.Argument(help="Source ids.")] = None,
    all_sources: Annotated[bool, typer.Option("--all", help="All sources.")] = False,
    backend: Annotated[
        str, typer.Option(help="ASR backend: faster-whisper | parakeet-mlx | none.")
    ] = "faster-whisper",
    model: Annotated[str | None, typer.Option(help="Backend model name.")] = None,
    device: Annotated[str, typer.Option(help="faster-whisper device (auto|cpu|cuda).")] = "auto",
    compute_type: Annotated[str, typer.Option(help="faster-whisper compute type.")] = "auto",
    align: Annotated[
        str, typer.Option(help="Reference aligner: mms_fa | uniform | none.")
    ] = "mms_fa",
    limit: Annotated[int | None, typer.Option(help="Max files per source.")] = None,
    force: Annotated[bool, typer.Option("--force", help="Re-transcribe existing files.")] = False,
    no_music_tag: Annotated[
        bool, typer.Option("--no-music-tag", help="Skip music tagging.")
    ] = False,
    data_root: DataRootOpt = None,
    sources_dir: SourcesDirOpt = None,
) -> None:
    """Transcribe pending files of the given sources."""
    if align not in ("mms_fa", "uniform", "none"):
        console.print("[red]--align must be mms_fa, uniform or none[/red]")
        raise typer.Exit(code=2)
    root = DataRoot.resolve(data_root)
    specs = _select_specs(source_ids or [], all_sources, sources_dir)
    asr = None
    if backend != "none":
        opts: dict[str, object] = {}
        if model:
            opts["model"] = model
        if backend == "faster-whisper":
            opts.update(
                device=device, compute_type=compute_type, download_root=root.cache / "models"
            )
        try:
            asr = get_backend(backend, **opts)
        except BackendUnavailable as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=1) from exc
    align_backend: AlignBackend = align  # type: ignore[assignment]
    table = Table(title="Transcription")
    for col in ("source", "done", "skipped", "failed", "hours", "origins"):
        table.add_column(col)
    for spec in specs:
        summary = transcribe_source(
            root,
            spec,
            backend=asr,
            force=force,
            limit=limit,
            align_backend=align_backend,
            tag=not no_music_tag,
            progress=True,
        )
        table.add_row(
            spec.id,
            str(summary.done),
            str(summary.skipped),
            str(summary.failed),
            f"{summary.hours:.2f}",
            ", ".join(f"{k}={v}" for k, v in sorted(summary.origins.items())) or "-",
        )
    console.print(table)


@transcribe_app.command("status")
def status(
    source_ids: Annotated[list[str] | None, typer.Argument(help="Source ids.")] = None,
    all_sources: Annotated[bool, typer.Option("--all", help="All sources.")] = False,
    data_root: DataRootOpt = None,
    sources_dir: SourcesDirOpt = None,
) -> None:
    """Show how many files have transcripts, by origin."""
    root = DataRoot.resolve(data_root)
    specs = _select_specs(source_ids or [], all_sources or not source_ids, sources_dir)
    table = Table(title="Transcript status")
    for col in (
        "source",
        "files",
        "with transcript",
        "asr",
        "reference",
        "reference_aligned",
        "failures",
    ):
        table.add_column(col)
    for spec in specs:
        recs = list(iter_file_records(root, spec.id))
        origins = {"asr": 0, "reference": 0, "reference_aligned": 0}
        n_done = 0
        for rec in recs:
            p = transcript_path(root, spec.id, rec.file_id)
            if p.exists():
                n_done += 1
                with contextlib.suppress(Exception):  # corrupt file counts as missing origin
                    origins[Transcript.load(p).origin] += 1
        failures = root.transcripts / spec.id / "failures.jsonl"
        n_fail = sum(1 for _ in failures.open()) if failures.exists() else 0
        table.add_row(
            spec.id,
            str(len(recs)),
            str(n_done),
            str(origins["asr"]),
            str(origins["reference"]),
            str(origins["reference_aligned"]),
            str(n_fail),
        )
    console.print(table)


@transcribe_app.command("show")
def show(
    file_id: Annotated[str, typer.Argument(help="File id.")],
    source: Annotated[str, typer.Option("--source", help="Source id.")],
    n: Annotated[int, typer.Option(help="Number of words to print.")] = 40,
    data_root: DataRootOpt = None,
) -> None:
    """Print the first words of a transcript with timings."""
    root = DataRoot.resolve(data_root)
    p = transcript_path(root, source, file_id)
    if not p.exists():
        console.print(f"[red]no transcript at {p}[/red]")
        raise typer.Exit(code=1)
    t = Transcript.load(p)
    console.print(
        f"[bold]{t.file_id}[/bold] {t.language} {t.backend} origin={t.origin} "
        f"duration={t.duration_s:.1f}s segments={len(t.segments)} "
        f"music={sum(1 for s in t.segments if s.is_music)}"
    )
    table = Table()
    for col in ("start", "end", "word", "conf", "speaker"):
        table.add_column(col)
    for w in t.words()[:n]:
        table.add_row(
            f"{w.start:8.2f}",
            f"{w.end:8.2f}",
            w.text,
            f"{w.confidence:.2f}" if w.confidence is not None else "-",
            w.speaker or "-",
        )
    console.print(table)


@transcribe_app.command("check-backends")
def check() -> None:
    """Report which ASR backends can be constructed in this environment."""
    table = Table(title="ASR backends")
    table.add_column("backend")
    table.add_column("status")
    for name, problem in check_backends().items():
        table.add_row(
            name, "[green]ok[/green]" if problem is None else f"[yellow]{problem}[/yellow]"
        )
    console.print(table)
