"""``wakewordworld augment`` commands."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from wakewordworld.util.paths import DataRoot, repo_root

augment_app = typer.Typer(
    help="Augmented lane: real noise and room impulse responses at controlled levels.",
    no_args_is_help=True,
)
noise_app = typer.Typer(help="Noise and RIR set registry.", no_args_is_help=True)
lane_app = typer.Typer(help="Build and list augmented conditions.", no_args_is_help=True)
augment_app.add_typer(noise_app, name="noise")
augment_app.add_typer(lane_app, name="lane")
console = Console()

DataRootOpt = Annotated[Path | None, typer.Option("--data-root", envvar="WWW_DATA_ROOT")]


@noise_app.command("list")
def noise_list(data_root: DataRootOpt = None) -> None:
    """List registered noise/RIR sets and how many files are available locally."""
    from wakewordworld.augment.noise import load_noise_sets

    root = DataRoot.resolve(data_root)
    table = Table(title="Noise and RIR sets")
    for col in ("id", "kind", "licence", "default", "local files", "note"):
        table.add_column(col)
    for s in load_noise_sets().values():
        base = root.cache / "noise" / s.id
        n = sum(1 for _ in base.glob(s.glob)) if base.exists() else 0
        table.add_row(
            s.id,
            s.kind,
            s.licence,
            "yes" if s.download_default else "no",
            str(n),
            s.size_note or "",
        )
    console.print(table)


@noise_app.command("download")
def noise_download(
    set_id: Annotated[str, typer.Argument(help="Set id from `augment noise list`.")],
    archive: Annotated[
        list[str] | None, typer.Option("--archive", help="Only these archives.")
    ] = None,
    force_unverified: Annotated[bool, typer.Option("--force-unverified")] = False,
    data_root: DataRootOpt = None,
) -> None:
    """Download and extract a set into the data root cache."""
    from wakewordworld.augment.noise import download_noise_set

    dirs = download_noise_set(
        set_id, DataRoot.resolve(data_root), archives=archive, force_unverified=force_unverified
    )
    for d in dirs:
        console.print(f"ready: {d}")


@lane_app.command("build")
def lane_build(
    manifest: Annotated[Path, typer.Option("--manifest", help="Base manifest directory.")],
    condition: Annotated[
        list[str] | None,
        typer.Option(
            "--condition", help="clean | snr10:kitchen,babble | rir:rirs_noises | speed:1.1"
        ),
    ] = None,
    noise_set: Annotated[list[str] | None, typer.Option("--noise-set")] = None,
    limit: Annotated[int | None, typer.Option("--limit")] = None,
    seed: Annotated[str, typer.Option("--seed")] = "wakewordworld-aug",
    resample: Annotated[str, typer.Option("--resample", help="numpy | ffmpeg")] = "numpy",
    out_manifests: Annotated[Path | None, typer.Option("--out-manifests")] = None,
    force: Annotated[bool, typer.Option("--force")] = False,
    data_root: DataRootOpt = None,
) -> None:
    """Create augmented chunks and one manifest per condition."""
    from wakewordworld.augment.lane import DEFAULT_CONDITIONS, build_lane, parse_condition
    from wakewordworld.augment.noise import NoiseBank, ResampleMode
    from wakewordworld.augment.rir import RirBank

    if resample not in ("numpy", "ffmpeg"):
        raise typer.BadParameter("resample must be numpy or ffmpeg")
    mode: ResampleMode = "ffmpeg" if resample == "ffmpeg" else "numpy"
    root = DataRoot.resolve(data_root)
    conds = [parse_condition(c) for c in condition] if condition else list(DEFAULT_CONDITIONS)
    sets = noise_set or ["demand", "musan"]
    rir_sets = sorted({c.rir_set for c in conds if c.rir_set})
    needs_noise = any(c.snr_db is not None for c in conds)
    noise_bank = NoiseBank(root, sets, resample=mode) if needs_noise else None
    rir_bank = RirBank(root, rir_sets, resample=mode) if rir_sets else None
    results = build_lane(
        root,
        manifest,
        conditions=conds,
        out_manifests_dir=out_manifests or (repo_root() / "manifests"),
        noise_bank=noise_bank,
        rir_bank=rir_bank,
        seed=seed,
        limit=limit,
        force=force,
    )
    for r in results:
        console.print(
            f"[bold]{r.condition.name}[/bold]: {r.n_chunks} chunks ({r.n_skipped} skipped) -> "
            f"{r.manifest_dir}\n  evaluate with: wakewordworld eval run <engine> --manifest "
            f"{r.manifest_dir} --data-root {r.data_root.root}"
        )


@lane_app.command("list")
def lane_list(data_root: DataRootOpt = None) -> None:
    """List materialised conditions."""
    root = DataRoot.resolve(data_root)
    aug = root.root / "aug"
    if not aug.exists():
        console.print("no augmented conditions yet")
        return
    table = Table(title="Augmented conditions")
    for col in ("condition", "chunks", "data root"):
        table.add_column(col)
    for d in sorted(p for p in aug.iterdir() if p.is_dir()):
        n = sum(1 for _ in (d / "chunks").glob("*/*.flac")) if (d / "chunks").exists() else 0
        table.add_row(d.name, str(n), str(d))
    console.print(table)
