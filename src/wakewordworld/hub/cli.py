"""``wakewordworld hub`` commands: build, card, upload, zenodo."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from wakewordworld.hub.dataset_card import render_dataset_card
from wakewordworld.hub.publish import build_release_files, upload_dataset
from wakewordworld.hub.zenodo import ZenodoClient, ZenodoError, deposition_metadata
from wakewordworld.manifest.schema import ManifestRelease
from wakewordworld.util.paths import DataRoot, repo_root

hub_app = typer.Typer(help="Publish dataset releases (Hugging Face, Zenodo).", no_args_is_help=True)
console = Console()

ManifestOpt = Annotated[Path, typer.Option("--manifest", help="Frozen manifest directory.")]
RepoOpt = Annotated[str, typer.Option("--repo", help="Hugging Face dataset repo id.")]


@hub_app.command("build-release")
def build_release(
    manifest: ManifestOpt,
    out: Annotated[Path, typer.Option("--out", help="Output directory for the release files.")],
    repo: RepoOpt = "wakewordworld/benchmark",
    no_audio: Annotated[bool, typer.Option("--no-audio", help="Metadata and card only.")] = False,
    data_root: Annotated[Path | None, typer.Option("--data-root", envvar="WWW_DATA_ROOT")] = None,
) -> None:
    """Assemble the public dataset directory from a frozen manifest."""
    build = build_release_files(
        manifest, DataRoot.resolve(data_root), out, repo_id=repo, include_audio=not no_audio
    )
    console.print(
        f"release built at [bold]{build.out_dir}[/bold]: {build.n_rows} rows, "
        f"{build.n_audio} audio chunks, families {build.families}"
    )
    if build.missing_audio:
        console.print(f"[yellow]{len(build.missing_audio)} tier A chunks had no audio[/yellow]")


@hub_app.command("card")
def card(
    manifest: ManifestOpt,
    out: Annotated[Path, typer.Option("--out")] = Path("README.md"),
    repo: RepoOpt = "wakewordworld/benchmark",
) -> None:
    """Render the dataset card only."""
    out.write_text(render_dataset_card(manifest, repo_id=repo), encoding="utf-8")
    console.print(f"card written to [bold]{out}[/bold]")


@hub_app.command("upload")
def upload(
    directory: Annotated[Path, typer.Option("--dir", help="Directory from build-release.")],
    repo: RepoOpt,
    tag: Annotated[str | None, typer.Option("--tag", help="Manifest version to tag.")] = None,
    private: Annotated[bool, typer.Option("--private/--public")] = True,
) -> None:
    """Upload a release directory to a gated Hugging Face dataset repo."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        console.print("[red]HF_TOKEN is not set[/red]")
        raise typer.Exit(code=2)
    url = upload_dataset(directory, repo, token=token, private=private, revision_tag=tag)
    console.print(f"uploaded to [bold]{url}[/bold]")


@hub_app.command("zenodo")
def zenodo(
    directory: Annotated[Path, typer.Option("--dir", help="Directory from build-release.")],
    sandbox: Annotated[bool, typer.Option("--sandbox")] = False,
    publish: Annotated[bool, typer.Option("--publish", help="Publish (mints the DOI).")] = False,
    github_tag_url: Annotated[str | None, typer.Option("--github-tag-url")] = None,
    hf_url: Annotated[str | None, typer.Option("--hf-url")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Print metadata only.")] = False,
) -> None:
    """Deposit release metadata, checksums and card on Zenodo."""
    release = ManifestRelease.model_validate_json(
        (directory / "release.json").read_text(encoding="utf-8")
    )
    meta = deposition_metadata(
        release,
        citation_cff=repo_root() / "CITATION.cff",
        github_tag_url=github_tag_url,
        hf_dataset_url=hf_url,
    )
    if dry_run:
        console.print_json(json.dumps(meta))
        return
    try:
        client = ZenodoClient.from_env(sandbox=sandbox)
        dep = client.create_deposition(meta)
        for name in ("release.json", "SHA256SUMS", "sources.md", "README.md", "metadata.parquet"):
            p = directory / name
            if p.exists():
                client.upload_file(dep, p)
        if publish:
            dep = client.publish(int(dep["id"]))
    except ZenodoError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    console.print(f"deposition {dep.get('id')} at {dep.get('links', {}).get('html', '')}")
