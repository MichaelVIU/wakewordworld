"""Render the Hugging Face dataset card (``README.md``) for a public release.

The card carries the gating terms, one dataset configuration per licence family (so a
CC0 download never mixes with CC BY-SA material), and the attribution table copied from
the frozen manifest's ``sources.md``.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path

import yaml

from wakewordworld.licences import LicenceTier
from wakewordworld.manifest.schema import CANARY, ChunkRow, ManifestRelease

__all__ = [
    "LICENCE_FAMILIES",
    "hf_licence_id",
    "licence_family",
    "load_manifest",
    "render_dataset_card",
]

LICENCE_FAMILIES: tuple[str, ...] = ("cc0", "cc-by", "cc-by-sa", "other-open")
"""Dataset configurations; each holds only audio under one licence family."""

_AGREE = "I agree to the terms above and will not train wake word models on this audio"
_TAKEDOWN = (
    "Open a takedown issue on the repository or write to the maintainers (see repository profile)."
)


def hf_licence_id(spdx: str) -> str:
    """Lower-case Hugging Face licence id for an SPDX-like identifier."""
    lic = spdx.lower()
    if lic in {"public-domain", "cc-pddc"}:
        return "cc0-1.0"
    # Jurisdiction ports (cc-by-3.0-de) are not HF ids; use the base licence.
    parts = lic.split("-")
    if lic.startswith("cc-by") and len(parts[-1]) == 2 and parts[-1].isalpha():
        lic = "-".join(parts[:-1])
    if lic.endswith("-unversioned"):
        lic = lic.removesuffix("-unversioned")
    return lic


def licence_family(spdx: str) -> str:
    """Map a licence to its dataset configuration."""
    lic = spdx.upper()
    if lic.startswith("CC0") or lic in {"PUBLIC-DOMAIN", "CC-PDDC"}:
        return "cc0"
    if lic.startswith("CC-BY-SA") or lic == "LAL-1.3":
        return "cc-by-sa"
    if lic.startswith("CC-BY-"):
        return "cc-by"
    return "other-open"


def load_manifest(manifest_dir: Path) -> tuple[list[ChunkRow], ManifestRelease]:
    """Read the public rows and release header of a frozen manifest."""
    rows: list[ChunkRow] = []
    with (manifest_dir / "chunks.jsonl").open("r", encoding="utf-8") as fh:
        rows.extend(ChunkRow.model_validate_json(line) for line in fh if line.strip())
    release = ManifestRelease.model_validate_json(
        (manifest_dir / "release.json").read_text(encoding="utf-8")
    )
    return rows, release


def _size_category(hours: float) -> str:
    # HF size categories count rows; we use chunks (10-20 min) so map via hours.
    n = int(hours * 4)
    for limit, label in ((1_000, "n<1K"), (10_000, "1K<n<10K"), (100_000, "10K<n<100K")):
        if n < limit:
            return label
    return "100K<n<1M"


def _gated_prompt() -> str:
    return (
        "By requesting access you agree to: (1) keep the attribution recorded in the "
        "`attribution` column when redistributing any clip; (2) not use this audio, or "
        "features derived from it, to train, fine-tune or calibrate wake word or keyword "
        "spotting models, because that would invalidate the benchmark; (3) honour "
        "share-alike terms for the `cc-by-sa` configuration; (4) acknowledge that the "
        f"data carries the canary string `{CANARY}`, whose appearance in a model's "
        "training data is evidence that benchmark material was used."
    )


def _front_matter(
    rows: Sequence[ChunkRow], release: ManifestRelease, families_present: Iterable[str]
) -> dict[str, object]:
    licences = sorted(
        {hf_licence_id(r.licence_spdx) for r in rows if r.licence_tier is LicenceTier.A}
    )
    languages = sorted({r.language.value for r in rows})
    hours_a = sum(r.duration_s for r in rows if r.licence_tier is LicenceTier.A) / 3600.0
    configs = [
        {"config_name": fam, "data_files": [{"split": "test", "path": f"data/{fam}/*.parquet"}]}
        for fam in LICENCE_FAMILIES
        if fam in set(families_present)
    ]
    configs.append(
        {"config_name": "metadata", "data_files": [{"split": "test", "path": "metadata.parquet"}]}
    )
    return {
        "license": licences,
        "language": languages,
        "task_categories": ["audio-classification"],
        "tags": [
            "wake-word",
            "keyword-spotting",
            "benchmark",
            "speech",
            f"version-{release.version}",
        ],
        "pretty_name": f"WakeWordWorld benchmark {release.version}",
        "size_categories": [_size_category(hours_a)],
        "configs": configs,
        "extra_gated_prompt": _gated_prompt(),
        "extra_gated_fields": {
            "Name": "text",
            "Affiliation": "text",
            "Intended use": {
                "type": "select",
                "options": [
                    "Evaluate a wake word engine",
                    "Reproduce published results",
                    "Speech research (not training wake word models)",
                    {"label": "Other", "value": "other"},
                ],
            },
            _AGREE: "checkbox",
        },
    }


def render_dataset_card(
    manifest_dir: Path,
    *,
    repo_id: str = "wakewordworld/benchmark",
    families_present: Iterable[str] | None = None,
    changelog: str | None = None,
) -> str:
    """Build the full README.md text for the dataset repository."""
    rows, release = load_manifest(manifest_dir)
    families = (
        set(families_present)
        if families_present is not None
        else {licence_family(r.licence_spdx) for r in rows if r.licence_tier is LicenceTier.A}
    )
    fm = _front_matter(rows, release, families)
    sources_md = (manifest_dir / "sources.md").read_text(encoding="utf-8")
    sources_body = sources_md.split("\n", 1)[1] if sources_md.startswith("# ") else sources_md

    hours_lang: dict[str, float] = defaultdict(float)
    for r in rows:
        hours_lang[r.language.value] += r.duration_s / 3600.0
    n_a = sum(1 for r in rows if r.licence_tier is LicenceTier.A)
    n_b = len(rows) - n_a
    hours_table = "\n".join(f"| {lang} | {h:.1f} |" for lang, h in sorted(hours_lang.items()))
    citation = json.dumps(
        {
            "title": "WakeWordWorld benchmark",
            "version": release.version,
            "url": f"https://huggingface.co/datasets/{repo_id}",
        },
        indent=2,
    )
    body = f"""# WakeWordWorld benchmark {release.version}

An independent, reproducible test set for wake word and keyword spotting engines built
only from real, openly licensed speech in English, German, French and Spanish. Every
chunk is fully transcribed with word timestamps so any word can be evaluated as a wake
word. Code, methodology and results: https://github.com/MichaelVIU/wakewordworld

## Contents

| language | hours in release |
|---|---|
{hours_table}

- `metadata` configuration: one row per evaluation chunk ({len(rows)} rows, {n_a} tier A
  with audio, {n_b} tier B metadata only).
- Audio configurations by licence family: {", ".join(f"`{f}`" for f in sorted(families))}.
  Each holds only clips under that family, so a download never mixes licences.
- `index/<source_id>.parquet`: word-level index (token, start, end, confidence) for
  public chunks; `transcripts/<source_id>.jsonl`: transcripts of public chunks.
- Chunks are 10 to 20 minutes, 16 kHz mono FLAC, cut at silences from their parent
  recording (`file_id`).

## What is not in this release

- **Tier B material** (non-commercial, no-derivatives or research-only licences): only
  the manifest row (source, URL, duration, tags) is published, never audio,
  transcripts or word index rows, because those are derivative works of the recording.
  Results on tier B audio are still published in the results repository.
- **The sealed split**: about ten percent of tier A parent files are withheld and used
  only in maintainer-run evaluations. The gap between public and sealed results is
  reported per engine. The sealed split rotates yearly; the previous one becomes public.

## How results are computed

See the methodology documents in the repository (`docs/methodology/protocol.md`,
`metrics.md`, `dataset.md`): fixed 80 ms streaming frames, identical detection semantics
for every engine, false accepts per hour per language and domain, DET curves with
cluster-bootstrap confidence intervals over parent files.

## Sources and attribution

Per-item attribution is in the `attribution` column of `metadata.parquet`.

{sources_body.strip()}

## Terms

Access is gated. Attribution is required; training wake word or keyword spotting
models on this audio is not permitted; share-alike terms apply to the `cc-by-sa`
configuration; the data contains the canary `{CANARY}`.

## Takedown

{_TAKEDOWN} Removed recordings are tombstoned in the next release, never silently
dropped.

## Citation

```json
{citation}
```

## Changelog

{changelog or f"- {release.version}: {release.notes or 'release'}"}
"""
    return "---\n" + yaml.safe_dump(fm, sort_keys=False, allow_unicode=True) + "---\n\n" + body
