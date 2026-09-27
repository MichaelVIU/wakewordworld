"""Build augmented copies of a manifest's chunks under named conditions.

Layout
------
Augmented audio is written to ``<data root>/chunks_aug/<condition>/<source_id>/<chunk_id>.flac``
with the *same* chunk ids as the originals, so the word index applies unchanged (for
speed perturbation the index is rescaled instead, see below). Each condition gets a
manifest directory ``<manifests>/<version>-aug-<condition>/`` whose rows are copies of
the public rows with ``background_tags += ["aug:<condition>"]`` and an extended
attribution, plus a ``release.json`` whose notes describe the condition, noise sets and
seed.

Evaluating a condition
----------------------
:mod:`wakewordworld.eval.run` finds audio at ``data_root.chunks/<source_id>/<chunk_id>.flac``
and the index at ``data_root.index/<source_id>.parquet``. ``DataRoot`` is a frozen
dataclass whose properties derive from ``root``, so instead of patching it, a full data
root directory is materialised at ``<data root>/aug/<condition>/``: ``audio``,
``items``, ``transcripts``, ``scores``, ``originals`` and ``cache`` are symlinks to the
base root, ``chunks`` is a symlink to ``chunks_aug/<condition>``, and ``index`` is a
symlink to the base index except for speed conditions, where it is a real directory
holding rescaled word indexes. :func:`data_root_for_condition` returns that
``DataRoot``; pass it as ``--data-root`` to ``wakewordworld eval run``.

Order of operations per chunk: speed perturbation, then room impulse response, then
additive noise at the target SNR (noise is at the microphone, after the room).
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
import soundfile as sf
from numpy.typing import NDArray

from wakewordworld.augment.mix import mix_at_snr, speed_perturb
from wakewordworld.augment.noise import SAMPLE_RATE, NoiseProvider, rng_for
from wakewordworld.augment.rir import apply_rir
from wakewordworld.manifest.freeze import freeze
from wakewordworld.manifest.schema import ChunkRow, ManifestRelease
from wakewordworld.util.hashing import sha256_file
from wakewordworld.util.paths import DataRoot

__all__ = [
    "DEFAULT_CONDITIONS",
    "DEFAULT_NOISE_CATEGORIES",
    "Condition",
    "LaneResult",
    "build_lane",
    "data_root_for_condition",
    "parse_condition",
]

DEFAULT_NOISE_CATEGORIES: tuple[str, ...] = (
    "kitchen",
    "living_room",
    "cafe",
    "street",
    "music",
    "babble",
)
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_+\-.]{0,63}$")


@dataclass(frozen=True)
class Condition:
    """One augmentation condition."""

    name: str
    snr_db: float | None = None
    noise_categories: tuple[str, ...] = ()
    rir_set: str | None = None
    speed: float | None = None

    def __post_init__(self) -> None:
        if not _NAME_RE.match(self.name):
            msg = f"condition name {self.name!r} must match {_NAME_RE.pattern}"
            raise ValueError(msg)
        if self.speed is not None and not 0.9 <= self.speed <= 1.1:
            msg = "speed must be within [0.9, 1.1]"
            raise ValueError(msg)

    @property
    def is_identity(self) -> bool:
        """True for the ``clean`` sanity condition."""
        return self.snr_db is None and self.rir_set is None and (self.speed in (None, 1.0))


def parse_condition(spec: str) -> Condition:
    """Parse ``clean`` | ``snr<dB>[:cat,cat]`` | ``rir:<set>`` | ``speed:<factor>``.

    Several parts may be combined with ``+``, e.g. ``rir:rirs_noises+snr10:kitchen``.
    """
    snr: float | None = None
    cats: tuple[str, ...] = ()
    rir: str | None = None
    speed: float | None = None
    for part in spec.split("+"):
        part = part.strip()
        if part == "clean":
            continue
        if part.startswith("snr"):
            head, _, tail = part.partition(":")
            snr = float(head[3:])
            cats = tuple(c for c in tail.split(",") if c) if tail else DEFAULT_NOISE_CATEGORIES
        elif part.startswith("rir:"):
            rir = part[4:]
        elif part.startswith("speed:"):
            speed = float(part[6:])
        else:
            msg = f"unknown condition part {part!r}"
            raise ValueError(msg)
    name = spec.replace(":", "-").replace(",", "+").replace(" ", "")
    if spec == "clean" or name == "":
        name = "clean"
    return Condition(name=name, snr_db=snr, noise_categories=cats, rir_set=rir, speed=speed)


DEFAULT_CONDITIONS: tuple[Condition, ...] = (
    Condition("clean"),
    Condition("snr20", snr_db=20.0, noise_categories=DEFAULT_NOISE_CATEGORIES),
    Condition("snr10", snr_db=10.0, noise_categories=DEFAULT_NOISE_CATEGORIES),
    Condition("snr5", snr_db=5.0, noise_categories=DEFAULT_NOISE_CATEGORIES),
    Condition("snr0", snr_db=0.0, noise_categories=DEFAULT_NOISE_CATEGORIES),
    Condition("rir-rirs_noises", rir_set="rirs_noises"),
    Condition("speed-0.9", speed=0.9),
    Condition("speed-1.1", speed=1.1),
)


def _link(link: Path, target: Path) -> None:
    if link.is_symlink() or link.exists():
        return
    link.symlink_to(target, target_is_directory=True)


def data_root_for_condition(data_root: DataRoot, condition: Condition | str) -> DataRoot:
    """Materialise (idempotently) the per-condition data root and return it."""
    name = condition if isinstance(condition, str) else condition.name
    speed = None if isinstance(condition, str) else condition.speed
    aug_root = data_root.root / "aug" / name
    aug_root.mkdir(parents=True, exist_ok=True)
    chunks_dir = data_root.root / "chunks_aug" / name
    chunks_dir.mkdir(parents=True, exist_ok=True)
    for prop in ("audio", "items", "transcripts", "scores", "originals", "cache"):
        (data_root.root / prop).mkdir(parents=True, exist_ok=True)
        _link(aug_root / prop, data_root.root / prop)
    _link(aug_root / "chunks", chunks_dir)
    if speed is not None and speed != 1.0:
        (aug_root / "index").mkdir(exist_ok=True)
    else:
        data_root.index.mkdir(parents=True, exist_ok=True)
        _link(aug_root / "index", data_root.index)
    return DataRoot(root=aug_root)


def _load_rows(manifest_dir: Path) -> list[ChunkRow]:
    p = manifest_dir / "chunks.jsonl"
    if not p.exists():
        msg = f"no chunks.jsonl in {manifest_dir}"
        raise FileNotFoundError(msg)
    with p.open("r", encoding="utf-8") as fh:
        return [ChunkRow.model_validate_json(line) for line in fh if line.strip()]


def _load_release(manifest_dir: Path) -> ManifestRelease:
    p = manifest_dir / "release.json"
    if p.exists():
        return ManifestRelease.model_validate_json(p.read_text(encoding="utf-8"))
    return ManifestRelease(
        version="0.0.0",
        created_at=datetime.now(UTC),
        n_chunks=0,
        n_files=0,
        hours_by_language={},
        hours_by_tier={},
        sources=[],
    )


def _read_chunk(path: Path) -> NDArray[np.float32]:
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    if sr != SAMPLE_RATE:
        msg = f"{path}: expected {SAMPLE_RATE} Hz, got {sr}"
        raise ValueError(msg)
    return np.ascontiguousarray(data[:, 0], dtype=np.float32)


def _write_chunk(path: Path, audio: NDArray[np.float32]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".flac.tmp")
    pcm = np.clip(np.round(audio * 32767.0), -32768, 32767).astype(np.int16)
    sf.write(tmp, pcm, SAMPLE_RATE, subtype="PCM_16", format="FLAC")
    tmp.replace(path)


def _rescale_index(src: Path, dst: Path, factor: float) -> None:
    df = pl.read_parquet(src)
    df = df.with_columns(
        (pl.col("start_s") / factor).alias("start_s"), (pl.col("end_s") / factor).alias("end_s")
    )
    dst.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(dst)


@dataclass
class LaneResult:
    """What :func:`build_lane` produced for one condition."""

    condition: Condition
    manifest_dir: Path
    data_root: DataRoot
    n_chunks: int
    n_skipped: int
    provenance: list[dict[str, object]] = field(default_factory=list)


def build_lane(
    data_root: DataRoot,
    manifest_dir: Path,
    *,
    conditions: Sequence[Condition],
    out_manifests_dir: Path,
    noise_bank: NoiseProvider | None = None,
    rir_bank: NoiseProvider | None = None,
    seed: str = "wakewordworld-aug",
    limit: int | None = None,
    force: bool = False,
) -> list[LaneResult]:
    """Create augmented chunks and manifests for every condition.

    Only the public rows of ``manifest_dir`` are used (the sealed split is never
    augmented on disk outside maintainer runs). Existing augmented chunks are reused
    unless ``force``.
    """
    rows = _load_rows(manifest_dir)
    if limit is not None:
        rows = rows[:limit]
    base_release = _load_release(manifest_dir)
    results: list[LaneResult] = []
    for cond in conditions:
        if cond.snr_db is not None and noise_bank is None:
            msg = f"condition {cond.name} needs a noise bank"
            raise ValueError(msg)
        if cond.rir_set is not None and rir_bank is None:
            msg = f"condition {cond.name} needs an RIR bank"
            raise ValueError(msg)
        aug_root = data_root_for_condition(data_root, cond)
        out_rows: list[ChunkRow] = []
        provenance: list[dict[str, object]] = []
        skipped = 0
        rescaled_sources: set[str] = set()
        noise_attributions: set[str] = set()
        for row in rows:
            src_path = data_root.chunks / row.source_id / f"{row.chunk_id}.flac"
            if not src_path.exists():
                skipped += 1
                continue
            dst_path = aug_root.chunks / row.source_id / f"{row.chunk_id}.flac"
            prov: dict[str, object] = {"chunk_id": row.chunk_id, "condition": cond.name}
            attribution = row.attribution
            if dst_path.exists() and not force:
                prov["reused"] = True
            else:
                audio = _read_chunk(src_path)
                if cond.speed is not None and cond.speed != 1.0:
                    audio = speed_perturb(audio, cond.speed)
                    prov["speed"] = cond.speed
                if cond.rir_set is not None and rir_bank is not None:
                    clip = rir_bank.pick(row.chunk_id, [], seed=seed + ":" + cond.name)
                    audio = apply_rir(audio, rir_bank.load(clip))
                    prov["rir_file"] = str(clip.path)
                    prov["rir_set"] = clip.set_id
                    noise_attributions.add(clip.attribution)
                if cond.snr_db is not None and noise_bank is not None:
                    clip = noise_bank.pick(
                        row.chunk_id, list(cond.noise_categories), seed=seed + ":" + cond.name
                    )
                    rng = rng_for(seed, cond.name, row.chunk_id, "offset")
                    mix = mix_at_snr(audio, noise_bank.load(clip), cond.snr_db, rng=rng)
                    audio = mix.audio
                    prov.update(
                        {
                            "noise_file": str(clip.path),
                            "noise_set": clip.set_id,
                            "noise_offset": mix.noise_offset,
                            "gain_db": mix.gain_db,
                            "achieved_snr_db": mix.achieved_snr_db,
                            "snr_db": cond.snr_db,
                        }
                    )
                    noise_attributions.add(clip.attribution)
                _write_chunk(dst_path, audio)
            if (
                cond.speed is not None
                and cond.speed != 1.0
                and row.source_id not in rescaled_sources
            ):
                src_idx = data_root.index / f"{row.source_id}.parquet"
                if src_idx.exists():
                    _rescale_index(src_idx, aug_root.index / f"{row.source_id}.parquet", cond.speed)
                rescaled_sources.add(row.source_id)
            duration = float(sf.info(dst_path).duration)
            out_rows.append(
                row.model_copy(
                    update={
                        "background_tags": [*row.background_tags, f"aug:{cond.name}"],
                        "duration_s": duration,
                        "audio_sha256": sha256_file(dst_path) if row.audio_sha256 else None,
                        "attribution": attribution,
                    }
                )
            )
            provenance.append(prov)
        if noise_attributions:
            suffix = "; augmented with: " + " | ".join(sorted(noise_attributions))
            out_rows = [
                r.model_copy(update={"attribution": r.attribution + suffix}) for r in out_rows
            ]
        notes = json.dumps(
            {
                "lane": "augmented",
                "condition": asdict(cond),
                "seed": seed,
                "base_manifest": str(manifest_dir),
                "noise_attributions": sorted(noise_attributions),
            },
            sort_keys=True,
        )
        hours_lang: dict[str, float] = {}
        hours_tier: dict[str, float] = {}
        for r in out_rows:
            hours_lang[r.language] = hours_lang.get(r.language, 0.0) + r.duration_s / 3600.0
            hours_tier[r.licence_tier] = hours_tier.get(r.licence_tier, 0.0) + r.duration_s / 3600.0
        release = ManifestRelease(
            version=base_release.version,
            created_at=datetime.now(UTC),
            git_commit=base_release.git_commit,
            n_chunks=len(out_rows),
            n_files=len({r.file_id for r in out_rows}),
            hours_by_language=hours_lang,
            hours_by_tier=hours_tier,
            sources=sorted({r.source_id for r in out_rows}),
            notes=notes,
        )
        out_dir = out_manifests_dir / f"{base_release.version}-aug-{cond.name}"
        freeze(out_rows, release, out_dir, force=True)
        prov_path = aug_root.root / "provenance.jsonl"
        with prov_path.open("w", encoding="utf-8") as fh:
            for p in provenance:
                fh.write(json.dumps(p, sort_keys=True, default=str) + "\n")
        results.append(LaneResult(cond, out_dir, aug_root, len(out_rows), skipped, provenance))
    return results
