"""Noise sets: registry, download, and deterministic selection of noise segments.

Noise recordings are real environmental recordings (kitchens, cafés, streets, music,
babble) under permissive licences, listed in ``noise_sources.yaml``. They are stored
under ``<data root>/cache/noise/<set_id>/<archive stem>/`` and served as 16 kHz mono
float32 arrays.

Resampling: files that are not 16 kHz are resampled with linear interpolation by
default, which is adequate for broadband noise; for release runs pass
``resample="ffmpeg"`` to use ffmpeg's soxr resampler instead.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import tarfile
import tempfile
import zipfile
from collections import OrderedDict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Literal, Protocol

import httpx
import numpy as np
import soundfile as sf
import yaml
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from wakewordworld.util.hashing import short_id
from wakewordworld.util.paths import DataRoot

__all__ = [
    "SAMPLE_RATE",
    "NoiseArchive",
    "NoiseBank",
    "NoiseClip",
    "NoiseProvider",
    "NoiseSet",
    "ResampleMode",
    "download_noise_set",
    "load_audio_16k",
    "load_noise_sets",
    "resample_linear",
    "rng_for",
]

SAMPLE_RATE = 16_000
ResampleMode = Literal["numpy", "ffmpeg"]


class NoiseArchive(BaseModel):
    """One downloadable archive of a set."""

    model_config = ConfigDict(extra="forbid")

    url: str
    file: str
    md5: str | None = None
    sha256: str | None = None
    categories: list[str] = Field(default_factory=list)
    size_note: str | None = None

    @property
    def stem(self) -> str:
        """Archive file name without its (possibly double) extension."""
        name = self.file
        for ext in (".tar.gz", ".tgz", ".zip", ".tar"):
            if name.endswith(ext):
                return name[: -len(ext)]
        return Path(name).stem


class NoiseSet(BaseModel):
    """A registered noise or RIR set."""

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    kind: Literal["noise", "rir"]
    licence: str
    licence_evidence: str
    url: str
    attribution: str
    glob: str = "**/*.wav"
    sample_rate: int = SAMPLE_RATE
    download_default: bool = True
    size_note: str | None = None
    archives: list[NoiseArchive] = Field(default_factory=list)
    path_categories: dict[str, list[str]] = Field(default_factory=dict)

    def categories_for(self, archive_stem: str, member_path: str) -> frozenset[str]:
        """Categories of a file given its archive and path inside the archive."""
        cats: set[str] = set()
        for a in self.archives:
            if a.stem == archive_stem:
                cats.update(a.categories)
        for prefix, values in self.path_categories.items():
            if member_path.startswith(prefix) or ("/" + prefix) in ("/" + member_path):
                cats.update(values)
        return frozenset(cats)


def load_noise_sets(path: Path | None = None) -> dict[str, NoiseSet]:
    """Load the registry (packaged ``noise_sources.yaml`` by default)."""
    if path is None:
        text = (
            resources.files("wakewordworld.augment")
            .joinpath("noise_sources.yaml")
            .read_text("utf-8")
        )
    else:
        text = path.read_text(encoding="utf-8")
    raw = yaml.safe_load(text)
    sets = [NoiseSet.model_validate(s) for s in raw["sets"]]
    return {s.id: s for s in sets}


def rng_for(seed: str, *parts: str) -> np.random.Generator:
    """Deterministic generator keyed by a seed string and identifying parts."""
    return np.random.default_rng(int(short_id(seed, *parts), 16))


def resample_linear(x: NDArray[np.float32], sr_in: int, sr_out: int) -> NDArray[np.float32]:
    """Linear-interpolation resampling (adequate for broadband noise, not for speech)."""
    if sr_in == sr_out or x.size == 0:
        return x
    n_out = round(x.shape[0] * sr_out / sr_in)
    t_in = np.arange(x.shape[0], dtype=np.float64) / sr_in
    t_out = np.arange(n_out, dtype=np.float64) / sr_out
    return np.interp(t_out, t_in, x.astype(np.float64)).astype(np.float32)


def _ffmpeg_binary() -> str:
    for candidate in ("/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"):
        if Path(candidate).exists():
            return candidate
    found = shutil.which("ffmpeg")
    if found is None:
        msg = "ffmpeg not found; install it or use resample='numpy'"
        raise RuntimeError(msg)
    return found


def load_audio_16k(path: Path, *, resample: ResampleMode = "numpy") -> NDArray[np.float32]:
    """Load any audio file as 16 kHz mono float32 in [-1, 1]."""
    if resample == "ffmpeg":
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "x.wav"
            cmd = [
                _ffmpeg_binary(),
                "-v",
                "error",
                "-y",
                "-i",
                str(path),
                "-ac",
                "1",
                "-ar",
                str(SAMPLE_RATE),
                "-af",
                "aresample=resampler=soxr",
                "-f",
                "wav",
                str(out),
            ]
            subprocess.run(cmd, check=True, capture_output=True, timeout=600)
            data, _ = sf.read(out, dtype="float32", always_2d=True)
            return np.ascontiguousarray(data.mean(axis=1), dtype=np.float32)
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    mono = np.ascontiguousarray(data.mean(axis=1), dtype=np.float32)
    return resample_linear(mono, int(sr), SAMPLE_RATE)


@dataclass(frozen=True)
class NoiseClip:
    """A selectable noise (or RIR) file."""

    path: Path
    set_id: str
    categories: frozenset[str]
    attribution: str


class NoiseProvider(Protocol):
    """What the lane needs from a noise source (so tests can use fakes)."""

    def pick(self, chunk_id: str, categories: Sequence[str], *, seed: str) -> NoiseClip:
        """Deterministically choose a clip matching any of ``categories``."""
        ...

    def load(self, clip: NoiseClip) -> NDArray[np.float32]:
        """Load a clip as 16 kHz mono float32."""
        ...


def _checksum_ok(path: Path, archive: NoiseArchive) -> bool:
    if archive.sha256 and archive.sha256 != "unpinned":
        h = hashlib.sha256()
    elif archive.md5 and archive.md5 != "unpinned":
        h = hashlib.md5()
    else:
        return True
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    expected = archive.sha256 if h.name == "sha256" else archive.md5
    return h.hexdigest() == expected


def _extract(archive_path: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as zf:
            for info in zf.infolist():
                p = Path(info.filename)
                if p.is_absolute() or ".." in p.parts:
                    continue
                zf.extract(info, dest)
    elif tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path) as tf:
            members = [
                m
                for m in tf.getmembers()
                if not Path(m.name).is_absolute() and ".." not in Path(m.name).parts
            ]
            tf.extractall(dest, members=members, filter="data")
    else:
        msg = f"{archive_path} is neither zip nor tar"
        raise ValueError(msg)


def download_noise_set(
    set_id: str,
    data_root: DataRoot,
    *,
    sets: dict[str, NoiseSet] | None = None,
    archives: Iterable[str] | None = None,
    force_unverified: bool = False,
    client: httpx.Client | None = None,
) -> list[Path]:
    """Download and extract a set into ``cache/noise/<set_id>/``; returns extracted dirs.

    Sets with ``download_default: false`` are skipped unless ``force_unverified``.
    """
    registry = sets or load_noise_sets()
    try:
        nset = registry[set_id]
    except KeyError as exc:
        msg = f"unknown noise set {set_id!r}; known: {', '.join(sorted(registry))}"
        raise LookupError(msg) from exc
    if not nset.download_default and not force_unverified:
        msg = (
            f"{set_id} is not downloaded by default ({nset.size_note or nset.licence}); pass force"
        )
        raise PermissionError(msg)
    wanted = set(archives) if archives is not None else None
    base = data_root.cache / "noise" / set_id
    base.mkdir(parents=True, exist_ok=True)
    own_client = client is None
    cl = client or httpx.Client(timeout=120.0, follow_redirects=True)
    out: list[Path] = []
    try:
        for a in nset.archives:
            if wanted is not None and a.file not in wanted and a.stem not in wanted:
                continue
            dest = base / a.stem
            if dest.exists() and any(dest.iterdir()):
                out.append(dest)
                continue
            tmp = base / (a.file + ".part")
            with cl.stream("GET", a.url) as resp:
                resp.raise_for_status()
                with tmp.open("wb") as fh:
                    for block in resp.iter_bytes(1 << 20):
                        fh.write(block)
            if not _checksum_ok(tmp, a):
                tmp.unlink(missing_ok=True)
                msg = f"checksum mismatch for {a.file}"
                raise ValueError(msg)
            final = base / a.file
            tmp.replace(final)
            _extract(final, dest)
            final.unlink(missing_ok=True)
            out.append(dest)
    finally:
        if own_client:
            cl.close()
    return out


class NoiseBank:
    """Locally available noise files of one or more sets, with deterministic picking."""

    def __init__(
        self,
        data_root: DataRoot,
        set_ids: Sequence[str],
        *,
        sets: dict[str, NoiseSet] | None = None,
        resample: ResampleMode = "numpy",
        cache_items: int = 8,
    ) -> None:
        self._sets = sets or load_noise_sets()
        self._resample: ResampleMode = resample
        self._cache: OrderedDict[Path, NDArray[np.float32]] = OrderedDict()
        self._cache_items = cache_items
        self._clips: list[NoiseClip] = []
        for sid in set_ids:
            nset = self._sets[sid]
            base = data_root.cache / "noise" / sid
            if not base.exists():
                continue
            for archive_dir in sorted(p for p in base.iterdir() if p.is_dir()):
                for f in sorted(archive_dir.glob(nset.glob)):
                    if not f.is_file():
                        continue
                    member = f.relative_to(archive_dir).as_posix()
                    cats = nset.categories_for(archive_dir.name, member)
                    self._clips.append(NoiseClip(f, sid, cats, nset.attribution))

    @property
    def clips(self) -> list[NoiseClip]:
        """All discovered clips."""
        return list(self._clips)

    def candidates(self, categories: Sequence[str]) -> list[NoiseClip]:
        """Clips matching any of the categories (all clips when none given)."""
        if not categories:
            return list(self._clips)
        wanted = set(categories)
        return [c for c in self._clips if c.categories & wanted]

    def pick(self, chunk_id: str, categories: Sequence[str], *, seed: str) -> NoiseClip:
        """Deterministic choice keyed by seed and chunk id."""
        cands = self.candidates(categories)
        if not cands:
            msg = f"no noise clips for categories {list(categories)}; download a set first"
            raise LookupError(msg)
        rng = rng_for(seed, chunk_id, ",".join(sorted(categories)))
        return cands[int(rng.integers(0, len(cands)))]

    def load(self, clip: NoiseClip) -> NDArray[np.float32]:
        """Load (and cache) a clip as 16 kHz mono float32."""
        if clip.path in self._cache:
            self._cache.move_to_end(clip.path)
            return self._cache[clip.path]
        audio = load_audio_16k(clip.path, resample=self._resample)
        self._cache[clip.path] = audio
        while len(self._cache) > self._cache_items:
            self._cache.popitem(last=False)
        return audio
