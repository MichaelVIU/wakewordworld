"""Expansion of zip/tar originals into individual member files."""

from __future__ import annotations

import tarfile
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path, PurePosixPath

from wakewordworld.util.hashing import sha256_file
from wakewordworld.util.paths import DataRoot

__all__ = ["ArchiveMember", "expand_archive", "is_archive"]

_ARCHIVE_SUFFIXES = (".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tar.xz")


@dataclass(frozen=True)
class ArchiveMember:
    """An extracted member with its checksum."""

    path: Path
    member_path: str
    sha256: str


def is_archive(path: Path) -> bool:
    """Whether a file looks like a supported archive by name or magic."""
    name = path.name.lower()
    if name.endswith(_ARCHIVE_SUFFIXES):
        return True
    return zipfile.is_zipfile(path) or tarfile.is_tarfile(path)


def _safe_members_zip(zf: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    out: list[zipfile.ZipInfo] = []
    for info in zf.infolist():
        p = PurePosixPath(info.filename)
        if info.is_dir() or p.is_absolute() or ".." in p.parts:
            continue
        out.append(info)
    return out


def _safe_members_tar(tf: tarfile.TarFile) -> list[tarfile.TarInfo]:
    out: list[tarfile.TarInfo] = []
    for info in tf.getmembers():
        p = PurePosixPath(info.name)
        if not info.isfile() or p.is_absolute() or ".." in p.parts:
            continue
        out.append(info)
    return out


def _glob_match(member: str, pattern: str) -> bool:
    if fnmatch(member, pattern):
        return True
    # Support "**/*.wav" meaning "any depth" including the top level.
    if pattern.startswith("**/"):
        return fnmatch(member, pattern[3:]) or fnmatch(member, pattern)
    return False


def expand_archive(
    archive: Path,
    *,
    item_id: str,
    data_root: DataRoot,
    audio_glob: str = "**/*.wav",
) -> Iterator[ArchiveMember]:
    """Extract matching members once into ``cache/extract/<item_id>/`` and yield them.

    Extraction is idempotent: already-present members are not re-extracted. Members
    with absolute paths or ``..`` components are skipped.
    """
    dest = data_root.cache / "extract" / item_id
    dest.mkdir(parents=True, exist_ok=True)
    names: list[str] = []
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            for info in _safe_members_zip(zf):
                if not _glob_match(info.filename, audio_glob):
                    continue
                target = dest / info.filename
                if not target.exists():
                    zf.extract(info, dest)
                names.append(info.filename)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as tf:
            for tinfo in _safe_members_tar(tf):
                if not _glob_match(tinfo.name, audio_glob):
                    continue
                target = dest / tinfo.name
                if not target.exists():
                    tf.extract(tinfo, dest, filter="data")
                names.append(tinfo.name)
    else:
        msg = f"{archive} is not a zip or tar archive"
        raise ValueError(msg)
    for name in sorted(names):
        path = dest / name
        yield ArchiveMember(path=path, member_path=name, sha256=sha256_file(path))
