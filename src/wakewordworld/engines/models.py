"""Model file handling shared by adapters: cache location, download, checksums."""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

import httpx

from wakewordworld.util.hashing import sha256_file
from wakewordworld.util.paths import DataRoot

__all__ = ["ModelChecksumError", "download_file", "hash_models", "models_dir", "unzip_model"]


class ModelChecksumError(RuntimeError):
    """Raised when a downloaded model does not match its expected checksum."""


def models_dir(engine_id: str, data_root: DataRoot | None = None) -> Path:
    """Directory where an engine's models are cached (``<data>/cache/models/<engine>``)."""
    root = DataRoot.resolve() if data_root is None else data_root
    path = root.cache / "models" / engine_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def download_file(
    url: str,
    dest: Path,
    *,
    expected_sha256: str | None = None,
    client: httpx.Client | None = None,
    timeout: float = 600.0,
) -> str:
    """Download ``url`` to ``dest`` (streaming) and return its SHA-256.

    Existing files are reused when they match ``expected_sha256`` (or when no checksum
    is known). A mismatch after download raises :class:`ModelChecksumError` and removes
    the file.
    """
    if dest.exists():
        digest = sha256_file(dest)
        if expected_sha256 is None or digest == expected_sha256:
            return digest
        dest.unlink()
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    own_client = client is None
    cl = client or httpx.Client(follow_redirects=True, timeout=timeout)
    try:
        with cl.stream("GET", url) as resp, tmp.open("wb") as fh:
            resp.raise_for_status()
            for block in resp.iter_bytes(1 << 20):
                fh.write(block)
    finally:
        if own_client:
            cl.close()
    digest = sha256_file(tmp)
    if expected_sha256 is not None and digest != expected_sha256:
        tmp.unlink(missing_ok=True)
        msg = f"{url}: sha256 {digest} does not match expected {expected_sha256}"
        raise ModelChecksumError(msg)
    tmp.replace(dest)
    return digest


def unzip_model(archive: Path, target_dir: Path) -> Path:
    """Extract a zip archive into ``target_dir`` (idempotent) and return the top folder."""
    with zipfile.ZipFile(archive) as zf:
        names = [n for n in zf.namelist() if n and not n.startswith("__MACOSX")]
        top = names[0].split("/")[0] if names else archive.stem
        out = target_dir / top
        if out.exists() and any(out.iterdir()):
            return out
        tmp = target_dir / f".{top}.extracting"
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        zf.extractall(tmp)
        (tmp / top).replace(out) if (tmp / top).exists() else tmp.replace(out)
        shutil.rmtree(tmp, ignore_errors=True)
    return out


def hash_models(paths: dict[str, Path]) -> dict[str, str]:
    """SHA-256 per named model file (directories are hashed via their sorted files)."""
    out: dict[str, str] = {}
    for name, p in paths.items():
        if p.is_dir():
            import hashlib

            digest = hashlib.sha256()
            for f in sorted(x for x in p.rglob("*") if x.is_file()):
                digest.update(f.relative_to(p).as_posix().encode())
                digest.update(bytes.fromhex(sha256_file(f)))
            out[name] = digest.hexdigest()
        elif p.is_file():
            out[name] = sha256_file(p)
    return out
