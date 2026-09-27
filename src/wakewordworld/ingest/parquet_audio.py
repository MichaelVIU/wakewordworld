"""Expansion of Parquet files with an embedded audio column (Hugging Face layout)."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

from wakewordworld.util.hashing import sha256_bytes
from wakewordworld.util.paths import DataRoot

__all__ = ["ParquetRow", "expand_parquet"]

_MAGIC: list[tuple[bytes, str]] = [
    (b"fLaC", "flac"),
    (b"OggS", "ogg"),
    (b"RIFF", "wav"),
    (b"ID3", "mp3"),
    (b"\xff\xfb", "mp3"),
    (b"\xff\xf3", "mp3"),
    (b"\xff\xf2", "mp3"),
]


@dataclass(frozen=True)
class ParquetRow:
    """One audio row written to a temporary file."""

    path: Path
    row_index: int
    sha256: str
    text: str | None

    @property
    def member_path(self) -> str:
        """Marker stored in the file record."""
        return f"row:{self.row_index}"


def _sniff_ext(data: bytes, hint: str | None) -> str:
    if hint:
        suffix = Path(hint).suffix.lstrip(".").lower()
        if suffix:
            return suffix
    for magic, ext in _MAGIC:
        if data.startswith(magic):
            return ext
    return "bin"


def expand_parquet(
    parquet: Path,
    *,
    item_id: str,
    source_id: str,
    data_root: DataRoot,
    audio_column: str = "audio",
    text_column: str | None = None,
    max_rows: int | None = 500,
    batch_rows: int = 64,
) -> Iterator[ParquetRow]:
    """Write each row's audio bytes to ``cache/parquet/<item_id>/row-<n>.<ext>``.

    Reference text (when ``text_column`` is given) is written to
    ``cache/reftext/<source_id>/<sha>.txt`` for the transcription stage.
    """
    dest = data_root.cache / "parquet" / item_id
    dest.mkdir(parents=True, exist_ok=True)
    ref_dir = data_root.cache / "reftext" / source_id
    pf = pq.ParquetFile(parquet)
    columns = [audio_column] + ([text_column] if text_column else [])
    row_index = 0
    for batch in pf.iter_batches(batch_size=batch_rows, columns=columns):
        audio_col = batch.column(audio_column).to_pylist()
        text_col = batch.column(text_column).to_pylist() if text_column else [None] * len(audio_col)
        for value, text in zip(audio_col, text_col, strict=True):
            if max_rows is not None and row_index >= max_rows:
                return
            idx = row_index
            row_index += 1
            raw: object
            hint: str | None
            if isinstance(value, dict):
                raw = value.get("bytes")
                hint = value.get("path")
            else:
                raw = value
                hint = None
            if not isinstance(raw, bytes | bytearray) or not raw:
                continue
            data = bytes(raw)
            ext = _sniff_ext(data, hint)
            path = dest / f"row-{idx}.{ext}"
            if not path.exists():
                tmp = path.with_suffix(path.suffix + ".tmp")
                tmp.write_bytes(data)
                tmp.replace(path)
            sha = sha256_bytes(data)
            if text is not None:
                ref_dir.mkdir(parents=True, exist_ok=True)
                (ref_dir / f"{sha}.txt").write_text(str(text), encoding="utf-8")
            yield ParquetRow(path=path, row_index=idx, sha256=sha, text=str(text) if text else None)
