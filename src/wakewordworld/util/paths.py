"""Data directory layout.

All large artefacts live under one data root (default ``./data``, override with the
``WWW_DATA_ROOT`` environment variable or ``--data-root``). Nothing under the data
root is ever committed to git.

Layout::

    <root>/
      originals/<sha256[:2]>/<sha256>.<ext>      content-addressed downloads
      audio/<source_id>/<file_id>.flac           16 kHz mono working copies
      chunks/<source_id>/<chunk_id>.flac         evaluation chunks
      transcripts/<source_id>/<file_id>.json     word-level transcripts
      index/<source_id>.parquet                  word index per source
      items/<source_id>.jsonl                    fetched item lists with licence evidence
      scores/<engine_id>/<chunk_id>.parquet      raw per-frame scores
      cache/                                     downloaded models, temp files
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

__all__ = ["DataRoot", "repo_root"]


def repo_root() -> Path:
    """Best-effort repository root (directory containing ``pyproject.toml``)."""
    here = Path.cwd().resolve()
    for candidate in (here, *here.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "sources").is_dir():
            return candidate
    return here


@dataclass(frozen=True)
class DataRoot:
    """Resolved data directory."""

    root: Path

    @classmethod
    def resolve(cls, override: Path | None = None) -> DataRoot:
        """Resolve from an explicit path, the environment, or ``./data``."""
        if override is not None:
            return cls(override.expanduser().resolve())
        env = os.environ.get("WWW_DATA_ROOT")
        if env:
            return cls(Path(env).expanduser().resolve())
        return cls((repo_root() / "data").resolve())

    @property
    def originals(self) -> Path:
        """Content-addressed originals."""
        return self.root / "originals"

    @property
    def audio(self) -> Path:
        """Normalised working copies."""
        return self.root / "audio"

    @property
    def chunks(self) -> Path:
        """Evaluation chunks."""
        return self.root / "chunks"

    @property
    def transcripts(self) -> Path:
        """Word-level transcripts."""
        return self.root / "transcripts"

    @property
    def index(self) -> Path:
        """Word index parquet files."""
        return self.root / "index"

    @property
    def items(self) -> Path:
        """Fetched item lists."""
        return self.root / "items"

    @property
    def scores(self) -> Path:
        """Raw engine scores."""
        return self.root / "scores"

    @property
    def cache(self) -> Path:
        """Model and download cache."""
        return self.root / "cache"

    def original_path(self, sha256: str, ext: str) -> Path:
        """Path of a content-addressed original."""
        ext = ext.lstrip(".").lower()
        return self.originals / sha256[:2] / f"{sha256}.{ext}"

    def ensure(self) -> None:
        """Create all directories."""
        for p in (
            self.originals,
            self.audio,
            self.chunks,
            self.transcripts,
            self.index,
            self.items,
            self.scores,
            self.cache,
        ):
            p.mkdir(parents=True, exist_ok=True)
