"""Item store: JSONL files with the fetched items of each source."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

from wakewordworld.manifest.schema import FetchedItem
from wakewordworld.util.paths import DataRoot

__all__ = ["ItemStore"]


class ItemStore:
    """Append-only, id-deduplicated JSONL store per source."""

    def __init__(self, data_root: DataRoot) -> None:
        self._root = data_root

    def path(self, source_id: str) -> Path:
        """File holding the items of a source."""
        return self._root.items / f"{source_id}.jsonl"

    def read(self, source_id: str) -> Iterator[FetchedItem]:
        """Yield stored items (empty if none)."""
        p = self.path(source_id)
        if not p.exists():
            return
        with p.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    yield FetchedItem.model_validate_json(line)

    def write(self, source_id: str, items: Iterable[FetchedItem], *, replace: bool = False) -> int:
        """Merge items into the store by ``item_id``; returns the number written."""
        existing: dict[str, FetchedItem] = {}
        if not replace:
            existing = {it.item_id: it for it in self.read(source_id)}
        for it in items:
            existing[it.item_id] = it
        p = self.path(source_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".jsonl.tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            for it in sorted(existing.values(), key=lambda x: x.item_id):
                fh.write(it.model_dump_json())
                fh.write("\n")
        tmp.replace(p)
        return len(existing)
