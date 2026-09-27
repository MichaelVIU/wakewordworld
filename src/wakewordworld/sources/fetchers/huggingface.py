"""Hugging Face dataset repository fetcher (Hub HTTP API, no SDK dependency)."""

from __future__ import annotations

import os
import re
from collections.abc import Iterator
from typing import Any

import httpx

from wakewordworld.manifest.schema import FetchedItem
from wakewordworld.sources.base import item_id_for, passes_filters, registry, resolve_item_licence
from wakewordworld.sources.fetchers import FetchError
from wakewordworld.sources.spec import HuggingFaceAccess, Language, SourceSpec

__all__ = ["HuggingFaceFetcher", "infer_language"]

HUB = "https://huggingface.co"
DATA_SUFFIXES = {
    ".parquet": "application/x-parquet",
    ".tar": "application/x-tar",
    ".tar.gz": "application/gzip",
    ".tgz": "application/gzip",
    ".zip": "application/zip",
}
_LANG_CODES = {lang.value for lang in Language}


def infer_language(path: str) -> Language | None:
    """Infer a benchmark language from path segments like ``/de/`` or ``de_train``."""
    for seg in re.split(r"[/_.\-]", path.lower()):
        if seg in _LANG_CODES:
            return Language(seg)
    return None


def _media_type(path: str) -> str | None:
    lower = path.lower()
    for suffix in (".tar.gz", ".tgz", ".parquet", ".tar", ".zip"):
        if lower.endswith(suffix):
            return DATA_SUFFIXES[suffix]
    return None


def _wanted(path: str, config: str | None, split: str | None) -> bool:
    segments = path.lower().split("/")
    name = segments[-1]
    if config and config.lower() not in segments and not name.startswith(config.lower()):
        return False
    return not (split and split.lower() not in segments and split.lower() not in name)


@registry.register("huggingface")
class HuggingFaceFetcher:
    """List data files of a dataset repository."""

    def _headers(self) -> dict[str, str]:
        token = os.environ.get("HF_TOKEN")
        return {"Authorization": f"Bearer {token}"} if token else {}

    def _tree(
        self, client: httpx.Client, repo_id: str, rev: str, path: str
    ) -> Iterator[dict[str, Any]]:
        url = f"{HUB}/api/datasets/{repo_id}/tree/{rev}/{path}".rstrip("/")
        cursor: str | None = None
        while True:
            params = {"cursor": cursor} if cursor else {}
            try:
                resp = client.get(url, params=params, headers=self._headers())
                resp.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise FetchError(url, f"HTTP {exc.response.status_code}") from exc
            except httpx.HTTPError as exc:
                raise FetchError(url, f"{type(exc).__name__}: {exc}") from exc
            entries = resp.json()
            if not isinstance(entries, list):
                raise FetchError(url, "tree response is not a list")
            for entry in entries:
                if entry.get("type") == "directory":
                    yield from self._tree(client, repo_id, rev, entry["path"])
                else:
                    yield entry
            link = resp.headers.get("Link", "")
            m = re.search(r"[?&]cursor=([^&>]+)", link)
            if not m:
                break
            cursor = m.group(1)

    def fetch(self, spec: SourceSpec, *, client: httpx.Client) -> Iterator[FetchedItem]:
        """Yield one item per Parquet or archive data file."""
        access = spec.access
        assert isinstance(access, HuggingFaceAccess)
        rev = access.revision or "main"
        single = spec.languages[0] if len(spec.languages) == 1 else None
        for entry in self._tree(client, access.repo_id, rev, ""):
            path = str(entry.get("path", ""))
            media_type = _media_type(path)
            if media_type is None or not _wanted(path, access.config, access.split):
                continue
            if not passes_filters(spec.filters, title=path, duration_s=None, published=None):
                continue
            language = single or infer_language(path)
            if single is None and language is not None and language not in spec.languages:
                language = None
            url = f"{HUB}/datasets/{access.repo_id}/resolve/{rev}/{path}"
            yield FetchedItem(
                item_id=item_id_for(spec.id, url),
                source_id=spec.id,
                url=url,
                page_url=f"{HUB}/datasets/{access.repo_id}",
                title=path,
                author=access.repo_id.split("/")[0],
                language=language,
                licence=resolve_item_licence(spec, None),
                media_type=media_type,
                extra={
                    "audio_column": access.audio_column,
                    "text_column": access.text_column or "",
                    "config": access.config or "",
                    "split": access.split or "",
                    "size": str(entry.get("size", "")),
                    "revision": rev,
                },
            )
