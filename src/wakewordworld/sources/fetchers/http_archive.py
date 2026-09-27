"""Fixed-URL archive fetcher (OpenSLR, Zenodo, university downloads)."""

from __future__ import annotations

from collections.abc import Iterator

import httpx

from wakewordworld.manifest.schema import FetchedItem
from wakewordworld.sources.base import item_id_for, registry, resolve_item_licence
from wakewordworld.sources.fetchers import spec_language
from wakewordworld.sources.spec import HttpArchiveAccess, SourceSpec

__all__ = ["HttpArchiveFetcher"]

_TYPES = {
    ".zip": "application/zip",
    ".tar": "application/x-tar",
    ".tar.gz": "application/gzip",
    ".tgz": "application/gzip",
    ".tar.xz": "application/x-xz",
    ".tar.bz2": "application/x-bzip2",
}


def _media_type(url: str) -> str | None:
    lower = url.lower().split("?")[0]
    for suffix in (".tar.gz", ".tar.xz", ".tar.bz2", ".tgz", ".zip", ".tar"):
        if lower.endswith(suffix):
            return _TYPES[suffix]
    return None


@registry.register("http_archive")
class HttpArchiveFetcher:
    """One item per configured archive URL."""

    def fetch(self, spec: SourceSpec, *, client: httpx.Client) -> Iterator[FetchedItem]:
        """Yield the configured archives; a HEAD request fills the content length."""
        access = spec.access
        assert isinstance(access, HttpArchiveAccess)
        language = spec_language(spec)
        for raw in access.urls:
            url = str(raw)
            name = url.rstrip("/").rsplit("/", 1)[-1].split("?")[0]
            content_length = ""
            try:
                head = client.head(url)
                if head.status_code < 400:
                    content_length = head.headers.get("Content-Length", "")
            except httpx.HTTPError:
                content_length = ""
            extra = {
                "audio_glob": access.audio_glob,
                "transcript_glob": access.transcript_glob or "",
                "transcript_format": access.transcript_format,
                "content_length": content_length,
            }
            if name in access.sha256:
                extra["sha256"] = access.sha256[name]
            yield FetchedItem(
                item_id=item_id_for(spec.id, url),
                source_id=spec.id,
                url=url,
                page_url=str(spec.homepage) if spec.homepage else None,
                title=name,
                language=language,
                licence=resolve_item_licence(spec, None),
                media_type=_media_type(url),
                extra=extra,
            )
