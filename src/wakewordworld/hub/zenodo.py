"""Minimal Zenodo deposition client (REST API v1).

Only what a release needs: create a deposition, upload files, publish, and open a new
version of an existing concept. Tokens come from ``ZENODO_TOKEN`` and are never logged.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import yaml

from wakewordworld.manifest.schema import ManifestRelease

__all__ = ["ZenodoClient", "ZenodoError", "deposition_metadata"]

PRODUCTION = "https://zenodo.org"
SANDBOX = "https://sandbox.zenodo.org"


class ZenodoError(RuntimeError):
    """Raised on a failed Zenodo API call (message never includes the token)."""


def deposition_metadata(
    release: ManifestRelease,
    *,
    citation_cff: Path | None,
    github_tag_url: str | None,
    hf_dataset_url: str | None,
) -> dict[str, Any]:
    """Zenodo metadata block from the release header and ``CITATION.cff``."""
    creators: list[dict[str, str]] = [{"name": "WakeWordWorld maintainers"}]
    title = "WakeWordWorld benchmark"
    if citation_cff is not None and citation_cff.exists():
        cff = yaml.safe_load(citation_cff.read_text(encoding="utf-8")) or {}
        creators = [
            {
                "name": a.get("name")
                or f"{a.get('family-names', '')}, {a.get('given-names', '')}".strip(", ")
            }
            for a in cff.get("authors", [])
        ] or creators
        title = str(cff.get("title", title))
    related = []
    if github_tag_url:
        related.append(
            {"identifier": github_tag_url, "relation": "isSupplementTo", "scheme": "url"}
        )
    if hf_dataset_url:
        related.append({"identifier": hf_dataset_url, "relation": "isIdenticalTo", "scheme": "url"})
    hours = ", ".join(f"{k}: {v:.1f} h" for k, v in sorted(release.hours_by_language.items()))
    description = (
        f"<p>Frozen manifest of the WakeWordWorld wake word benchmark, version {release.version} "
        f"({release.n_chunks} chunks from {release.n_files} recordings; {hours}). "
        "Audio is licensed per recording (CC0, CC BY, CC BY-SA); see the attribution table. "
        "This record holds the manifest, checksums and dataset card; audio is distributed "
        "through the gated Hugging Face dataset linked below.</p>"
    )
    return {
        "metadata": {
            "title": f"{title} (dataset release {release.version})",
            "upload_type": "dataset",
            "description": description,
            "creators": creators,
            "version": release.version,
            "license": "other-open",
            "access_right": "open",
            "keywords": ["wake word", "keyword spotting", "benchmark", "speech", "evaluation"],
            "notes": "Per-recording licences are listed in sources.md and the metadata table; "
            "share-alike material is published under the same licence.",
            "related_identifiers": related,
        }
    }


@dataclass
class ZenodoClient:
    """Thin client; ``base_url`` selects production or sandbox."""

    token: str
    base_url: str = PRODUCTION
    timeout: float = 120.0

    @classmethod
    def from_env(cls, *, sandbox: bool = False) -> ZenodoClient:
        """Build from ``ZENODO_TOKEN``."""
        token = os.environ.get("ZENODO_TOKEN", "")
        if not token:
            msg = "ZENODO_TOKEN is not set"
            raise ZenodoError(msg)
        return cls(token=token, base_url=SANDBOX if sandbox else PRODUCTION)

    def _client(self) -> httpx.Client:
        return httpx.Client(
            base_url=self.base_url,
            timeout=self.timeout,
            headers={"Authorization": f"Bearer {self.token}"},
        )

    @staticmethod
    def _check(resp: httpx.Response, what: str) -> dict[str, Any]:
        if resp.status_code >= 400:
            msg = f"zenodo {what} failed: HTTP {resp.status_code} {resp.text[:300]}"
            raise ZenodoError(msg)
        data: dict[str, Any] = resp.json() if resp.content else {}
        return data

    def create_deposition(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """Create a draft deposition with metadata; returns the deposition JSON."""
        with self._client() as c:
            return self._check(c.post("/api/deposit/depositions", json=metadata), "create")

    def upload_file(self, deposition: dict[str, Any], path: Path) -> dict[str, Any]:
        """Upload one file through the bucket API."""
        bucket = deposition.get("links", {}).get("bucket")
        if not bucket:
            msg = "deposition has no bucket link"
            raise ZenodoError(msg)
        with self._client() as c, path.open("rb") as fh:
            return self._check(
                c.put(f"{bucket}/{path.name}", content=fh.read()), f"upload {path.name}"
            )

    def publish(self, deposition_id: int) -> dict[str, Any]:
        """Publish a draft; this mints the DOI and cannot be undone."""
        with self._client() as c:
            return self._check(
                c.post(f"/api/deposit/depositions/{deposition_id}/actions/publish"), "publish"
            )

    def new_version(self, deposition_id: int) -> dict[str, Any]:
        """Open a new version draft of a published deposition; returns the draft."""
        with self._client() as c:
            created = self._check(
                c.post(f"/api/deposit/depositions/{deposition_id}/actions/newversion"), "newversion"
            )
            latest = created.get("links", {}).get("latest_draft")
            if not latest:
                return created
            return self._check(c.get(latest), "fetch draft")
