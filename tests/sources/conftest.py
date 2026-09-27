from __future__ import annotations

from typing import Any

import httpx
import pytest

from wakewordworld.sources.spec import SourceSpec

EVIDENCE = {
    "type": "page",
    "url": "https://example.org/about",
    "quote": "All shows are CC BY-SA 4.0.",
    "captured_at": "2026-09-26",
}


def make_spec(access: dict[str, Any], **overrides: Any) -> SourceSpec:
    raw: dict[str, Any] = {
        "id": "example",
        "name": "Example",
        "languages": ["de"],
        "domain": "podcast",
        "access": access,
        "licence": {"spdx": "CC-BY-SA-4.0", "evidence": EVIDENCE},
        "filters": {"min_duration_s": 0},
    }
    raw.update(overrides)
    return SourceSpec.model_validate(raw)


@pytest.fixture
def client() -> httpx.Client:
    with httpx.Client(follow_redirects=True) as c:
        yield c
