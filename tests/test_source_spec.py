from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
import yaml

from wakewordworld.licences import LicenceTier
from wakewordworld.sources.spec import SourceSpec, load_source_spec, load_source_specs

BASE = {
    "id": "example",
    "name": "Example feed",
    "languages": ["de"],
    "domain": "podcast",
    "access": {"type": "rss", "url": "https://example.org/feed.xml"},
    "licence": {
        "spdx": "CC-BY-SA-4.0",
        "evidence": {
            "type": "feed_tag",
            "url": "https://example.org/feed.xml",
            "quote": "<podcast:license>CC-BY-SA-4.0</podcast:license>",
            "captured_at": "2026-09-26",
        },
    },
}


def test_valid_spec_is_tier_a() -> None:
    spec = SourceSpec.model_validate(BASE)
    assert spec.tier is LicenceTier.A
    assert spec.licence.evidence is not None
    assert spec.licence.evidence.captured_at == date(2026, 9, 26)


def test_tier_a_without_evidence_needs_note() -> None:
    raw = {**BASE, "licence": {"spdx": "CC-BY-4.0"}}
    with pytest.raises(ValueError, match="without evidence"):
        SourceSpec.model_validate(raw)
    raw["notes"] = "licence page unreachable on 2026-09-26; keep tier B until confirmed"
    spec = SourceSpec.model_validate(raw)
    assert spec.tier is LicenceTier.B


def test_tier_override_only_downgrades() -> None:
    raw = {**BASE, "licence": {**BASE["licence"], "tier_override": "B"}}
    assert SourceSpec.model_validate(raw).tier is LicenceTier.B
    raw = {
        **BASE,
        "licence": {"spdx": "CC-BY-NC-4.0", "tier_override": "A"},
    }
    assert SourceSpec.model_validate(raw).tier is LicenceTier.B


def test_bad_id_rejected() -> None:
    with pytest.raises(ValueError, match="must match"):
        SourceSpec.model_validate({**BASE, "id": "Bad Id"})


def test_unknown_field_rejected() -> None:
    with pytest.raises(ValueError, match="extra"):
        SourceSpec.model_validate({**BASE, "surprise": 1})


def test_access_discriminator() -> None:
    raw = {
        **BASE,
        "access": {"type": "peertube", "base_url": "https://videos.example.org"},
    }
    spec = SourceSpec.model_validate(raw)
    assert spec.access.type == "peertube"
    assert spec.access.licence_ids == [1, 2, 7]


def test_file_stem_must_match_id(tmp_path: Path) -> None:
    p = tmp_path / "other.yaml"
    p.write_text(yaml.safe_dump(BASE), encoding="utf-8")
    with pytest.raises(ValueError, match="file stem"):
        load_source_spec(p)


def test_repo_sources_all_valid(sources_dir: Path) -> None:
    specs = load_source_specs(sources_dir)
    assert specs, "the repository must ship at least one source spec"
    assert len({s.id for s in specs}) == len(specs)
    for s in specs:
        if s.tier is LicenceTier.A:
            assert s.licence.evidence is not None, s.id
