from __future__ import annotations

import pytest

from wakewordworld.licences import LicenceTier, is_share_alike, normalise_spdx, tier_for


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("CC-BY-SA-4.0", "CC-BY-SA-4.0"),
        ("cc-by-sa-4.0", "CC-BY-SA-4.0"),
        ("CC BY-SA 4.0", "CC-BY-SA-4.0"),
        ("CC0", "CC0-1.0"),
        ("cc0-1.0", "CC0-1.0"),
        ("PDM", "PDM-1.0"),
        ("CC-BY", "CC-BY-4.0-UNVERSIONED"),
        ("CC BY-NC-ND 4.0", "CC-BY-NC-ND-4.0"),
    ],
)
def test_normalise_spdx(raw: str, expected: str) -> None:
    assert normalise_spdx(raw) == expected


@pytest.mark.parametrize(
    ("raw", "tier"),
    [
        ("CC0-1.0", LicenceTier.A),
        ("CC-BY-4.0", LicenceTier.A),
        ("CC-BY-SA-3.0-DE", LicenceTier.A),
        ("CDLA-Permissive-1.0", LicenceTier.A),
        ("CC-BY-NC-4.0", LicenceTier.B),
        ("CC-BY-NC-SA-3.0-DE", LicenceTier.B),
        ("CC-BY-ND-4.0", LicenceTier.B),
        ("PDM-1.0", LicenceTier.B),
        ("CC-BY", LicenceTier.B),  # unversioned -> needs human confirmation
        ("all-rights-reserved", LicenceTier.B),
        ("", LicenceTier.B),
        ("NO-AI-USE", LicenceTier.FORBIDDEN),
    ],
)
def test_tier_for_fails_closed(raw: str, tier: LicenceTier) -> None:
    assert tier_for(raw) is tier


def test_share_alike() -> None:
    assert is_share_alike("CC-BY-SA-4.0")
    assert is_share_alike("LAL-1.3")
    assert not is_share_alike("CC-BY-4.0")
    assert not is_share_alike("CC0-1.0")
