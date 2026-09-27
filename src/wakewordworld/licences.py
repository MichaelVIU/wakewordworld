"""Licence tiers and the mapping from licence identifiers to tiers.

The benchmark distinguishes two tiers:

* ``A`` (redistributable): audio may be published as part of the benchmark
  dataset, with attribution and, for share-alike licences, under the same licence.
* ``B`` (internal): audio is used for evaluation by the maintainers only; the
  manifest row and all results are published, the audio is not.

Anything that is not a known, explicitly permissive licence falls into tier ``B``
(fail closed). Sources whose terms forbid data-mining or AI use are not ingested at
all and are marked ``forbidden``.
"""

from __future__ import annotations

from enum import StrEnum

__all__ = [
    "FORBIDDEN_LICENCES",
    "SHARE_ALIKE_LICENCES",
    "TIER_A_LICENCES",
    "LicenceTier",
    "is_share_alike",
    "normalise_spdx",
    "tier_for",
]


class LicenceTier(StrEnum):
    """Redistribution tier of a recording."""

    A = "A"
    B = "B"
    FORBIDDEN = "forbidden"


# SPDX identifiers (upper-cased) that permit redistribution of the audio inside a
# published benchmark. Attribution is always given; share-alike is honoured by
# publishing the derived subset under the same licence.
TIER_A_LICENCES: frozenset[str] = frozenset(
    {
        "CC0-1.0",
        "CC-PDDC",
        "PUBLIC-DOMAIN",  # explicit dedication or expired copyright, documented per item
        "CC-BY-1.0",
        "CC-BY-2.0",
        "CC-BY-2.5",
        "CC-BY-3.0",
        "CC-BY-3.0-DE",
        "CC-BY-3.0-FR",
        "CC-BY-3.0-AT",
        "CC-BY-4.0",
        "CC-BY-SA-1.0",
        "CC-BY-SA-2.0",
        "CC-BY-SA-2.5",
        "CC-BY-SA-3.0",
        "CC-BY-SA-3.0-DE",
        "CC-BY-SA-3.0-FR",
        "CC-BY-SA-3.0-AT",
        "CC-BY-SA-4.0",
        "CDLA-PERMISSIVE-1.0",
        "CDLA-PERMISSIVE-2.0",
        "APACHE-2.0",
        "MIT",
        "MIT-0",
        "BSD-2-CLAUSE",
        "BSD-3-CLAUSE",
        "LAL-1.3",  # Licence Art Libre, copyleft but permissive on redistribution
    }
)

SHARE_ALIKE_LICENCES: frozenset[str] = frozenset(
    {lic for lic in TIER_A_LICENCES if "-SA-" in lic} | {"LAL-1.3"}
)

# Identifiers we use internally for terms that exclude the material completely.
FORBIDDEN_LICENCES: frozenset[str] = frozenset({"NO-AI-USE", "NO-TDM"})

_ALIASES: dict[str, str] = {
    "CC0": "CC0-1.0",
    "CC-0": "CC0-1.0",
    "CC0 1.0": "CC0-1.0",
    "PD": "PUBLIC-DOMAIN",
    "PUBLIC DOMAIN": "PUBLIC-DOMAIN",
    "PDM": "PDM-1.0",  # Public Domain Mark is *not* a dedication -> tier B
    "PDM-1.0": "PDM-1.0",
    "CC-BY": "CC-BY-4.0-UNVERSIONED",
    "CC-BY-SA": "CC-BY-SA-4.0-UNVERSIONED",
    "CC BY 4.0": "CC-BY-4.0",
    "CC BY-SA 4.0": "CC-BY-SA-4.0",
    "CC BY 3.0": "CC-BY-3.0",
    "CC BY-SA 3.0": "CC-BY-SA-3.0",
    "CC BY-NC 4.0": "CC-BY-NC-4.0",
    "CC BY-NC-SA 4.0": "CC-BY-NC-SA-4.0",
    "CC BY-NC-ND 4.0": "CC-BY-NC-ND-4.0",
    "CC BY-ND 4.0": "CC-BY-ND-4.0",
    "CDLA-PERMISSIVE": "CDLA-PERMISSIVE-1.0",
}


def normalise_spdx(identifier: str) -> str:
    """Normalise a licence string to an upper-case SPDX-like identifier.

    Unversioned ``CC-BY`` and ``CC-BY-SA`` are kept distinct from versioned ones
    (suffix ``-UNVERSIONED``) so that a human has to confirm the version before the
    item is promoted to tier A.
    """
    key = identifier.strip().upper().replace("_", "-")
    if key in _ALIASES:
        return _ALIASES[key]
    key = key.replace(" ", "-")
    return _ALIASES.get(key, key)


def tier_for(identifier: str) -> LicenceTier:
    """Return the tier for a licence identifier (fail closed to tier B)."""
    spdx = normalise_spdx(identifier)
    if spdx in FORBIDDEN_LICENCES:
        return LicenceTier.FORBIDDEN
    if spdx in TIER_A_LICENCES:
        return LicenceTier.A
    return LicenceTier.B


def is_share_alike(identifier: str) -> bool:
    """Whether the (normalised) licence carries a share-alike obligation."""
    return normalise_spdx(identifier) in SHARE_ALIKE_LICENCES
