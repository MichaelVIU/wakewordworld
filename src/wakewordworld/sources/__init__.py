"""Source registry: declarative descriptions of where benchmark audio comes from."""

from wakewordworld.sources.spec import (
    AccessSpec,
    CccAccess,
    CommonsAccess,
    HttpArchiveAccess,
    HuggingFaceAccess,
    InternetArchiveAccess,
    LicenceEvidence,
    LicenceSpec,
    PeerTubeAccess,
    RssAccess,
    SourceFilters,
    SourceSpec,
    load_source_specs,
)

__all__ = [
    "AccessSpec",
    "CccAccess",
    "CommonsAccess",
    "HttpArchiveAccess",
    "HuggingFaceAccess",
    "InternetArchiveAccess",
    "LicenceEvidence",
    "LicenceSpec",
    "PeerTubeAccess",
    "RssAccess",
    "SourceFilters",
    "SourceSpec",
    "load_source_specs",
]
