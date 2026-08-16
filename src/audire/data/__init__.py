"""External data acquisition, provenance manifests and stimulus catalogues."""

from audire.data.manifest import FileRecord, Manifest, sha256_file
from audire.data.sources import (
    AcknowledgementRequired,
    LiteratureRef,
    Source,
    SourceRegistry,
    SourceUseViolation,
    load_registry,
    registry,
)

__all__ = [
    "AcknowledgementRequired",
    "FileRecord",
    "LiteratureRef",
    "Manifest",
    "Source",
    "SourceRegistry",
    "SourceUseViolation",
    "load_registry",
    "registry",
    "sha256_file",
]
