"""BODACC API and source normalization boundaries."""

from src.bodacc.api import BodaccFetchError
from src.bodacc.normalization import (
    BodaccDialect,
    BodaccNormalizationError,
    NormalizedBodaccAnnouncement,
    NormalizedParty,
    extract_siren_candidates,
    normalize_bodacc_announcement,
)

__all__ = (
    "BodaccDialect",
    "BodaccFetchError",
    "BodaccNormalizationError",
    "NormalizedBodaccAnnouncement",
    "NormalizedParty",
    "extract_siren_candidates",
    "normalize_bodacc_announcement",
)
