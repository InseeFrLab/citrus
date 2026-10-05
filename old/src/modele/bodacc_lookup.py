"""Shared exact BODACC-id resolution for real-data benchmark runners."""

from __future__ import annotations

from collections.abc import Mapping
import re
from typing import Any

from src.modele.benchmark import JOIN_KEY


_REFERENCE_SUFFIX = re.compile(r"(?P<publication>[ABC])(?P<parution>\d{8})$")
_DIRECT_BODACC_ID = re.compile(r"^[ABC]\d{9,}$")


class BodaccLookupResolutionError(ValueError):
    """Raised when annotation references cannot identify one exact API id."""


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BodaccLookupResolutionError(f"{field} must be a non-empty string")
    return value.strip()


def _announcement_number(value: Any) -> str:
    if isinstance(value, bool) or value is None:
        raise BodaccLookupResolutionError(
            "numero_annonce must be a positive integer"
        )
    if isinstance(value, int):
        number = value
    elif isinstance(value, float) and value.is_integer():
        number = int(value)
    elif isinstance(value, str) and value.strip().isdigit():
        number = int(value.strip())
    else:
        raise BodaccLookupResolutionError(
            "numero_annonce must be a positive integer"
        )
    if number <= 0:
        raise BodaccLookupResolutionError(
            "numero_annonce must be a positive integer"
        )
    return str(number)


def resolve_bodacc_announcement_id(annotation: Mapping[str, Any]) -> str:
    """Validate and return the direct OpenData id from annotation references.

    Current references follow this exact relationship:
    ``ref_annonce_complet == <A|B|C><YYYY><parution><numero_annonce>``.
    The ``<A|B|C><YYYY><parution>`` prefix is the final nine characters of
    ``ref_annonce``. No identifier is synthesized when any part is ambiguous.
    """

    complete = _required_text(annotation.get(JOIN_KEY), JOIN_KEY)
    reference = _required_text(annotation.get("ref_annonce"), "ref_annonce")
    number = _announcement_number(annotation.get("numero_annonce"))
    suffix = _REFERENCE_SUFFIX.search(reference)
    if suffix is None:
        raise BodaccLookupResolutionError(
            "ref_annonce has no supported A/B/C + 8-digit publication suffix"
        )
    expected = f"{suffix.group(0)}{number}"
    if not _DIRECT_BODACC_ID.fullmatch(complete):
        raise BodaccLookupResolutionError(
            "ref_annonce_complet is not a directly usable BODACC OpenData id"
        )
    if complete != expected:
        raise BodaccLookupResolutionError(
            f"inconsistent annotation references: expected {expected}"
        )
    return complete


__all__ = (
    "BodaccLookupResolutionError",
    "resolve_bodacc_announcement_id",
)
