"""Deterministic location-gerance extraction from normalized BODACC facts."""

from __future__ import annotations

from datetime import datetime
import re
from typing import Any

from src.bodacc import (
    BodaccDialect,
    NormalizedBodaccAnnouncement,
    normalize_bodacc_announcement,
)
from src.operation.base import OperationResult


_DATE_FORMATS = ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y")
_AFTER_EFFECTIVE_FROM = re.compile(
    r"\b(?:a|à)\s+compter\s+du\s*(?:[:\-]\s*)?"
    r"(?P<date>\d{4}-\d{2}-\d{2}|\d{2}[/-]\d{2}[/-]\d{4})",
    re.IGNORECASE,
)


def _iso_date(value: str | None) -> str | None:
    """Normalize documented BODACC date forms to an ISO calendar date."""

    if value is None:
        return None
    text = value.strip()
    for date_format in _DATE_FORMATS:
        try:
            return datetime.strptime(text, date_format).date().isoformat()
        except ValueError:
            pass
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date().isoformat()
    except ValueError:
        return None


def _rcs_b_description_date(description: str | None) -> str | None:
    if description is None:
        return None
    match = _AFTER_EFFECTIVE_FROM.search(description)
    return _iso_date(match.group("date")) if match else None


def _campaign_year(publication_date: str | None) -> int | None:
    normalized = _iso_date(publication_date)
    if normalized is not None:
        return int(normalized[:4])
    if publication_date is None:
        return None
    year = re.fullmatch(r"\s*(?P<year>\d{4})\s*", publication_date)
    return int(year.group("year")) if year else None


def _accounting_effect_date(
    announcement: NormalizedBodaccAnnouncement,
) -> str | None:
    candidates = [
        announcement.commencement_date,
        announcement.effect_date,
    ]
    if announcement.dialect is BodaccDialect.RCS_B:
        candidates.append(
            _rcs_b_description_date(announcement.modification_description)
        )
    candidates.append(announcement.publication_date)
    return next(
        (normalized for value in candidates if (normalized := _iso_date(value))),
        None,
    )


class LocationGeranceSkill:
    """Announcement-level LG extraction using normalized source facts only."""

    operation_type = "LG"

    def extract(self, announcement: dict[str, Any]) -> OperationResult:
        normalized = normalize_bodacc_announcement(announcement)
        previous_operator = next(
            (
                party
                for party in normalized.previous_operators
                if party.siren is not None
            ),
            None,
        )
        legal_date = (
            _iso_date(normalized.immatriculation_date)
            if normalized.dialect is BodaccDialect.RCS_A
            else None
        )

        return {
            "anneeCampagne": _campaign_year(normalized.publication_date),
            "typeOperation": self.operation_type,
            "sirenCedant": (
                previous_operator.siren if previous_operator is not None else None
            ),
            "raisonSocialeCedant": (
                previous_operator.name if previous_operator is not None else None
            ),
            "sirenBeneficiaire": normalized.main_siren,
            "raisonSocialeBeneficiaire": normalized.main_name,
            "dateEffetComptable": _accounting_effect_date(normalized),
            "dateRealisationJuridique": legal_date,
            "montantNet": None,
            "source": normalized.source_url,
        }


location_gerance_skill = LocationGeranceSkill()


__all__ = ("LocationGeranceSkill", "location_gerance_skill")
