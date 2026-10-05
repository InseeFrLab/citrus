'''Deterministic TP/TUP extraction from normalized RCS-B facts.'''

from __future__ import annotations

from datetime import date, datetime
import re
from typing import Any
import unicodedata

from src.bodacc import (
    BodaccDialect,
    extract_siren_candidates,
    normalize_bodacc_announcement,
)
from src.operation.base import OperationResult


_MONTHS = {
    'janvier': 1,
    'fevrier': 2,
    'mars': 3,
    'avril': 4,
    'mai': 5,
    'juin': 6,
    'juillet': 7,
    'aout': 8,
    'septembre': 9,
    'octobre': 10,
    'novembre': 11,
    'decembre': 12,
}
_TEXTUAL_MONTH_PATTERN = (
    r'janvier|f[eé]vrier|mars|avril|mai|juin|juillet|ao[uû]t|'
    r'septembre|octobre|novembre|d[eé]cembre'
)
_DESCRIPTION_DATE = re.compile(
    rf'(?<!\d)(?:'
    rf'(?P<iso>\d{{4}}-\d{{2}}-\d{{2}})'
    rf'|(?P<numeric>\d{{1,2}}[/-]\d{{1,2}}[/-]\d{{4}})'
    rf'|(?P<textual>\d{{1,2}}(?:er)?\s+(?:{_TEXTUAL_MONTH_PATTERN})\s+\d{{4}})'
    rf')(?!\d)',
    re.IGNORECASE,
)
_TP_WORDING = re.compile(
    r'\btransmiss(?:ion)?[\s.\-]*univers(?:elle)?[\s.\-]*'
    r'(?:(?:du|de)[\s.\-]*)?patrimoine\b',
    re.IGNORECASE,
)
_PUBLICATION_DATE_FORMATS = ('%Y-%m-%d', '%d/%m/%Y', '%d-%m-%Y')


def _accentless(value: str) -> str:
    return ''.join(
        character
        for character in unicodedata.normalize('NFKD', value)
        if not unicodedata.combining(character)
    )


def contains_tp_wording(text: str | None) -> bool:
    '''Return whether text contains a documented TP/TUP wording variant.'''

    return bool(text and _TP_WORDING.search(_accentless(text)))


def _parse_description_date(match: re.Match[str]) -> str | None:
    if match.group('iso') is not None:
        try:
            return date.fromisoformat(match.group('iso')).isoformat()
        except ValueError:
            return None

    numeric = match.group('numeric')
    if numeric is not None:
        for date_format in ('%d/%m/%Y', '%d-%m-%Y'):
            try:
                return datetime.strptime(numeric, date_format).date().isoformat()
            except ValueError:
                pass
        return None

    textual = _accentless(match.group('textual')).lower()
    parts = textual.split()
    try:
        day = int(parts[0].removesuffix('er'))
        month = _MONTHS[parts[1]]
        year = int(parts[2])
        return date(year, month, day).isoformat()
    except (IndexError, KeyError, ValueError):
        return None


def extract_description_dates(description: str | None) -> tuple[str, ...]:
    '''Extract valid dates while preserving their left-to-right text order.'''

    if description is None:
        return ()
    if not isinstance(description, str):
        raise TypeError('description must be a string or None')
    dates: list[str] = []
    for match in _DESCRIPTION_DATE.finditer(description):
        parsed = _parse_description_date(match)
        if parsed is not None:
            dates.append(parsed)
    return tuple(dates)


def _publication_date(value: str | None) -> str | None:
    if value is None:
        return None
    text = value.strip()
    for date_format in _PUBLICATION_DATE_FORMATS:
        try:
            return datetime.strptime(text, date_format).date().isoformat()
        except ValueError:
            pass
    try:
        return datetime.fromisoformat(text.replace('Z', '+00:00')).date().isoformat()
    except ValueError:
        return None


def _campaign_year(publication_date: str | None) -> int | None:
    normalized = _publication_date(publication_date)
    if normalized is not None:
        return int(normalized[:4])
    if publication_date is None:
        return None
    year = re.fullmatch(r'\s*(?P<year>\d{4})\s*', publication_date)
    return int(year.group('year')) if year else None


class TransmissionPatrimoineSkill:
    '''Announcement-level TP extraction using normalized RCS-B facts only.'''

    operation_type = 'TP'

    def extract(self, announcement: dict[str, Any]) -> OperationResult:
        normalized = normalize_bodacc_announcement(announcement)
        is_rcs_b = normalized.dialect is BodaccDialect.RCS_B
        description = normalized.modification_description if is_rcs_b else None
        dates = extract_description_dates(description)
        beneficiaries = extract_siren_candidates(
            description,
            excluded_sirens=(normalized.main_siren,),
        )

        return {
            'anneeCampagne': _campaign_year(normalized.publication_date),
            'typeOperation': self.operation_type,
            'sirenCedant': normalized.main_siren if is_rcs_b else None,
            'raisonSocialeCedant': normalized.main_name if is_rcs_b else None,
            'sirenBeneficiaire': beneficiaries[0] if beneficiaries else None,
            'raisonSocialeBeneficiaire': None,
            'dateEffetComptable': (
                dates[0]
                if dates
                else _publication_date(normalized.publication_date)
                if is_rcs_b
                else None
            ),
            'dateRealisationJuridique': dates[-1] if dates else None,
            'montantNet': None,
            'source': normalized.source_url,
        }


transmission_patrimoine_skill = TransmissionPatrimoineSkill()


__all__ = (
    'TransmissionPatrimoineSkill',
    'contains_tp_wording',
    'extract_description_dates',
    'transmission_patrimoine_skill',
)
