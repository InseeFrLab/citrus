"""Non-mutating normalization of BODACC OpenData announcements.

This module exposes source facts only. It deliberately contains no operation
classification, role inference, date cascade, or amount normalization.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
import json
import re
from typing import Any, Iterable, Mapping

from src.utils import is_luhn_valid


class BodaccNormalizationError(ValueError):
    """Raised when a present BODACC structure cannot be interpreted safely."""


class BodaccDialect(str, Enum):
    """Known source-level BODACC registry dialects."""

    RCS_A = "RCS-A"
    RCS_B = "RCS-B"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class NormalizedParty:
    """A source party without any inferred operation role."""

    siren: str | None
    name: str | None


@dataclass(frozen=True, slots=True)
class NormalizedBodaccAnnouncement:
    """Stable, source-oriented view of one raw BODACC announcement."""

    raw_payload: Mapping[str, Any]
    dialect: BodaccDialect
    current_persons: tuple[NormalizedParty, ...]
    previous_owners: tuple[NormalizedParty, ...]
    previous_operators: tuple[NormalizedParty, ...]
    act_description: str | None
    sale_description: str | None
    modification_description: str | None
    all_descriptions: tuple[str, ...]
    publication_date: str | None
    commencement_date: str | None
    effect_date: str | None
    immatriculation_category: str | None
    immatriculation_date: str | None
    legal_publication_date: str | None
    source_url: str | None
    origin_funds: tuple[str, ...]

    @property
    def source_dialect(self) -> BodaccDialect:
        """Compatibility-friendly explicit name for the detected dialect."""

        return self.dialect

    @property
    def main_siren(self) -> str | None:
        """SIREN of the first current person, when supplied by the source."""

        return self.current_persons[0].siren if self.current_persons else None

    @property
    def main_name(self) -> str | None:
        """Name of the first current person, when supplied by the source."""

        return self.current_persons[0].name if self.current_persons else None

    @property
    def primary_description(self) -> str | None:
        """Dialect-specific description without cross-source concatenation."""

        if self.dialect is BodaccDialect.RCS_A:
            return self.sale_description
        if self.dialect is BodaccDialect.RCS_B:
            return self.modification_description
        return None

    @property
    def descriptions(self) -> tuple[str, ...]:
        """All supported descriptions in stable act, sale, modification order."""

        return self.all_descriptions

    @property
    def first_origin_funds(self) -> str | None:
        """First establishment origin-of-funds value, when present."""

        return self.origin_funds[0] if self.origin_funds else None


_JSON_CONTAINER_FIELDS = (
    "listepersonnes",
    "listeprecedentproprietaire",
    "listeprecedentexploitant",
    "precedentExploitantPM",
    "precedentExploitantPP",
    "listeetablissements",
    "acte",
    "modifications",
    "modificationsGenerales",
    "modificationsgenerales",
)

_DIALECT_METADATA_FIELDS = (
    "registre",
    "typeavis",
    "typeavis_lib",
    "familleavis",
    "familleavis_lib",
    "typeAnnonce",
    "type_annonce",
)

_SIREN_PATTERN = re.compile(
    r"(?<!\d)(?:\d{9}|\d{3}(?:[.\s]+\d{3}){2})(?!\d)"
)
_CURRENCY_AFTER_PATTERN = re.compile(
    r"^\s*(?:[,.]\d{1,2})?\s*(?:€|euros?\b|eur\b)",
    re.IGNORECASE,
)
_CURRENCY_BEFORE_PATTERN = re.compile(r"(?:€|euros?\b|eur\b)\s*$", re.IGNORECASE)


def _parse_optional_container(
    value: Any, field: str
) -> dict[str, Any] | list[Any] | None:
    if value is None:
        return None
    if isinstance(value, str):
        if not value.strip():
            return None
        try:
            value = json.loads(value)
        except json.JSONDecodeError as error:
            raise BodaccNormalizationError(
                f"Malformed JSON container in {field}: {error.msg}"
            ) from error
        if value is None:
            return None
    if not isinstance(value, (dict, list)):
        raise BodaccNormalizationError(
            f"Expected a JSON object or array in {field}, got {type(value).__name__}"
        )
    return deepcopy(value)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _non_empty_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def _first_text(*values: Any) -> str | None:
    for value in values:
        text = _non_empty_text(value)
        if text is not None:
            return text
    return None


def _contained_items(
    container: Any, item_key: str
) -> tuple[Mapping[str, Any], ...]:
    if container is None:
        return ()
    if isinstance(container, list):
        values = container
    elif isinstance(container, Mapping):
        values = container.get(item_key)
        if values is None:
            values = container if item_key == "personne" else []
        if isinstance(values, str):
            values = _parse_optional_container(values, item_key)
        if isinstance(values, Mapping):
            values = [values]
    else:
        return ()
    if not isinstance(values, list):
        return ()
    return tuple(value for value in values if isinstance(value, Mapping))


def _normalize_structured_siren(value: Any) -> str | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        digits = str(value)
    elif isinstance(value, str):
        digits = re.sub(r"[.\s]", "", value.strip())
    else:
        return None
    if not digits.isdigit():
        return None
    if len(digits) == 8:
        digits = digits.zfill(9)
    return digits if len(digits) == 9 else None


def _party_from_person(person: Mapping[str, Any]) -> NormalizedParty:
    registration = _mapping(person.get("numeroImmatriculation"))
    siren = _normalize_structured_siren(
        registration.get("numeroIdentification")
        if registration.get("numeroIdentification") is not None
        else registration.get("numeroIdentificationRCS")
    )
    denomination = _non_empty_text(person.get("denomination"))
    if denomination is None:
        denomination = _first_text(person.get("nom"), person.get("raisonSociale"))
    return NormalizedParty(siren=siren, name=denomination)


def _normalize_parties(container: Any) -> tuple[NormalizedParty, ...]:
    return tuple(
        _party_from_person(person)
        for person in _contained_items(container, "personne")
    )


def _normalize_party_source(
    value: Any, field: str
) -> tuple[NormalizedParty, ...]:
    if isinstance(value, str):
        value = _parse_optional_container(value, field)
    return _normalize_parties(value)


def _previous_operators(
    parsed: Mapping[str, Any],
    modifications_generales: Mapping[str, Any],
) -> tuple[NormalizedParty, ...]:
    sources = (
        (parsed["listeprecedentexploitant"], "listeprecedentexploitant"),
        (parsed["precedentExploitantPM"], "precedentExploitantPM"),
        (parsed["precedentExploitantPP"], "precedentExploitantPP"),
        (
            modifications_generales.get("precedentExploitantPM"),
            "modificationsGenerales.precedentExploitantPM",
        ),
        (
            modifications_generales.get("precedentExploitantPP"),
            "modificationsGenerales.precedentExploitantPP",
        ),
    )
    return tuple(
        party
        for source, field in sources
        for party in _normalize_party_source(source, field)
    )


def _metadata_dialect(raw_payload: Mapping[str, Any]) -> BodaccDialect | None:
    detected: set[BodaccDialect] = set()
    for field in _DIALECT_METADATA_FIELDS:
        value = raw_payload.get(field)
        if not isinstance(value, str):
            continue
        compact = re.sub(r"[^A-Z0-9]", "", value.upper())
        if "RCSA" in compact:
            detected.add(BodaccDialect.RCS_A)
        if "RCSB" in compact:
            detected.add(BodaccDialect.RCS_B)
        normalized = re.sub(r"\s+", " ", value).strip().casefold()
        if field in {"familleavis", "familleavis_lib"} and normalized in {
            "modification",
            "modifications",
            "modifications diverses",
        }:
            detected.add(BodaccDialect.RCS_B)
    if len(detected) == 1:
        return detected.pop()
    if len(detected) > 1:
        return BodaccDialect.UNKNOWN
    return None


def _structural_dialect(
    current_people: tuple[Mapping[str, Any], ...],
    acte: Mapping[str, Any],
    modifications_generales: Mapping[str, Any],
) -> BodaccDialect:
    evidence: set[BodaccDialect] = set()
    for person in current_people:
        registration = _mapping(person.get("numeroImmatriculation"))
        if registration.get("numeroIdentification") is not None:
            evidence.add(BodaccDialect.RCS_A)
        if registration.get("numeroIdentificationRCS") is not None:
            evidence.add(BodaccDialect.RCS_B)
    vente = _mapping(acte.get("vente"))
    if _non_empty_text(vente.get("descriptif")) is not None:
        evidence.add(BodaccDialect.RCS_A)
    if _non_empty_text(modifications_generales.get("descriptif")) is not None:
        evidence.add(BodaccDialect.RCS_B)
    if len(evidence) == 1:
        return evidence.pop()
    return BodaccDialect.UNKNOWN


def _modification_alias(
    container: Mapping[str, Any], *, prefix: str = ""
) -> Any:
    values: list[tuple[str, Any]] = []
    for key in ("modificationsGenerales", "modificationsgenerales"):
        value = container.get(key)
        field = f"{prefix}{key}"
        if isinstance(value, str):
            value = _parse_optional_container(value, field)
        if value is not None:
            values.append((field, value))

    if len(values) == 2 and values[0][1] != values[1][1]:
        raise BodaccNormalizationError(
            "Conflicting modification containers in "
            f"{values[0][0]} and {values[1][0]}"
        )
    return values[0][1] if values else None


def _modifications_generales(parsed: Mapping[str, Any]) -> Mapping[str, Any]:
    direct = _modification_alias(parsed)
    if direct is not None:
        return _mapping(direct)
    modifications = _mapping(parsed.get("modifications"))
    nested = _modification_alias(modifications, prefix="modifications.")
    return _mapping(nested)


def _origin_funds(establishments: Any) -> tuple[str, ...]:
    values: list[str] = []
    for establishment in _contained_items(establishments, "etablissement"):
        value = _non_empty_text(establishment.get("origineFonds"))
        if value is not None:
            values.append(value)
    return tuple(values)


def normalize_bodacc_announcement(
    raw_payload: Mapping[str, Any],
) -> NormalizedBodaccAnnouncement:
    """Return a normalized source view without mutating the raw payload.

    Missing optional structures become empty tuples or None. Present,
    non-empty stringified structures must decode to a JSON object or array;
    malformed JSON and scalar JSON values raise BodaccNormalizationError.
    """

    if not isinstance(raw_payload, Mapping):
        raise BodaccNormalizationError("A BODACC announcement must be a mapping")

    raw_copy = deepcopy(dict(raw_payload))
    parsed = {
        field: _parse_optional_container(raw_payload.get(field), field)
        for field in _JSON_CONTAINER_FIELDS
    }

    acte = _mapping(parsed["acte"])
    vente = _mapping(acte.get("vente"))
    immatriculation_value = acte.get("immatriculation")
    if isinstance(immatriculation_value, str):
        immatriculation_value = _parse_optional_container(
            immatriculation_value, "acte.immatriculation"
        )
    immatriculation = _mapping(immatriculation_value)
    modifications_generales = _modifications_generales(parsed)
    current_people = _contained_items(parsed["listepersonnes"], "personne")

    act_description = _non_empty_text(acte.get("descriptif"))
    sale_description = _non_empty_text(vente.get("descriptif"))
    modification_description = _non_empty_text(
        modifications_generales.get("descriptif")
    )
    descriptions = tuple(
        dict.fromkeys(
            description
            for description in (
                act_description,
                sale_description,
                modification_description,
            )
            if description is not None
        )
    )

    explicit_dialect = _metadata_dialect(raw_payload)
    dialect = (
        explicit_dialect
        if explicit_dialect is not None
        else _structural_dialect(current_people, acte, modifications_generales)
    )

    return NormalizedBodaccAnnouncement(
        raw_payload=raw_copy,
        dialect=dialect,
        current_persons=tuple(
            _party_from_person(person) for person in current_people
        ),
        previous_owners=_normalize_parties(
            parsed["listeprecedentproprietaire"]
        ),
        previous_operators=_previous_operators(
            parsed, modifications_generales
        ),
        act_description=act_description,
        sale_description=sale_description,
        modification_description=modification_description,
        all_descriptions=descriptions,
        publication_date=_non_empty_text(raw_payload.get("dateparution")),
        commencement_date=_first_text(
            raw_payload.get("dateCommencementActivite"),
            acte.get("dateCommencementActivite"),
            vente.get("dateCommencementActivite"),
            immatriculation.get("dateCommencementActivite"),
            modifications_generales.get("dateCommencementActivite"),
        ),
        effect_date=_first_text(
            raw_payload.get("dateEffet"),
            acte.get("dateEffet"),
            immatriculation.get("dateEffet"),
            modifications_generales.get("dateEffet"),
        ),
        immatriculation_category=_non_empty_text(
            immatriculation.get("categorieImmatriculation")
        ),
        immatriculation_date=_non_empty_text(
            immatriculation.get("dateImmatriculation")
        ),
        legal_publication_date=_non_empty_text(
            _mapping(vente.get("publiciteLegale")).get("date")
        ),
        source_url=_first_text(
            raw_payload.get("url_complete"), raw_payload.get("url")
        ),
        origin_funds=_origin_funds(parsed["listeetablissements"]),
    )


def extract_siren_candidates(
    text: str | None,
    excluded_sirens: Iterable[str | int] = (),
) -> tuple[str, ...]:
    """Find ordered, unique, Luhn-valid SIRENs without inferring party roles."""

    if text is None:
        return ()
    if not isinstance(text, str):
        raise TypeError("text must be a string or None")

    excluded = {
        normalized
        for value in excluded_sirens
        if (normalized := _normalize_structured_siren(value)) is not None
    }
    candidates: list[str] = []
    seen: set[str] = set()
    for match in _SIREN_PATTERN.finditer(text):
        if _CURRENCY_AFTER_PATTERN.search(text[match.end():match.end() + 16]):
            continue
        if _CURRENCY_BEFORE_PATTERN.search(
            text[max(0, match.start() - 16):match.start()]
        ):
            continue
        candidate = re.sub(r"[.\s]", "", match.group())
        if (
            candidate in excluded
            or candidate in seen
            or not is_luhn_valid(candidate)
        ):
            continue
        seen.add(candidate)
        candidates.append(candidate)
    return tuple(candidates)


__all__ = (
    "BodaccDialect",
    "BodaccNormalizationError",
    "NormalizedBodaccAnnouncement",
    "NormalizedParty",
    "extract_siren_candidates",
    "normalize_bodacc_announcement",
)
