"""LLM-based semantic family router for raw BODACC announcements."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
import json
from typing import Any

from src.bodacc import (
    NormalizedBodaccAnnouncement,
    NormalizedParty,
    normalize_bodacc_announcement,
)
from src.llm.client import ask
from src.routing.prompt import build_family_routing_messages


ROUTING_TAXONOMY_VERSION = "family-routing-v1"
MAX_EVIDENCE_ITEMS = 3
MAX_EVIDENCE_LENGTH = 300
MAX_REASON_LENGTH = 500


class RoutingFamily(str, Enum):
    """Stable internal families, distinct from final Citrus operation codes."""

    VE = "VE"
    LG = "LG"
    TP = "TP"
    FUSION_FAMILY = "FUSION_FAMILY"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class RoutingResult:
    """Validated semantic output returned by the family router."""

    family: RoutingFamily
    evidence: tuple[str, ...]
    reason: str


class RoutingError(RuntimeError):
    """Base class for observable router failures."""

    def __init__(self, code: str, detail: str):
        super().__init__(detail)
        self.code = code
        self.detail = detail


class RoutingLLMError(RoutingError):
    """Raised when the injected or configured LLM call itself fails."""


class RoutingOutputError(RoutingError, ValueError):
    """Raised when an LLM response violates the strict routing schema."""


AskFunction = Callable[..., str]


def _party_payload(party: NormalizedParty) -> dict[str, str | None]:
    return {"name": party.name, "siren": party.siren}


def _useful_parties(
    parties: Sequence[NormalizedParty],
) -> list[dict[str, str | None]]:
    return [
        _party_payload(party)
        for party in parties
        if party.name is not None or party.siren is not None
    ]


def _unique_strings(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            unique.append(value)
    return unique


def build_routing_context(
    announcement: NormalizedBodaccAnnouncement,
) -> dict[str, Any]:
    """Project normalized source facts into a compact deterministic context.

    Dates and raw payload metadata are deliberately absent: they do not help the
    first family decision. Duplicate descriptions and origin-of-funds strings
    are emitted only once.
    """

    if not isinstance(announcement, NormalizedBodaccAnnouncement):
        raise TypeError("announcement must be normalized before context building")

    context: dict[str, Any] = {"dialect": announcement.dialect.value}
    if announcement.current_persons:
        main_party = announcement.current_persons[0]
        if main_party.name is not None or main_party.siren is not None:
            context["main_party"] = _party_payload(main_party)

    seen_descriptions: set[str] = set()
    for field, description in (
        ("act_description", announcement.act_description),
        ("sale_description", announcement.sale_description),
        ("modification_description", announcement.modification_description),
    ):
        if description is not None and description not in seen_descriptions:
            context[field] = description
            seen_descriptions.add(description)

    origins = _unique_strings(announcement.origin_funds)
    if origins:
        context["origin_funds"] = origins
    if announcement.immatriculation_category is not None:
        context["immatriculation_category"] = (
            announcement.immatriculation_category
        )

    previous_owners = _useful_parties(announcement.previous_owners)
    if previous_owners:
        context["previous_owners"] = previous_owners
    previous_operators = _useful_parties(announcement.previous_operators)
    if previous_operators:
        context["previous_operators"] = previous_operators
    return context


def _strict_json_object(raw: str) -> dict[str, Any]:
    if not isinstance(raw, str) or not raw.strip():
        raise RoutingOutputError("invalid_json", "LLM response is empty")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise RoutingOutputError(
                    "invalid_schema", f"Duplicate output field: {key}"
                )
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise RoutingOutputError(
            "invalid_json", f"Non-standard JSON constant: {value}"
        )

    try:
        parsed = json.loads(
            raw,
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except RoutingOutputError:
        raise
    except (json.JSONDecodeError, TypeError) as error:
        raise RoutingOutputError(
            "invalid_json", "LLM response is not a strict JSON object"
        ) from error
    if not isinstance(parsed, dict):
        raise RoutingOutputError(
            "invalid_schema", "Routing output must be a JSON object"
        )
    return parsed


def validate_routing_output(raw: str) -> RoutingResult:
    """Validate strict JSON without coercing invalid output to UNKNOWN."""

    payload = _strict_json_object(raw)
    expected_fields = {"family", "evidence", "reason"}
    actual_fields = set(payload)
    if actual_fields != expected_fields:
        missing = sorted(expected_fields - actual_fields)
        unexpected = sorted(actual_fields - expected_fields)
        raise RoutingOutputError(
            "invalid_schema",
            f"Routing fields mismatch; missing={missing}, unexpected={unexpected}",
        )

    family_value = payload["family"]
    if not isinstance(family_value, str):
        raise RoutingOutputError(
            "invalid_schema", "family must be a string"
        )
    try:
        family = RoutingFamily(family_value)
    except ValueError as error:
        raise RoutingOutputError(
            "invalid_family", f"Unsupported routing family: {family_value!r}"
        ) from error

    evidence_value = payload["evidence"]
    if not isinstance(evidence_value, list):
        raise RoutingOutputError(
            "invalid_schema", "evidence must be a JSON array"
        )
    if len(evidence_value) > MAX_EVIDENCE_ITEMS:
        raise RoutingOutputError(
            "invalid_schema",
            f"evidence must contain at most {MAX_EVIDENCE_ITEMS} items",
        )
    evidence: list[str] = []
    for item in evidence_value:
        if not isinstance(item, str) or not item.strip():
            raise RoutingOutputError(
                "invalid_schema", "evidence items must be non-empty strings"
            )
        normalized_item = item.strip()
        if len(normalized_item) > MAX_EVIDENCE_LENGTH:
            raise RoutingOutputError(
                "invalid_schema", "evidence items must remain brief"
            )
        evidence.append(normalized_item)

    reason_value = payload["reason"]
    if not isinstance(reason_value, str) or not reason_value.strip():
        raise RoutingOutputError(
            "invalid_schema", "reason must be a non-empty string"
        )
    reason = reason_value.strip()
    if len(reason) > MAX_REASON_LENGTH:
        raise RoutingOutputError(
            "invalid_schema", "reason must remain concise"
        )
    return RoutingResult(family=family, evidence=tuple(evidence), reason=reason)


class FamilyRouter:
    """Normalize one raw announcement, prompt the LLM, and validate its result."""

    def __init__(self, ask_function: AskFunction | None = None):
        self._ask = ask_function or ask

    def route(self, announcement: Mapping[str, Any]) -> RoutingResult:
        normalized = normalize_bodacc_announcement(announcement)
        context = build_routing_context(normalized)
        messages = build_family_routing_messages(context)
        try:
            raw = self._ask(messages, temperature=0)
        except Exception as error:
            raise RoutingLLMError(
                type(error).__name__,
                f"LLM routing call failed ({type(error).__name__})",
            ) from error
        return validate_routing_output(raw)


family_router = FamilyRouter()


__all__ = (
    "FamilyRouter",
    "ROUTING_TAXONOMY_VERSION",
    "RoutingError",
    "RoutingFamily",
    "RoutingLLMError",
    "RoutingOutputError",
    "RoutingResult",
    "build_routing_context",
    "family_router",
    "validate_routing_output",
)
