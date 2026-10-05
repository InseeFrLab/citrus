"""LLM router specialized in final fusion-family subtypes."""

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
from src.routing.fusion_prompt import build_fusion_subtype_messages


FUSION_SUBTYPE_TAXONOMY_VERSION = "fusion-subtype-routing-v1"
MAX_EVIDENCE_ITEMS = 5
MAX_EVIDENCE_LENGTH = 300
MAX_REASON_LENGTH = 500


class FusionSubtype(str, Enum):
    """Canonical final Citrus codes handled by the specialized router."""

    FU = "FU"
    AB = "AB"
    SP = "SP"
    ST = "ST"
    AP = "AP"
    UNKNOWN = "UNKNOWN"


class TransferScope(str, Enum):
    """Whether the announced transfer covers all or part of the assets."""

    TOTAL = "TOTAL"
    PARTIAL = "PARTIAL"
    UNKNOWN = "UNKNOWN"


class TransferorFate(str, Enum):
    """Whether the transferor disappears or survives the operation."""

    DISAPPEARS = "DISAPPEARS"
    SURVIVES = "SURVIVES"
    UNKNOWN = "UNKNOWN"


class BeneficiaryCreation(str, Enum):
    """Whether the beneficiary was created specifically for the operation."""

    NEW = "NEW"
    EXISTING = "EXISTING"
    MIXED_OR_UNKNOWN = "MIXED_OR_UNKNOWN"


class BeneficiaryCount(str, Enum):
    """Number of beneficiaries established by normalized source facts."""

    ONE = "ONE"
    MULTIPLE = "MULTIPLE"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class FusionSubtypeResult:
    """Validated label and inspectable semantic axes returned by the LLM."""

    subtype: FusionSubtype
    transfer_scope: TransferScope
    transferor_fate: TransferorFate
    beneficiary_creation: BeneficiaryCreation
    beneficiary_count: BeneficiaryCount
    evidence: tuple[str, ...]
    reason: str


class FusionSubtypeRoutingError(RuntimeError):
    """Base class for observable specialized-router failures."""

    def __init__(self, code: str, detail: str):
        super().__init__(detail)
        self.code = code
        self.detail = detail


class FusionSubtypeLLMError(FusionSubtypeRoutingError):
    """Raised when the injected or configured LLM call itself fails."""


class FusionSubtypeOutputError(FusionSubtypeRoutingError, ValueError):
    """Raised when the LLM response violates the strict output schema."""


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


def build_fusion_subtype_context(
    announcement: NormalizedBodaccAnnouncement,
) -> dict[str, Any]:
    """Project only deterministic normalized facts useful to subtype routing."""

    if not isinstance(announcement, NormalizedBodaccAnnouncement):
        raise TypeError("announcement must be normalized before context building")

    context: dict[str, Any] = {"dialect": announcement.dialect.value}
    if announcement.current_persons:
        main_party = announcement.current_persons[0]
        if main_party.name is not None or main_party.siren is not None:
            context["main_party"] = _party_payload(main_party)

    descriptions: list[str] = []
    seen_descriptions: set[str] = set()
    for field, description in (
        ("act_description", announcement.act_description),
        ("sale_description", announcement.sale_description),
        ("modification_description", announcement.modification_description),
    ):
        if description is not None and description not in seen_descriptions:
            context[field] = description
            descriptions.append(description)
            seen_descriptions.add(description)
    if descriptions:
        context["all_descriptions"] = descriptions

    previous_owners = _useful_parties(announcement.previous_owners)
    if previous_owners:
        context["previous_owners"] = previous_owners
    if announcement.immatriculation_category is not None:
        context["immatriculation_category"] = (
            announcement.immatriculation_category
        )
    if announcement.immatriculation_date is not None:
        context["immatriculation_date"] = announcement.immatriculation_date
    if announcement.publication_date is not None:
        context["publication_date"] = announcement.publication_date
    return context


def _strict_json_object(raw: str) -> dict[str, Any]:
    if not isinstance(raw, str) or not raw.strip():
        raise FusionSubtypeOutputError("invalid_json", "LLM response is empty")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise FusionSubtypeOutputError(
                    "invalid_schema", f"Duplicate output field: {key}"
                )
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise FusionSubtypeOutputError(
            "invalid_json", f"Non-standard JSON constant: {value}"
        )

    try:
        parsed = json.loads(
            raw,
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except FusionSubtypeOutputError:
        raise
    except (json.JSONDecodeError, TypeError) as error:
        raise FusionSubtypeOutputError(
            "invalid_json", "LLM response is not a strict JSON object"
        ) from error
    if not isinstance(parsed, dict):
        raise FusionSubtypeOutputError(
            "invalid_schema", "Fusion subtype output must be a JSON object"
        )
    return parsed


def _validated_enum(
    payload: Mapping[str, Any],
    field: str,
    enum_type: type[Enum],
    *,
    invalid_value_code: str,
) -> Enum:
    value = payload[field]
    if not isinstance(value, str):
        raise FusionSubtypeOutputError(
            "invalid_schema", f"{field} must be a string"
        )
    try:
        return enum_type(value)
    except ValueError as error:
        raise FusionSubtypeOutputError(
            invalid_value_code,
            f"Unsupported {field}: {value!r}",
        ) from error


def validate_fusion_subtype_output(raw: str) -> FusionSubtypeResult:
    """Validate strict JSON without coercing invalid values to UNKNOWN."""

    payload = _strict_json_object(raw)
    expected_fields = {
        "subtype",
        "transfer_scope",
        "transferor_fate",
        "beneficiary_creation",
        "beneficiary_count",
        "evidence",
        "reason",
    }
    actual_fields = set(payload)
    if actual_fields != expected_fields:
        missing = sorted(expected_fields - actual_fields)
        unexpected = sorted(actual_fields - expected_fields)
        raise FusionSubtypeOutputError(
            "invalid_schema",
            "Fusion subtype fields mismatch; "
            f"missing={missing}, unexpected={unexpected}",
        )

    subtype = _validated_enum(
        payload,
        "subtype",
        FusionSubtype,
        invalid_value_code="invalid_subtype",
    )
    transfer_scope = _validated_enum(
        payload,
        "transfer_scope",
        TransferScope,
        invalid_value_code="invalid_semantic_axis",
    )
    transferor_fate = _validated_enum(
        payload,
        "transferor_fate",
        TransferorFate,
        invalid_value_code="invalid_semantic_axis",
    )
    beneficiary_creation = _validated_enum(
        payload,
        "beneficiary_creation",
        BeneficiaryCreation,
        invalid_value_code="invalid_semantic_axis",
    )
    beneficiary_count = _validated_enum(
        payload,
        "beneficiary_count",
        BeneficiaryCount,
        invalid_value_code="invalid_semantic_axis",
    )

    evidence_value = payload["evidence"]
    if not isinstance(evidence_value, list):
        raise FusionSubtypeOutputError(
            "invalid_schema", "evidence must be a JSON array"
        )
    if len(evidence_value) > MAX_EVIDENCE_ITEMS:
        raise FusionSubtypeOutputError(
            "invalid_schema",
            f"evidence must contain at most {MAX_EVIDENCE_ITEMS} items",
        )
    evidence: list[str] = []
    for item in evidence_value:
        if not isinstance(item, str) or not item.strip():
            raise FusionSubtypeOutputError(
                "invalid_schema", "evidence items must be non-empty strings"
            )
        normalized_item = item.strip()
        if len(normalized_item) > MAX_EVIDENCE_LENGTH:
            raise FusionSubtypeOutputError(
                "invalid_schema", "evidence items must remain brief"
            )
        evidence.append(normalized_item)

    reason_value = payload["reason"]
    if not isinstance(reason_value, str) or not reason_value.strip():
        raise FusionSubtypeOutputError(
            "invalid_schema", "reason must be a non-empty string"
        )
    reason = reason_value.strip()
    if len(reason) > MAX_REASON_LENGTH:
        raise FusionSubtypeOutputError(
            "invalid_schema", "reason must remain concise"
        )

    return FusionSubtypeResult(
        subtype=subtype,
        transfer_scope=transfer_scope,
        transferor_fate=transferor_fate,
        beneficiary_creation=beneficiary_creation,
        beneficiary_count=beneficiary_count,
        evidence=tuple(evidence),
        reason=reason,
    )


_CONSISTENCY_EXPECTATIONS: dict[
    FusionSubtype, tuple[tuple[str, Enum], ...]
] = {
    FusionSubtype.FU: (
        ("transfer_scope", TransferScope.TOTAL),
        ("transferor_fate", TransferorFate.DISAPPEARS),
        ("beneficiary_creation", BeneficiaryCreation.NEW),
    ),
    FusionSubtype.AB: (
        ("transfer_scope", TransferScope.TOTAL),
        ("transferor_fate", TransferorFate.DISAPPEARS),
        ("beneficiary_creation", BeneficiaryCreation.EXISTING),
    ),
    FusionSubtype.SP: (
        ("transfer_scope", TransferScope.PARTIAL),
        ("transferor_fate", TransferorFate.SURVIVES),
        ("beneficiary_creation", BeneficiaryCreation.NEW),
    ),
    FusionSubtype.AP: (
        ("transfer_scope", TransferScope.PARTIAL),
        ("transferor_fate", TransferorFate.SURVIVES),
        ("beneficiary_creation", BeneficiaryCreation.EXISTING),
    ),
    FusionSubtype.ST: (
        ("transfer_scope", TransferScope.TOTAL),
        ("transferor_fate", TransferorFate.DISAPPEARS),
        ("beneficiary_count", BeneficiaryCount.MULTIPLE),
    ),
}

_UNCERTAIN_AXIS_VALUES = {
    TransferScope.UNKNOWN,
    TransferorFate.UNKNOWN,
    BeneficiaryCreation.MIXED_OR_UNKNOWN,
    BeneficiaryCount.UNKNOWN,
}


def semantic_consistency_issues(
    result: FusionSubtypeResult,
) -> tuple[str, ...]:
    """Return obvious subtype/axis contradictions without changing subtype."""

    if not isinstance(result, FusionSubtypeResult):
        raise TypeError("result must be a FusionSubtypeResult")

    issues: list[str] = []
    for field, expected in _CONSISTENCY_EXPECTATIONS.get(
        result.subtype, ()
    ):
        actual = getattr(result, field)
        if actual in _UNCERTAIN_AXIS_VALUES:
            continue
        if actual != expected:
            issues.append(
                f"{field}:expected={expected.value},actual={actual.value}"
            )
    return tuple(issues)


def is_semantically_consistent(result: FusionSubtypeResult) -> bool:
    """Whether explicit semantic axes avoid obvious subtype contradictions."""

    return not semantic_consistency_issues(result)


class FusionSubtypeRouter:
    """Normalize one raw announcement, call the LLM, and validate its result."""

    def __init__(self, ask_function: AskFunction | None = None):
        self._ask = ask_function or ask

    def route(
        self, announcement: Mapping[str, Any]
    ) -> FusionSubtypeResult:
        if not isinstance(announcement, Mapping):
            raise TypeError("announcement must be a raw BODACC mapping")
        normalized = normalize_bodacc_announcement(announcement)
        context = build_fusion_subtype_context(normalized)
        messages = build_fusion_subtype_messages(context)
        try:
            raw = self._ask(messages, temperature=0)
        except Exception as error:
            raise FusionSubtypeLLMError(
                type(error).__name__,
                "LLM fusion-subtype routing call failed "
                f"({type(error).__name__})",
            ) from error
        return validate_fusion_subtype_output(raw)


fusion_subtype_router = FusionSubtypeRouter()


__all__ = (
    "BeneficiaryCount",
    "BeneficiaryCreation",
    "FUSION_SUBTYPE_TAXONOMY_VERSION",
    "FusionSubtype",
    "FusionSubtypeLLMError",
    "FusionSubtypeOutputError",
    "FusionSubtypeResult",
    "FusionSubtypeRouter",
    "FusionSubtypeRoutingError",
    "TransferScope",
    "TransferorFate",
    "build_fusion_subtype_context",
    "fusion_subtype_router",
    "is_semantically_consistent",
    "semantic_consistency_issues",
    "validate_fusion_subtype_output",
)
