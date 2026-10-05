"""Announcement-level LLM parser for fusion-family source semantics."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
import json
import re
from typing import Any
import unicodedata

from src.bodacc import (
    NormalizedBodaccAnnouncement,
    NormalizedParty,
    extract_siren_candidates,
    normalize_bodacc_announcement,
)
from src.llm.client import ask
from src.routing.fusion_semantics_prompt import (
    FUSION_SEMANTICS_PROMPT_VERSION,
    build_fusion_semantic_messages,
)
from src.routing.fusion_subtype import (
    BeneficiaryCreation,
    TransferScope,
    TransferorFate,
)


FUSION_SEMANTICS_SCHEMA_VERSION = "fusion-semantics-v1"
MAX_EVIDENCE_ITEMS = 6
MAX_EVIDENCE_LENGTH = 300
MAX_PARTICIPANTS = 20
MAX_PARTICIPANT_NAME_LENGTH = 300
MAX_REASON_LENGTH = 500


class LegalFamily(str, Enum):
    """Locally observable legal family, without a final Citrus code."""

    FUSION = "FUSION"
    SCISSION = "SCISSION"
    UNKNOWN = "UNKNOWN"


class PartialAssetTransferWording(str, Enum):
    """Whether the source explicitly uses partial-asset-transfer wording."""

    YES = "YES"
    NO = "NO"
    UNKNOWN = "UNKNOWN"


class ParticipantRole(str, Enum):
    """Participant role established by one announcement."""

    TRANSFEROR = "TRANSFEROR"
    BENEFICIARY = "BENEFICIARY"
    BOTH_OR_UNCLEAR = "BOTH_OR_UNCLEAR"


@dataclass(frozen=True, slots=True)
class SemanticParticipant:
    """Minimal source-grounded participant needed for reconciliation."""

    siren: str | None
    name: str | None
    role: ParticipantRole


@dataclass(frozen=True, slots=True)
class FusionSemanticResult:
    """Validated facts returned by the local semantic parser."""

    legal_family: LegalFamily
    transfer_scope: TransferScope
    transferor_fate: TransferorFate
    beneficiary_creation: BeneficiaryCreation
    partial_asset_transfer_wording: PartialAssetTransferWording
    participants: tuple[SemanticParticipant, ...]
    evidence: tuple[str, ...]
    reason: str


class FusionSemanticError(RuntimeError):
    """Base class for observable local-parser failures."""

    def __init__(
        self,
        code: str,
        detail: str,
        *,
        raw_response: str | None = None,
    ):
        super().__init__(detail)
        self.code = code
        self.detail = detail
        self.raw_response = raw_response


class FusionSemanticLLMError(FusionSemanticError):
    """Raised when the injected or configured LLM call itself fails."""


class FusionSemanticOutputError(FusionSemanticError, ValueError):
    """Raised when the LLM response violates the strict semantic schema."""


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


def build_fusion_semantic_context(
    announcement: NormalizedBodaccAnnouncement,
) -> dict[str, Any]:
    """Project compact normalized facts useful to local semantic parsing."""

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
    return context


def _semantic_source_text(context: Mapping[str, Any]) -> str:
    """Serialize only projected source facts for deterministic SIREN checks."""

    return json.dumps(
        dict(context),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def fusion_semantic_source_sirens(
    announcement: NormalizedBodaccAnnouncement,
) -> tuple[str, ...]:
    """Return the exact validated SIREN universe accepted by the parser."""

    if not isinstance(announcement, NormalizedBodaccAnnouncement):
        raise TypeError("announcement must be normalized before SIREN discovery")
    context = build_fusion_semantic_context(announcement)
    return extract_siren_candidates(_semantic_source_text(context))


def _canonical_text(value: str) -> str:
    """Normalize Unicode, accents, case, punctuation and whitespace."""

    decomposed = unicodedata.normalize("NFKD", value).casefold()
    without_marks = "".join(
        character
        for character in decomposed
        if not unicodedata.combining(character)
    )
    words = "".join(
        character if character.isalnum() else " "
        for character in without_marks
    )
    return " ".join(words.split())


def _source_contains_name(name: str, source_text: str) -> bool:
    canonical_name = _canonical_text(name)
    canonical_source = _canonical_text(source_text)
    if not canonical_name:
        return False
    if f" {canonical_name} " in f" {canonical_source} ":
        return True
    return (
        canonical_name.replace(" ", "")
        in canonical_source.replace(" ", "")
    )


def _output_error(
    code: str,
    detail: str,
    raw: Any,
) -> FusionSemanticOutputError:
    return FusionSemanticOutputError(
        code,
        detail,
        raw_response=raw if isinstance(raw, str) else None,
    )


def _strict_json_object(raw: str) -> dict[str, Any]:
    if not isinstance(raw, str) or not raw.strip():
        raise _output_error("invalid_json", "LLM response is empty", raw)

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise _output_error(
                    "invalid_schema",
                    f"Duplicate output field: {key}",
                    raw,
                )
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise _output_error(
            "invalid_json",
            f"Non-standard JSON constant: {value}",
            raw,
        )

    try:
        parsed = json.loads(
            raw,
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except FusionSemanticOutputError:
        raise
    except (json.JSONDecodeError, TypeError) as error:
        raise _output_error(
            "invalid_json",
            "LLM response is not a strict JSON object",
            raw,
        ) from error
    if not isinstance(parsed, dict):
        raise _output_error(
            "invalid_schema",
            "Fusion semantic output must be a JSON object",
            raw,
        )
    return parsed


def _validated_enum(
    payload: Mapping[str, Any],
    field: str,
    enum_type: type[Enum],
    *,
    invalid_value_code: str,
    raw: str,
) -> Enum:
    value = payload[field]
    if not isinstance(value, str):
        raise _output_error(
            "invalid_schema",
            f"{field} must be a string",
            raw,
        )
    try:
        return enum_type(value)
    except ValueError as error:
        raise _output_error(
            invalid_value_code,
            f"Unsupported {field}: {value!r}",
            raw,
        ) from error


def _validated_participants(
    value: Any,
    *,
    source_text: str,
    raw: str,
) -> tuple[SemanticParticipant, ...]:
    if not isinstance(value, list):
        raise _output_error(
            "invalid_schema",
            "participants must be a JSON array",
            raw,
        )

    if len(value) > MAX_PARTICIPANTS:
        raise _output_error(
            "invalid_schema",
            f"participants must contain at most {MAX_PARTICIPANTS} items",
            raw,
        )

    source_sirens = set(extract_siren_candidates(source_text))
    participants: list[SemanticParticipant] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise _output_error(
                "invalid_schema",
                f"participants[{index}] must be a JSON object",
                raw,
            )
        expected_fields = {"siren", "name", "role"}
        actual_fields = set(item)
        if actual_fields != expected_fields:
            missing = sorted(expected_fields - actual_fields)
            unexpected = sorted(actual_fields - expected_fields)
            raise _output_error(
                "invalid_schema",
                f"participants[{index}] fields mismatch; "
                f"missing={missing}, unexpected={unexpected}",
                raw,
            )

        siren_value = item["siren"]
        if siren_value is not None:
            if (
                not isinstance(siren_value, str)
                or re.fullmatch(r"\d{9}", siren_value) is None
                or siren_value not in source_sirens
            ):
                raise _output_error(
                    "invalid_participant_siren",
                    f"participants[{index}].siren must be a source-present, "
                    "Luhn-valid 9-digit string or null",
                    raw,
                )

        name_value = item["name"]
        if name_value is not None:
            if not isinstance(name_value, str) or not name_value.strip():
                raise _output_error(
                    "invalid_schema",
                    f"participants[{index}].name must be a non-empty "
                    "string or null",
                    raw,
                )
            name_value = name_value.strip()
            if len(name_value) > MAX_PARTICIPANT_NAME_LENGTH:
                raise _output_error(
                    "invalid_schema",
                    f"participants[{index}].name must remain brief",
                    raw,
                )
            if not _source_contains_name(name_value, source_text):
                raise _output_error(
                    "invalid_participant_name",
                    f"participants[{index}].name must be present in the "
                    "normalized source context",
                    raw,
                )
        if siren_value is None and name_value is None:
            raise _output_error(
                "invalid_schema",
                f"participants[{index}] must identify a source participant",
                raw,
            )

        role = _validated_enum(
            item,
            "role",
            ParticipantRole,
            invalid_value_code="invalid_participant_role",
            raw=raw,
        )
        participants.append(
            SemanticParticipant(
                siren=siren_value,
                name=name_value,
                role=role,
            )
        )
    return tuple(participants)


def _validated_evidence(value: Any, *, raw: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise _output_error(
            "invalid_schema",
            "evidence must be a JSON array",
            raw,
        )
    if len(value) > MAX_EVIDENCE_ITEMS:
        raise _output_error(
            "invalid_schema",
            f"evidence must contain at most {MAX_EVIDENCE_ITEMS} items",
            raw,
        )

    evidence: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise _output_error(
                "invalid_schema",
                "evidence items must be non-empty strings",
                raw,
            )
        normalized_item = item.strip()
        if len(normalized_item) > MAX_EVIDENCE_LENGTH:
            raise _output_error(
                "invalid_schema",
                "evidence items must remain brief",
                raw,
            )
        evidence.append(normalized_item)
    return tuple(evidence)


def validate_fusion_semantic_output(
    raw: str,
    source_text: str | None = None,
) -> FusionSemanticResult:
    """Validate strict JSON and every non-null SIREN against source facts."""

    if source_text is not None and not isinstance(source_text, str):
        raise TypeError("source_text must be a string or None")

    payload = _strict_json_object(raw)
    expected_fields = {
        "legal_family",
        "transfer_scope",
        "transferor_fate",
        "beneficiary_creation",
        "partial_asset_transfer_wording",
        "participants",
        "evidence",
        "reason",
    }
    actual_fields = set(payload)
    if actual_fields != expected_fields:
        missing = sorted(expected_fields - actual_fields)
        unexpected = sorted(actual_fields - expected_fields)
        raise _output_error(
            "invalid_schema",
            "Fusion semantic fields mismatch; "
            f"missing={missing}, unexpected={unexpected}",
            raw,
        )

    legal_family = _validated_enum(
        payload,
        "legal_family",
        LegalFamily,
        invalid_value_code="invalid_legal_family",
        raw=raw,
    )
    transfer_scope = _validated_enum(
        payload,
        "transfer_scope",
        TransferScope,
        invalid_value_code="invalid_semantic_axis",
        raw=raw,
    )
    transferor_fate = _validated_enum(
        payload,
        "transferor_fate",
        TransferorFate,
        invalid_value_code="invalid_semantic_axis",
        raw=raw,
    )
    beneficiary_creation = _validated_enum(
        payload,
        "beneficiary_creation",
        BeneficiaryCreation,
        invalid_value_code="invalid_semantic_axis",
        raw=raw,
    )
    partial_asset_transfer_wording = _validated_enum(
        payload,
        "partial_asset_transfer_wording",
        PartialAssetTransferWording,
        invalid_value_code="invalid_semantic_axis",
        raw=raw,
    )
    participants = _validated_participants(
        payload["participants"],
        source_text=source_text or "",
        raw=raw,
    )
    evidence = _validated_evidence(payload["evidence"], raw=raw)

    reason_value = payload["reason"]
    if not isinstance(reason_value, str) or not reason_value.strip():
        raise _output_error(
            "invalid_schema",
            "reason must be a non-empty string",
            raw,
        )
    reason = reason_value.strip()
    if len(reason) > MAX_REASON_LENGTH:
        raise _output_error(
            "invalid_schema",
            "reason must remain concise",
            raw,
        )

    return FusionSemanticResult(
        legal_family=legal_family,
        transfer_scope=transfer_scope,
        transferor_fate=transferor_fate,
        beneficiary_creation=beneficiary_creation,
        partial_asset_transfer_wording=partial_asset_transfer_wording,
        participants=participants,
        evidence=evidence,
        reason=reason,
    )


class FusionSemanticParser:
    """Normalize one raw announcement and parse only its supported facts."""

    def __init__(self, ask_function: AskFunction | None = None):
        self._ask = ask_function or ask

    def parse(
        self,
        announcement: Mapping[str, Any],
    ) -> FusionSemanticResult:
        if not isinstance(announcement, Mapping):
            raise TypeError("announcement must be a raw BODACC mapping")
        normalized = normalize_bodacc_announcement(announcement)
        context = build_fusion_semantic_context(normalized)
        messages = build_fusion_semantic_messages(context)
        try:
            raw = self._ask(messages, temperature=0)
        except Exception as error:
            raise FusionSemanticLLMError(
                type(error).__name__,
                "LLM fusion-semantic parsing call failed "
                f"({type(error).__name__})",
            ) from error
        return validate_fusion_semantic_output(
            raw,
            source_text=_semantic_source_text(context),
        )


fusion_semantic_parser = FusionSemanticParser()


__all__ = (
    "FUSION_SEMANTICS_PROMPT_VERSION",
    "FUSION_SEMANTICS_SCHEMA_VERSION",
    "FusionSemanticError",
    "FusionSemanticLLMError",
    "FusionSemanticOutputError",
    "FusionSemanticParser",
    "FusionSemanticResult",
    "LegalFamily",
    "ParticipantRole",
    "PartialAssetTransferWording",
    "SemanticParticipant",
    "build_fusion_semantic_context",
    "fusion_semantic_parser",
    "fusion_semantic_source_sirens",
    "validate_fusion_semantic_output",
)
