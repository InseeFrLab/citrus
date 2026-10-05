"""Pure provisional modelling and global fusion-family reconciliation.

This module is deliberately downstream from the announcement-level semantic
parser.  It never calls an LLM or a network service, and it accepts no
benchmark labels.  ``FZ`` and ``SZ`` are internal provisional states only;
reconciled records expose canonical final Citrus types or ``UNKNOWN``.

The historical isolated-``FU``/``ST`` to ``AP`` fallback is intentionally not
implemented in this first reconciliation version.  The available description
does not establish a faithful mapping from the historical operation pairs to
the current one-row-per-announcement representation.  Applying it here would
therefore turn missing linkage into an invented final classification.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
import re
import unicodedata

from src.bodacc import (
    NormalizedBodaccAnnouncement,
    extract_siren_candidates,
)
from src.routing.fusion_semantics import (
    FusionSemanticResult,
    LegalFamily,
    ParticipantRole,
    PartialAssetTransferWording,
    SemanticParticipant,
)
from src.routing.fusion_subtype import (
    BeneficiaryCreation,
    TransferScope,
    TransferorFate,
)


FUSION_RECONCILIATION_VERSION = "fusion-reconciliation-v1"
DESCRIPTION_FINGERPRINT_VERSION = "unicode-nfkc-whitespace-sha256-v1"
HISTORICAL_ISOLATED_AP_FALLBACK_IMPLEMENTED = False
HISTORICAL_ISOLATED_AP_FALLBACK_NOTE = (
    "Historical isolated FU/ST-to-AP fallback not applied: the current "
    "announcement representation does not reproduce its operation-pair "
    "preconditions faithfully."
)


class ProvisionalType(str, Enum):
    """Internal local states retained for global reconciliation."""

    AB = "AB"
    FZ = "FZ"
    SP = "SP"
    SZ = "SZ"
    AP = "AP"
    UNKNOWN = "UNKNOWN"


class FinalFusionType(str, Enum):
    """Canonical final types emitted by this reconciliation boundary."""

    FU = "FU"
    AB = "AB"
    SP = "SP"
    ST = "ST"
    AP = "AP"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class FusionProvisionalRecord:
    """One source announcement projected onto inspectable provisional facts."""

    ref_annonce_complet: str
    publication_year: int | None
    campaign_year: int | None
    legal_family: LegalFamily
    provisional_type: ProvisionalType
    main_siren: str | None
    main_name: str | None
    previous_owner_sirens: tuple[str, ...]
    previous_owner_names: tuple[str, ...]
    participants: tuple[SemanticParticipant, ...]
    transferor_sirens: tuple[str, ...]
    beneficiary_sirens: tuple[str, ...]
    ambiguous_participant_sirens: tuple[str, ...]
    canonical_description: str | None
    description_fingerprint: str | None
    transfer_scope: TransferScope
    transferor_fate: TransferorFate
    beneficiary_creation: BeneficiaryCreation
    partial_asset_transfer_wording: PartialAssetTransferWording
    evidence: tuple[str, ...]
    reason: str
    beneficiary_link_keys: tuple[str, ...]
    transferor_link_keys: tuple[str, ...]
    self_relation: bool
    self_relation_sirens: tuple[str, ...]
    provisional_rule: str
    diagnostics: tuple[str, ...]

    @property
    def description_present(self) -> bool:
        """Whether an exact-description grouping key can be constructed."""

        return self.description_fingerprint is not None

    @property
    def description_group_key(self) -> str | None:
        """Campaign-scoped exact-description key used for sample expansion."""

        if self.campaign_year is None or self.description_fingerprint is None:
            return None
        return _campaign_group_key(
            self.campaign_year,
            "description",
            f"SHA256:{self.description_fingerprint}",
        )

    @property
    def beneficiary_group_keys(self) -> tuple[str, ...]:
        """Campaign-scoped beneficiary keys used by fusion reconciliation."""

        return _scoped_link_group_keys(
            self.campaign_year,
            "beneficiary",
            self.beneficiary_link_keys,
        )

    @property
    def transferor_group_keys(self) -> tuple[str, ...]:
        """Campaign-scoped transferor keys used by scission reconciliation."""

        return _scoped_link_group_keys(
            self.campaign_year,
            "transferor",
            self.transferor_link_keys,
        )

    @property
    def grouping_keys(self) -> tuple[str, ...]:
        """All deterministic keys through which related rows may be expanded."""

        keys: list[str] = []
        if self.description_group_key is not None:
            keys.append(self.description_group_key)
        keys.extend(self.beneficiary_group_keys)
        keys.extend(self.transferor_group_keys)
        return tuple(dict.fromkeys(keys))


@dataclass(frozen=True, slots=True)
class FusionReconciledRecord:
    """Final classification and the exact deterministic decision trace."""

    ref_annonce_complet: str
    publication_year: int | None
    campaign_year: int | None
    legal_family: LegalFamily
    provisional_type: ProvisionalType
    final_type: FinalFusionType
    main_siren: str | None
    previous_owner_sirens: tuple[str, ...]
    transferor_sirens: tuple[str, ...]
    beneficiary_sirens: tuple[str, ...]
    ambiguous_participant_sirens: tuple[str, ...]
    partial_asset_transfer_wording: PartialAssetTransferWording
    canonical_description: str | None
    description_fingerprint: str | None
    beneficiary_link_keys: tuple[str, ...]
    transferor_link_keys: tuple[str, ...]
    self_relation: bool
    self_relation_sirens: tuple[str, ...]
    reconciliation_rule: str
    reconciliation_group_keys: tuple[str, ...]
    anchor_refs: tuple[str, ...]
    changed: bool
    diagnostics: tuple[str, ...]

    @property
    def final_predicted_type(self) -> str:
        """Serialization-friendly canonical final code."""

        return self.final_type.value

    @property
    def reconciliation_group_key(self) -> str | None:
        """First matched/grouped key for compact tabular artifacts."""

        return (
            self.reconciliation_group_keys[0]
            if self.reconciliation_group_keys
            else None
        )


_WHITESPACE_RE = re.compile(r"\s+")
_YEAR_RE = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")
_COMPLETE_REFERENCE_YEAR_RE = re.compile(
    r"(?:^|_)[AB]((?:19|20)\d{2})",
    re.IGNORECASE,
)


def canonicalize_description(description: str | None) -> str | None:
    """Normalize Unicode and whitespace while retaining exact text semantics.

    Case, accents and punctuation remain significant.  This is exact grouping,
    not fuzzy or semantic clustering.
    """

    if description is None:
        return None
    if not isinstance(description, str):
        raise TypeError("description must be a string or None")
    canonical = _WHITESPACE_RE.sub(
        " ", unicodedata.normalize("NFKC", description)
    ).strip()
    return canonical or None


def description_fingerprint(description: str | None) -> str | None:
    """Return the stable SHA-256 fingerprint of a canonical description."""

    canonical = canonicalize_description(description)
    if canonical is None:
        return None
    return sha256(canonical.encode("utf-8")).hexdigest()


def _validated_siren(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    compact = re.sub(r"[.\s]", "", value.strip())
    if re.fullmatch(r"\d{9}", compact) is None:
        return None
    return (
        compact
        if extract_siren_candidates(compact) == (compact,)
        else None
    )


def _normalized_name(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    return canonicalize_description(value)


def _ordered_unique(values: Sequence[str | None]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value for value in values if value is not None))


def _source_year(value: str | None) -> int | None:
    if value is None:
        return None
    match = _YEAR_RE.search(value)
    return int(match.group(1)) if match is not None else None


def _reference_year(reference: str) -> int | None:
    match = _COMPLETE_REFERENCE_YEAR_RE.search(reference)
    return int(match.group(1)) if match is not None else None


def _canonical_announcement_description(
    announcement: NormalizedBodaccAnnouncement,
) -> str | None:
    # Generic act text is the decisive source for fusion-family notices.  The
    # RCS-B modification and sale-specific descriptions are conservative
    # fallbacks; concatenating distinct fields would stop exact duplicates from
    # sharing their historical grouping key.
    for description in (
        announcement.act_description,
        announcement.modification_description,
        announcement.sale_description,
    ):
        canonical = canonicalize_description(description)
        if canonical is not None:
            return canonical
    return None


def _validated_participants(
    participants: Sequence[SemanticParticipant],
) -> tuple[tuple[SemanticParticipant, ...], tuple[str, ...]]:
    validated: list[SemanticParticipant] = []
    diagnostics: list[str] = []
    seen: set[tuple[str | None, str | None, ParticipantRole]] = set()
    for index, participant in enumerate(participants):
        if not isinstance(participant, SemanticParticipant):
            raise TypeError("semantic participants must be SemanticParticipant")
        if not isinstance(participant.role, ParticipantRole):
            raise TypeError("semantic participant role must be ParticipantRole")

        siren = _validated_siren(participant.siren)
        name = _normalized_name(participant.name)
        if participant.siren is not None and siren is None:
            diagnostics.append(f"participant_{index}_invalid_siren_removed")
        if siren is None and name is None:
            diagnostics.append(f"participant_{index}_empty_removed")
            continue

        key = (siren, name, participant.role)
        if key in seen:
            diagnostics.append(f"participant_{index}_duplicate_removed")
            continue
        seen.add(key)
        validated.append(
            SemanticParticipant(siren=siren, name=name, role=participant.role)
        )
    return tuple(validated), tuple(diagnostics)


def _participant_sirens(
    participants: Sequence[SemanticParticipant],
    role: ParticipantRole,
) -> tuple[str, ...]:
    return _ordered_unique(
        tuple(
            participant.siren
            for participant in participants
            if participant.role is role
        )
    )


def _entity_link_keys(sirens: Sequence[str]) -> tuple[str, ...]:
    # SIRENs are the documented exact entity linkage primitive.  Names remain
    # inspectable on participants, but name-only equality is not promoted into
    # a historical reclassification rule.
    return tuple(f"SIREN:{siren}" for siren in sorted(set(sirens)))


def _campaign_group_key(year: int, role: str, key: str) -> str:
    return f"campaign={year}|{role}={key}"


def _scoped_link_group_keys(
    campaign_year: int | None,
    role: str,
    link_keys: Sequence[str],
) -> tuple[str, ...]:
    if campaign_year is None:
        return ()
    return tuple(
        _campaign_group_key(campaign_year, role, key)
        for key in sorted(set(link_keys))
    )


def _validate_semantic_result(semantic: FusionSemanticResult) -> None:
    if not isinstance(semantic, FusionSemanticResult):
        raise TypeError("semantic must be a FusionSemanticResult")
    if not isinstance(semantic.legal_family, LegalFamily):
        raise TypeError("semantic.legal_family must be a LegalFamily")
    if not isinstance(semantic.transfer_scope, TransferScope):
        raise TypeError("semantic.transfer_scope must be a TransferScope")
    if not isinstance(semantic.transferor_fate, TransferorFate):
        raise TypeError("semantic.transferor_fate must be a TransferorFate")
    if not isinstance(semantic.beneficiary_creation, BeneficiaryCreation):
        raise TypeError(
            "semantic.beneficiary_creation must be a BeneficiaryCreation"
        )
    if not isinstance(
        semantic.partial_asset_transfer_wording,
        PartialAssetTransferWording,
    ):
        raise TypeError(
            "semantic.partial_asset_transfer_wording must be a "
            "PartialAssetTransferWording"
        )
    if not isinstance(semantic.participants, tuple):
        raise TypeError("semantic.participants must be a tuple")
    if not isinstance(semantic.evidence, tuple):
        raise TypeError("semantic.evidence must be a tuple")
    if not isinstance(semantic.reason, str):
        raise TypeError("semantic.reason must be a string")


def _local_provisional_type(
    semantic: FusionSemanticResult,
    *,
    historical_self_relation: bool,
) -> tuple[ProvisionalType, str, tuple[str, ...]]:
    supported_sp_profile = (
        semantic.transfer_scope is TransferScope.PARTIAL
        and semantic.transferor_fate is TransferorFate.SURVIVES
        and semantic.beneficiary_creation is BeneficiaryCreation.NEW
    )
    if supported_sp_profile:
        return ProvisionalType.SP, "local_supported_sp_profile", ()

    supported_ap_profile = (
        semantic.partial_asset_transfer_wording
        is PartialAssetTransferWording.YES
        and semantic.transfer_scope is TransferScope.PARTIAL
        and semantic.transferor_fate is TransferorFate.SURVIVES
        and semantic.beneficiary_creation is BeneficiaryCreation.EXISTING
    )
    if supported_ap_profile:
        return ProvisionalType.AP, "local_supported_ap_profile", ()

    if semantic.legal_family is LegalFamily.FUSION:
        if historical_self_relation:
            return ProvisionalType.AB, "local_previous_owner_self_anchor", ()
        return ProvisionalType.FZ, "local_fusion_provisional", ()

    if semantic.legal_family is LegalFamily.SCISSION:
        if historical_self_relation:
            return ProvisionalType.SP, "local_previous_owner_self_anchor", ()
        return ProvisionalType.SZ, "local_scission_provisional", ()

    if (
        semantic.partial_asset_transfer_wording
        is PartialAssetTransferWording.YES
    ):
        return (
            ProvisionalType.UNKNOWN,
            "local_partial_asset_transfer_wording_unresolved",
            ("partial_asset_transfer_missing_complete_ap_profile",),
        )

    return ProvisionalType.UNKNOWN, "local_semantic_unknown", ()


def build_fusion_provisional(
    ref_annonce_complet: str,
    normalized: NormalizedBodaccAnnouncement,
    semantic: FusionSemanticResult,
) -> FusionProvisionalRecord:
    """Build one conservative provisional row from source and local semantics."""

    if not isinstance(ref_annonce_complet, str) or not ref_annonce_complet.strip():
        raise ValueError("ref_annonce_complet must be a non-empty string")
    if not isinstance(normalized, NormalizedBodaccAnnouncement):
        raise TypeError("normalized must be a NormalizedBodaccAnnouncement")
    _validate_semantic_result(semantic)

    reference = ref_annonce_complet.strip()
    diagnostics: list[str] = []

    publication_year = _source_year(normalized.publication_date)
    reference_year = _reference_year(reference)
    campaign_year = publication_year if publication_year is not None else reference_year
    if publication_year is None and reference_year is not None:
        diagnostics.append("campaign_year_derived_from_announcement_reference")
    elif publication_year is None:
        diagnostics.append("campaign_year_missing")
    elif reference_year is not None and reference_year != publication_year:
        diagnostics.append("publication_reference_year_mismatch")

    main_siren = _validated_siren(normalized.main_siren)
    if normalized.main_siren is not None and main_siren is None:
        diagnostics.append("invalid_main_siren_removed")
    main_name = _normalized_name(normalized.main_name)

    previous_owner_sirens_list: list[str] = []
    previous_owner_names_list: list[str] = []
    for index, owner in enumerate(normalized.previous_owners):
        siren = _validated_siren(owner.siren)
        name = _normalized_name(owner.name)
        if owner.siren is not None and siren is None:
            diagnostics.append(f"previous_owner_{index}_invalid_siren_removed")
        if siren is not None:
            previous_owner_sirens_list.append(siren)
        if name is not None:
            previous_owner_names_list.append(name)
    previous_owner_sirens = _ordered_unique(previous_owner_sirens_list)
    previous_owner_names = _ordered_unique(previous_owner_names_list)

    participants, participant_diagnostics = _validated_participants(
        semantic.participants
    )
    diagnostics.extend(participant_diagnostics)

    semantic_transferors = _participant_sirens(
        participants, ParticipantRole.TRANSFEROR
    )
    semantic_beneficiaries = _participant_sirens(
        participants, ParticipantRole.BENEFICIARY
    )
    ambiguous_participant_sirens = _participant_sirens(
        participants, ParticipantRole.BOTH_OR_UNCLEAR
    )

    historical_self_relation = (
        main_siren is not None and main_siren in previous_owner_sirens
    )
    beneficiary_sirens = semantic_beneficiaries
    non_beneficiary_participants = tuple(
        siren
        for siren in _ordered_unique(
            (*semantic_transferors, *ambiguous_participant_sirens)
        )
        if siren not in set(beneficiary_sirens)
    )
    if semantic.legal_family is LegalFamily.FUSION:
        # Fusion linkage is beneficiary-led. previous_owner is only the AB
        # anchor signal and must not redefine announcement participant roles.
        transferor_sirens = non_beneficiary_participants
    elif semantic.legal_family is LegalFamily.SCISSION:
        # Scission linkage is transferor-led; previous_owner is a documented
        # source signal for the ceding company in this legal family.
        transferor_sirens = _ordered_unique(
            (*previous_owner_sirens, *non_beneficiary_participants)
        )
    else:
        transferor_sirens = non_beneficiary_participants
    transferor_link_keys = _entity_link_keys(transferor_sirens)
    beneficiary_link_keys = _entity_link_keys(beneficiary_sirens)
    if not transferor_link_keys:
        diagnostics.append("transferor_linkage_missing")
    if not beneficiary_link_keys:
        diagnostics.append("beneficiary_linkage_missing")

    semantic_self_relation_sirens = set(transferor_sirens).intersection(
        beneficiary_sirens
    )
    self_relation_sirens = tuple(
        sorted(
            semantic_self_relation_sirens
            | ({main_siren} if historical_self_relation else set())
        )
    )
    if semantic_self_relation_sirens and not historical_self_relation:
        diagnostics.append("semantic_self_relation_without_previous_owner_anchor")

    provisional_type, provisional_rule, local_diagnostics = (
        _local_provisional_type(
            semantic,
            historical_self_relation=historical_self_relation,
        )
    )
    diagnostics.extend(local_diagnostics)
    if provisional_type is ProvisionalType.AB and not beneficiary_link_keys:
        diagnostics.append("ab_anchor_beneficiary_linkage_missing")
    if provisional_type is ProvisionalType.SP and not transferor_link_keys:
        diagnostics.append("sp_anchor_transferor_linkage_missing")

    canonical_description = _canonical_announcement_description(normalized)
    fingerprint = description_fingerprint(canonical_description)
    if fingerprint is None:
        diagnostics.append("description_grouping_missing")

    return FusionProvisionalRecord(
        ref_annonce_complet=reference,
        publication_year=publication_year,
        campaign_year=campaign_year,
        legal_family=semantic.legal_family,
        provisional_type=provisional_type,
        main_siren=main_siren,
        main_name=main_name,
        previous_owner_sirens=previous_owner_sirens,
        previous_owner_names=previous_owner_names,
        participants=participants,
        transferor_sirens=transferor_sirens,
        beneficiary_sirens=beneficiary_sirens,
        ambiguous_participant_sirens=ambiguous_participant_sirens,
        canonical_description=canonical_description,
        description_fingerprint=fingerprint,
        transfer_scope=semantic.transfer_scope,
        transferor_fate=semantic.transferor_fate,
        beneficiary_creation=semantic.beneficiary_creation,
        partial_asset_transfer_wording=(
            semantic.partial_asset_transfer_wording
        ),
        evidence=semantic.evidence,
        reason=semantic.reason,
        beneficiary_link_keys=beneficiary_link_keys,
        transferor_link_keys=transferor_link_keys,
        self_relation=bool(self_relation_sirens),
        self_relation_sirens=self_relation_sirens,
        provisional_rule=provisional_rule,
        diagnostics=tuple(dict.fromkeys(diagnostics)),
    )


def _anchor_index(
    records: Sequence[FusionProvisionalRecord],
    *,
    anchor_type: ProvisionalType,
    linkage_field: str,
) -> dict[tuple[int, str], tuple[str, ...]]:
    mutable: dict[tuple[int, str], list[str]] = {}
    for record in records:
        if record.provisional_type is not anchor_type:
            continue
        if record.campaign_year is None:
            continue
        for key in getattr(record, linkage_field):
            mutable.setdefault((record.campaign_year, key), []).append(
                record.ref_annonce_complet
            )
    return {
        key: tuple(sorted(set(references)))
        for key, references in mutable.items()
    }


def _matched_anchors(
    record: FusionProvisionalRecord,
    index: dict[tuple[int, str], tuple[str, ...]],
    *,
    linkage_field: str,
    role: str,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if record.campaign_year is None:
        return (), ()
    matched_references: set[str] = set()
    matched_group_keys: list[str] = []
    for key in getattr(record, linkage_field):
        references = index.get((record.campaign_year, key), ())
        if not references:
            continue
        matched_references.update(references)
        matched_group_keys.append(
            _campaign_group_key(record.campaign_year, role, key)
        )
    return (
        tuple(sorted(matched_references)),
        tuple(sorted(set(matched_group_keys))),
    )


def _sp_anchor_conflict_diagnostics(
    record: FusionProvisionalRecord,
) -> tuple[str, ...]:
    diagnostics: list[str] = []
    if record.transfer_scope is TransferScope.TOTAL:
        diagnostics.append("sp_anchor_conflicts_with_transfer_scope_total")
    if record.transferor_fate is TransferorFate.DISAPPEARS:
        diagnostics.append(
            "sp_anchor_conflicts_with_transferor_fate_disappears"
        )
    if diagnostics:
        diagnostics.insert(0, "local_global_sp_conflict")
    return tuple(diagnostics)


def _default_group_keys(record: FusionProvisionalRecord) -> tuple[str, ...]:
    if record.provisional_type in (ProvisionalType.AB, ProvisionalType.FZ):
        return record.beneficiary_group_keys
    if record.provisional_type in (ProvisionalType.SP, ProvisionalType.SZ):
        return record.transferor_group_keys
    if record.description_group_key is not None:
        return (record.description_group_key,)
    return ()


def _reconciled_record(
    provisional: FusionProvisionalRecord,
    *,
    final_type: FinalFusionType,
    rule: str,
    group_keys: tuple[str, ...],
    anchor_refs: tuple[str, ...] = (),
    diagnostics: tuple[str, ...] = (),
) -> FusionReconciledRecord:
    return FusionReconciledRecord(
        ref_annonce_complet=provisional.ref_annonce_complet,
        publication_year=provisional.publication_year,
        campaign_year=provisional.campaign_year,
        legal_family=provisional.legal_family,
        provisional_type=provisional.provisional_type,
        final_type=final_type,
        main_siren=provisional.main_siren,
        previous_owner_sirens=provisional.previous_owner_sirens,
        transferor_sirens=provisional.transferor_sirens,
        beneficiary_sirens=provisional.beneficiary_sirens,
        ambiguous_participant_sirens=(
            provisional.ambiguous_participant_sirens
        ),
        partial_asset_transfer_wording=(
            provisional.partial_asset_transfer_wording
        ),
        canonical_description=provisional.canonical_description,
        description_fingerprint=provisional.description_fingerprint,
        beneficiary_link_keys=provisional.beneficiary_link_keys,
        transferor_link_keys=provisional.transferor_link_keys,
        self_relation=provisional.self_relation,
        self_relation_sirens=provisional.self_relation_sirens,
        reconciliation_rule=rule,
        reconciliation_group_keys=group_keys,
        anchor_refs=anchor_refs,
        changed=provisional.provisional_type.value != final_type.value,
        diagnostics=tuple(
            dict.fromkeys((*provisional.diagnostics, *diagnostics))
        ),
    )


def reconcile_fusion_family(
    provisional_rows: Sequence[FusionProvisionalRecord],
) -> tuple[FusionReconciledRecord, ...]:
    """Resolve historical FZ/SZ branches in a pure campaign-global pass.

    ``AB`` anchors propagate to ``FZ`` rows sharing an exact validated
    beneficiary SIREN in the same campaign.  Remaining ``FZ`` rows become
    ``FU``.  ``SP`` anchors analogously propagate through exact transferor
    SIRENs unless local ``TOTAL`` or ``DISAPPEARS`` facts conflict; those
    rows remain ``ST`` with an explicit diagnostic. Other remaining ``SZ``
    rows become ``ST``. Source rows, including self-relations, are never
    dropped.
    """

    if not isinstance(provisional_rows, Sequence) or isinstance(
        provisional_rows, (str, bytes)
    ):
        raise TypeError("provisional_rows must be a sequence")

    rows = tuple(provisional_rows)
    references: set[str] = set()
    for record in rows:
        if not isinstance(record, FusionProvisionalRecord):
            raise TypeError(
                "provisional_rows must contain FusionProvisionalRecord"
            )
        if record.ref_annonce_complet in references:
            raise ValueError(
                "duplicate provisional announcement reference: "
                f"{record.ref_annonce_complet}"
            )
        references.add(record.ref_annonce_complet)

    sorted_rows = tuple(sorted(rows, key=lambda row: row.ref_annonce_complet))
    absorption_anchors = _anchor_index(
        sorted_rows,
        anchor_type=ProvisionalType.AB,
        linkage_field="beneficiary_link_keys",
    )
    partial_scission_anchors = _anchor_index(
        sorted_rows,
        anchor_type=ProvisionalType.SP,
        linkage_field="transferor_link_keys",
    )

    reconciled: list[FusionReconciledRecord] = []
    for row in sorted_rows:
        default_groups = _default_group_keys(row)
        if row.provisional_type is ProvisionalType.AB:
            reconciled.append(
                _reconciled_record(
                    row,
                    final_type=FinalFusionType.AB,
                    rule="local_ab_anchor_preserved",
                    group_keys=default_groups,
                )
            )
            continue

        if row.provisional_type is ProvisionalType.FZ:
            anchor_refs, matched_groups = _matched_anchors(
                row,
                absorption_anchors,
                linkage_field="beneficiary_link_keys",
                role="beneficiary",
            )
            if anchor_refs:
                reconciled.append(
                    _reconciled_record(
                        row,
                        final_type=FinalFusionType.AB,
                        rule="fz_same_beneficiary_as_ab_anchor",
                        group_keys=matched_groups,
                        anchor_refs=anchor_refs,
                    )
                )
            else:
                reconciled.append(
                    _reconciled_record(
                        row,
                        final_type=FinalFusionType.FU,
                        rule="fz_remaining_to_fu",
                        group_keys=default_groups,
                        diagnostics=(
                            "isolated_ap_fallback_intentionally_not_applied",
                        ),
                    )
                )
            continue

        if row.provisional_type is ProvisionalType.SP:
            reconciled.append(
                _reconciled_record(
                    row,
                    final_type=FinalFusionType.SP,
                    rule="local_sp_anchor_preserved",
                    group_keys=default_groups,
                )
            )
            continue

        if row.provisional_type is ProvisionalType.SZ:
            anchor_refs, matched_groups = _matched_anchors(
                row,
                partial_scission_anchors,
                linkage_field="transferor_link_keys",
                role="transferor",
            )
            conflict_diagnostics = _sp_anchor_conflict_diagnostics(row)
            if anchor_refs and conflict_diagnostics:
                reconciled.append(
                    _reconciled_record(
                        row,
                        final_type=FinalFusionType.ST,
                        rule="sz_local_semantics_override_sp_anchor_to_st",
                        group_keys=matched_groups,
                        anchor_refs=anchor_refs,
                        diagnostics=conflict_diagnostics,
                    )
                )
            elif anchor_refs:
                reconciled.append(
                    _reconciled_record(
                        row,
                        final_type=FinalFusionType.SP,
                        rule="sz_same_transferor_as_sp_anchor",
                        group_keys=matched_groups,
                        anchor_refs=anchor_refs,
                    )
                )
            else:
                reconciled.append(
                    _reconciled_record(
                        row,
                        final_type=FinalFusionType.ST,
                        rule="sz_remaining_to_st",
                        group_keys=default_groups,
                        diagnostics=(
                            "isolated_ap_fallback_intentionally_not_applied",
                        ),
                    )
                )
            continue

        if row.provisional_type is ProvisionalType.AP:
            reconciled.append(
                _reconciled_record(
                    row,
                    final_type=FinalFusionType.AP,
                    rule="local_supported_ap_preserved",
                    group_keys=default_groups,
                )
            )
            continue

        reconciled.append(
            _reconciled_record(
                row,
                final_type=FinalFusionType.UNKNOWN,
                rule="local_unknown_preserved",
                group_keys=default_groups,
            )
        )

    return tuple(reconciled)


__all__ = (
    "DESCRIPTION_FINGERPRINT_VERSION",
    "FUSION_RECONCILIATION_VERSION",
    "FinalFusionType",
    "FusionProvisionalRecord",
    "FusionReconciledRecord",
    "HISTORICAL_ISOLATED_AP_FALLBACK_IMPLEMENTED",
    "HISTORICAL_ISOLATED_AP_FALLBACK_NOTE",
    "ProvisionalType",
    "build_fusion_provisional",
    "canonicalize_description",
    "description_fingerprint",
    "reconcile_fusion_family",
)
