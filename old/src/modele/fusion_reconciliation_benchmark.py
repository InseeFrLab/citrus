"""Global benchmark for fusion-family semantic parsing and reconciliation.

The benchmark deliberately keeps reference labels outside the executable
pipeline.  ``type_op`` is used once to construct the fusion-family reference
corpus, is then split from source lookup rows, and is joined back only after
the pure reconciliation pass has completed.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Protocol

import polars as pl

from src import logger

from src.bodacc import (
    BodaccNormalizationError,
    NormalizedBodaccAnnouncement,
    normalize_bodacc_announcement,
)
from src.bodacc.api import BodaccFetchError, bodacc_api
from src.llm.client import get_model_name
from src.modele.benchmark import JOIN_KEY
from src.modele.bodacc_lookup import (
    BodaccLookupResolutionError,
    resolve_bodacc_announcement_id,
)
from src.routing.fusion_reconciliation import (
    DESCRIPTION_FINGERPRINT_VERSION,
    FUSION_RECONCILIATION_VERSION,
    HISTORICAL_ISOLATED_AP_FALLBACK_IMPLEMENTED,
    HISTORICAL_ISOLATED_AP_FALLBACK_NOTE,
    FinalFusionType,
    FusionProvisionalRecord,
    FusionReconciledRecord,
    ProvisionalType,
    build_fusion_provisional,
    description_fingerprint,
    reconcile_fusion_family,
)
from src.routing.fusion_semantics import (
    FUSION_SEMANTICS_PROMPT_VERSION,
    FUSION_SEMANTICS_SCHEMA_VERSION,
    FusionSemanticLLMError,
    FusionSemanticOutputError,
    FusionSemanticResult,
    LegalFamily,
    ParticipantRole,
    PartialAssetTransferWording,
    fusion_semantic_parser,
    fusion_semantic_source_sirens,
)
from src.routing.fusion_subtype import (
    BeneficiaryCreation,
    TransferScope,
    TransferorFate,
)


FUSION_TYPES = ("FU", "AB", "SP", "ST", "AP")
FINAL_OUTCOMES = (*FUSION_TYPES, FinalFusionType.UNKNOWN.value)
ERROR_OUTCOME = "__ERROR__"
CONFUSION_OUTCOMES = (*FINAL_OUTCOMES, ERROR_OUTCOME)
LOOKUP_COLUMNS = ("ref_annonce", "numero_annonce", JOIN_KEY)
REQUIRED_ANNOTATION_COLUMNS = (*LOOKUP_COLUMNS, "type_op")
DEFAULT_MAX_SEEDS = 5

PARTICIPANT_DTYPE = pl.Struct(
    [
        pl.Field("siren", pl.String),
        pl.Field("name", pl.String),
        pl.Field("role", pl.String),
    ]
)
SEMANTIC_SCHEMA = {
    JOIN_KEY: pl.String,
    "is_seed": pl.Boolean,
    "selection_reasons": pl.List(pl.String),
    "legal_family": pl.String,
    "transfer_scope": pl.String,
    "transferor_fate": pl.String,
    "beneficiary_creation": pl.String,
    "partial_asset_transfer_wording": pl.String,
    "participants": pl.List(PARTICIPANT_DTYPE),
    "semantically_consistent": pl.Boolean,
    "semantic_consistency_issues": pl.List(pl.String),
    "evidence": pl.List(pl.String),
    "reason": pl.String,
}
PROVISIONAL_SCHEMA = {
    JOIN_KEY: pl.String,
    "is_seed": pl.Boolean,
    "selection_reasons": pl.List(pl.String),
    "publication_year": pl.Int64,
    "campaign_year": pl.Int64,
    "legal_family": pl.String,
    "provisional_type": pl.String,
    "main_siren": pl.String,
    "main_name": pl.String,
    "previous_owner_sirens": pl.List(pl.String),
    "previous_owner_names": pl.List(pl.String),
    "participants": pl.List(PARTICIPANT_DTYPE),
    "transferor_sirens": pl.List(pl.String),
    "beneficiary_sirens": pl.List(pl.String),
    "ambiguous_participant_sirens": pl.List(pl.String),
    "canonical_description": pl.String,
    "description_fingerprint": pl.String,
    "description_group_key": pl.String,
    "beneficiary_link_keys": pl.List(pl.String),
    "transferor_link_keys": pl.List(pl.String),
    "beneficiary_group_keys": pl.List(pl.String),
    "transferor_group_keys": pl.List(pl.String),
    "transfer_scope": pl.String,
    "transferor_fate": pl.String,
    "beneficiary_creation": pl.String,
    "partial_asset_transfer_wording": pl.String,
    "self_relation": pl.Boolean,
    "self_relation_sirens": pl.List(pl.String),
    "provisional_rule": pl.String,
    "diagnostics": pl.List(pl.String),
    "evidence": pl.List(pl.String),
    "reason": pl.String,
}
RECONCILED_SCHEMA = {
    JOIN_KEY: pl.String,
    "reference_type": pl.String,
    "is_seed": pl.Boolean,
    "selection_reasons": pl.List(pl.String),
    "campaign_year": pl.Int64,
    "legal_family": pl.String,
    "partial_asset_transfer_wording": pl.String,
    "provisional_type": pl.String,
    "final_predicted_type": pl.String,
    "reconciliation_rule": pl.String,
    "reconciliation_group_key": pl.String,
    "reconciliation_group_keys": pl.List(pl.String),
    "anchor_refs": pl.List(pl.String),
    "changed": pl.Boolean,
    "correct": pl.Boolean,
    "main_siren": pl.String,
    "previous_owner_sirens": pl.List(pl.String),
    "transferor_sirens": pl.List(pl.String),
    "beneficiary_sirens": pl.List(pl.String),
    "ambiguous_participant_sirens": pl.List(pl.String),
    "description_fingerprint": pl.String,
    "beneficiary_link_keys": pl.List(pl.String),
    "transferor_link_keys": pl.List(pl.String),
    "self_relation": pl.Boolean,
    "self_relation_sirens": pl.List(pl.String),
    "diagnostics": pl.List(pl.String),
}
ERROR_SCHEMA = {
    JOIN_KEY: pl.String,
    "reference_type": pl.String,
    "bodacc_id": pl.String,
    "is_seed": pl.Boolean,
    "in_final_denominator": pl.Boolean,
    "selection_reasons": pl.List(pl.String),
    "failure_stage": pl.String,
    "failure_code": pl.String,
    "failure_reason": pl.String,
    "raw_llm_response": pl.String,
    "provisional_type": pl.String,
    "final_predicted_type": pl.String,
    "reconciliation_rule": pl.String,
    "correct": pl.Boolean,
}


class SemanticParser(Protocol):
    """Injectable announcement-level parser used by offline tests."""

    def parse(self, announcement: Mapping[str, Any]) -> FusionSemanticResult:
        ...


@dataclass(frozen=True, slots=True)
class ReferenceCorpus:
    """Label-isolated benchmark corpus created before pipeline execution."""

    source_rows: pl.DataFrame
    labels: pl.DataFrame
    total_rows_loaded: int
    supported_rows_available: int
    out_of_scope_counts: dict[str, int]
    reference_issues: tuple[dict[str, Any], ...]


@dataclass(frozen=True, slots=True)
class _DiscoveredAnnouncement:
    ref_annonce_complet: str
    bodacc_id: str
    raw: Mapping[str, Any]
    normalized: NormalizedBodaccAnnouncement
    campaign_year: int | None
    grouping_keys: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FusionReconciliationBenchmarkResult:
    """In-memory views matching the five persisted artifacts."""

    source_rows: pl.DataFrame
    reference_labels: pl.DataFrame
    seed_rows: pl.DataFrame
    expanded_source_rows: pl.DataFrame
    semantic_predictions: pl.DataFrame
    provisional: pl.DataFrame
    reconciled: pl.DataFrame
    errors: pl.DataFrame
    summary: dict[str, Any]


def _read_annotations(path: str | Path) -> pl.DataFrame:
    source = Path(path)
    if source.suffix.lower() == ".parquet":
        return pl.read_parquet(source)
    if source.suffix.lower() == ".csv":
        return pl.read_csv(source)
    raise ValueError("annotations must be a .parquet or .csv file")


def _text_key(value: Any) -> str | None:
    if value is None:
        return None
    return value if isinstance(value, str) else str(value)


def load_fusion_reconciliation_reference(
    path: str | Path,
) -> ReferenceCorpus:
    """Construct the reference scope, then separate labels from source rows."""

    frame = _read_annotations(path)
    missing = set(REQUIRED_ANNOTATION_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(
            "Missing fusion reconciliation annotation columns: "
            f"{sorted(missing)}"
        )
    projected = frame.select(REQUIRED_ANNOTATION_COLUMNS)
    in_scope = pl.col("type_op").is_in(FUSION_TYPES).fill_null(False)
    supported = projected.filter(in_scope)
    out_of_scope = projected.filter(~in_scope)
    out_of_scope_counts = Counter(
        "__NULL__" if value is None else str(value)
        for value in out_of_scope["type_op"].to_list()
    )

    normalized_key = pl.col(JOIN_KEY).cast(pl.String)
    invalid_mask = normalized_key.str.strip_chars().eq("").fill_null(True)
    invalid = supported.filter(invalid_mask)
    candidates = supported.filter(~invalid_mask).with_columns(
        normalized_key.alias(JOIN_KEY)
    )
    duplicate_counts = (
        candidates.group_by(JOIN_KEY).len().filter(pl.col("len") > 1)
    )
    duplicate_keys = duplicate_counts[JOIN_KEY].to_list()
    duplicates = candidates.filter(pl.col(JOIN_KEY).is_in(duplicate_keys))
    eligible = candidates.filter(~pl.col(JOIN_KEY).is_in(duplicate_keys))

    issues: list[dict[str, Any]] = []
    for row in invalid.iter_rows(named=True):
        issues.append(
            {
                JOIN_KEY: _text_key(row.get(JOIN_KEY)),
                "reference_type": row.get("type_op"),
                "failure_stage": "lookup_resolution",
                "failure_code": "invalid_join_key",
                "failure_reason": (
                    "ref_annonce_complet must be non-null and non-empty"
                ),
            }
        )
    duplicate_sizes = dict(duplicate_counts.iter_rows())
    for row in duplicates.iter_rows(named=True):
        key = str(row[JOIN_KEY])
        issues.append(
            {
                JOIN_KEY: key,
                "reference_type": row.get("type_op"),
                "failure_stage": "lookup_resolution",
                "failure_code": "duplicate_join_key",
                "failure_reason": (
                    f"ref_annonce_complet occurs {duplicate_sizes[key]} "
                    "times in the fusion reference corpus"
                ),
            }
        )

    source_rows = eligible.select(LOOKUP_COLUMNS).sort(JOIN_KEY)
    labels = eligible.select(
        JOIN_KEY, pl.col("type_op").alias("reference_type")
    ).sort(JOIN_KEY)
    return ReferenceCorpus(
        source_rows=source_rows,
        labels=labels,
        total_rows_loaded=frame.height,
        supported_rows_available=supported.height,
        out_of_scope_counts=dict(sorted(out_of_scope_counts.items())),
        reference_issues=tuple(issues),
    )


def select_fusion_seed_rows(
    source_rows: pl.DataFrame,
    *,
    max_seeds: int | None = DEFAULT_MAX_SEEDS,
) -> pl.DataFrame:
    """Select globally sorted seeds without accepting or consulting labels."""

    if "type_op" in source_rows.columns or "reference_type" in source_rows.columns:
        raise ValueError("seed selection accepts source lookup rows only")
    missing = set(LOOKUP_COLUMNS) - set(source_rows.columns)
    if missing:
        raise ValueError(f"Missing source lookup columns: {sorted(missing)}")
    if max_seeds is not None and max_seeds < 0:
        raise ValueError("max_seeds must be non-negative or None")
    ordered = source_rows.select(LOOKUP_COLUMNS).sort(JOIN_KEY)
    return ordered if max_seeds is None else ordered.head(max_seeds)


def _frame_from_rows(
    rows: Sequence[Mapping[str, Any]],
    schema: Mapping[str, pl.DataType],
) -> pl.DataFrame:
    return (
        pl.DataFrame(rows, schema=schema, strict=False)
        if rows
        else pl.DataFrame(schema=schema)
    )


def _technical_error(
    *,
    key: str | None,
    stage: str,
    code: str,
    reason: str,
    bodacc_id: str | None = None,
    raw_llm_response: str | None = None,
) -> dict[str, Any]:
    return {
        JOIN_KEY: key,
        "bodacc_id": bodacc_id,
        "failure_stage": stage,
        "failure_code": code,
        "failure_reason": reason,
        "raw_llm_response": raw_llm_response,
        "provisional_type": None,
        "final_predicted_type": None,
        "reconciliation_rule": None,
        "correct": None,
    }


_REFERENCE_YEAR = re.compile(r"^[ABC]((?:19|20)\d{2})\d{5,}$")
_SOURCE_DATE_YEAR = re.compile(r"^((?:19|20)\d{2})")


def _campaign_year(
    key: str,
    normalized: NormalizedBodaccAnnouncement,
) -> int | None:
    publication = normalized.publication_date
    if isinstance(publication, str):
        match = _SOURCE_DATE_YEAR.match(publication.strip())
        if match is not None:
            return int(match.group(1))
    match = _REFERENCE_YEAR.fullmatch(key)
    return int(match.group(1)) if match is not None else None


def _discovery_grouping_keys(
    key: str,
    normalized: NormalizedBodaccAnnouncement,
) -> tuple[int | None, tuple[str, ...]]:
    year = _campaign_year(key, normalized)
    if year is None:
        return None, ()
    keys: list[str] = []
    for description in normalized.descriptions:
        fingerprint = description_fingerprint(description)
        if fingerprint is not None:
            keys.append(f"campaign={year}|description=SHA256:{fingerprint}")
    for siren in fusion_semantic_source_sirens(normalized):
        keys.append(f"campaign={year}|participant=SIREN:{siren}")
    return year, tuple(dict.fromkeys(keys))


def _expanded_keys_and_reasons(
    discoveries: Mapping[str, _DiscoveredAnnouncement],
    seed_keys: Sequence[str],
    *,
    full_run: bool,
    all_source_keys: Sequence[str],
) -> tuple[tuple[str, ...], dict[str, tuple[str, ...]], dict[str, int]]:
    group_members: dict[str, set[str]] = defaultdict(set)
    for key, discovery in discoveries.items():
        for group_key in discovery.grouping_keys:
            group_members[group_key].add(key)

    adjacency: dict[str, set[str]] = defaultdict(set)
    for members in group_members.values():
        if len(members) < 2:
            continue
        ordered = sorted(members)
        first = ordered[0]
        for other in ordered[1:]:
            adjacency[first].add(other)
            adjacency[other].add(first)

    if full_run:
        reference_keys = set(all_source_keys)
        selected = set(discoveries)
        reasons = {
            key: (
                ("FULL_RUN",)
                if key in reference_keys
                else ("SOURCE_LINK_EXPANSION",)
            )
            for key in sorted(selected)
        }
    else:
        selected = set(seed_keys)
        queue = deque(sorted(seed_keys))
        while queue:
            current = queue.popleft()
            for linked in sorted(adjacency.get(current, ())):
                if linked in selected:
                    continue
                selected.add(linked)
                queue.append(linked)

        shared_groups = {
            group_key: members
            for group_key, members in group_members.items()
            if len(members) > 1 and members.intersection(selected)
        }
        reasons = {}
        seed_set = set(seed_keys)
        for key in sorted(selected):
            row_reasons: list[str] = ["SEED"] if key in seed_set else []
            row_reasons.extend(
                sorted(
                    group_key
                    for group_key, members in shared_groups.items()
                    if key in members
                )
            )
            reasons[key] = tuple(row_reasons or ("EXPANDED",))

    visited: set[str] = set()
    component_sizes: list[int] = []
    for key in sorted(discoveries):
        if key in visited:
            continue
        component: set[str] = set()
        pending = [key]
        while pending:
            item = pending.pop()
            if item in component:
                continue
            component.add(item)
            pending.extend(adjacency.get(item, ()))
        visited.update(component)
        component_sizes.append(len(component))
    return (
        tuple(sorted(selected)),
        reasons,
        {
            "component_count": len(component_sizes),
            "multi_row_component_count": sum(
                size > 1 for size in component_sizes
            ),
            "largest_component_size": max(component_sizes, default=0),
        },
    )

def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _semantic_consistency_issues(
    result: FusionSemanticResult,
) -> tuple[str, ...]:
    """Return documented cross-axis contradictions, currently none.

    In particular, SCISSION and partial-asset-transfer wording may coexist.
    """
    return ()


def _participant_rows(
    participants: Sequence[Any],
) -> list[dict[str, str | None]]:
    return [
        {
            "siren": participant.siren,
            "name": participant.name,
            "role": participant.role.value,
        }
        for participant in participants
    ]


def _semantic_artifact_row(
    key: str,
    result: FusionSemanticResult,
    *,
    is_seed: bool,
    selection_reasons: Sequence[str],
) -> dict[str, Any]:
    issues = _semantic_consistency_issues(result)
    return {
        JOIN_KEY: key,
        "is_seed": is_seed,
        "selection_reasons": list(selection_reasons),
        "legal_family": result.legal_family.value,
        "transfer_scope": result.transfer_scope.value,
        "transferor_fate": result.transferor_fate.value,
        "beneficiary_creation": result.beneficiary_creation.value,
        "partial_asset_transfer_wording": (
            result.partial_asset_transfer_wording.value
        ),
        "participants": _participant_rows(result.participants),
        "semantically_consistent": not issues,
        "semantic_consistency_issues": list(issues),
        "evidence": list(result.evidence),
        "reason": result.reason,
    }


def _provisional_artifact_row(
    record: FusionProvisionalRecord,
    *,
    is_seed: bool,
    selection_reasons: Sequence[str],
) -> dict[str, Any]:
    return {
        JOIN_KEY: record.ref_annonce_complet,
        "is_seed": is_seed,
        "selection_reasons": list(selection_reasons),
        "publication_year": record.publication_year,
        "campaign_year": record.campaign_year,
        "legal_family": record.legal_family.value,
        "provisional_type": record.provisional_type.value,
        "main_siren": record.main_siren,
        "main_name": record.main_name,
        "previous_owner_sirens": list(record.previous_owner_sirens),
        "previous_owner_names": list(record.previous_owner_names),
        "participants": _participant_rows(record.participants),
        "transferor_sirens": list(record.transferor_sirens),
        "beneficiary_sirens": list(record.beneficiary_sirens),
        "ambiguous_participant_sirens": list(
            record.ambiguous_participant_sirens
        ),
        "canonical_description": record.canonical_description,
        "description_fingerprint": record.description_fingerprint,
        "description_group_key": record.description_group_key,
        "beneficiary_link_keys": list(record.beneficiary_link_keys),
        "transferor_link_keys": list(record.transferor_link_keys),
        "beneficiary_group_keys": list(record.beneficiary_group_keys),
        "transferor_group_keys": list(record.transferor_group_keys),
        "transfer_scope": record.transfer_scope.value,
        "transferor_fate": record.transferor_fate.value,
        "beneficiary_creation": record.beneficiary_creation.value,
        "partial_asset_transfer_wording": (
            record.partial_asset_transfer_wording.value
        ),
        "self_relation": record.self_relation,
        "self_relation_sirens": list(record.self_relation_sirens),
        "provisional_rule": record.provisional_rule,
        "diagnostics": list(record.diagnostics),
        "evidence": list(record.evidence),
        "reason": record.reason,
    }


def _reconciled_artifact_row(
    record: FusionReconciledRecord,
    *,
    reference_type: str | None,
    is_seed: bool,
    selection_reasons: Sequence[str],
) -> dict[str, Any]:
    return {
        JOIN_KEY: record.ref_annonce_complet,
        "reference_type": reference_type,
        "is_seed": is_seed,
        "selection_reasons": list(selection_reasons),
        "campaign_year": record.campaign_year,
        "legal_family": record.legal_family.value,
        "partial_asset_transfer_wording": (
            record.partial_asset_transfer_wording.value
        ),
        "provisional_type": record.provisional_type.value,
        "final_predicted_type": record.final_type.value,
        "reconciliation_rule": record.reconciliation_rule,
        "reconciliation_group_key": record.reconciliation_group_key,
        "reconciliation_group_keys": list(
            record.reconciliation_group_keys
        ),
        "anchor_refs": list(record.anchor_refs),
        "changed": record.changed,
        "correct": (
            record.final_type.value == reference_type
            if reference_type is not None
            else None
        ),
        "main_siren": record.main_siren,
        "previous_owner_sirens": list(record.previous_owner_sirens),
        "transferor_sirens": list(record.transferor_sirens),
        "beneficiary_sirens": list(record.beneficiary_sirens),
        "ambiguous_participant_sirens": list(
            record.ambiguous_participant_sirens
        ),
        "description_fingerprint": record.description_fingerprint,
        "beneficiary_link_keys": list(record.beneficiary_link_keys),
        "transferor_link_keys": list(record.transferor_link_keys),
        "self_relation": record.self_relation,
        "self_relation_sirens": list(record.self_relation_sirens),
        "diagnostics": list(record.diagnostics),
    }


def _discover_source_rows(
    source_rows: pl.DataFrame,
    fetch_announcement: Callable[[str], Mapping[str, Any]],
) -> tuple[
    dict[str, _DiscoveredAnnouncement],
    list[dict[str, Any]],
]:
    discoveries: dict[str, _DiscoveredAnnouncement] = {}
    errors: list[dict[str, Any]] = []
    total = source_rows.height
    for index, source_row in enumerate(
        source_rows.iter_rows(named=True), start=1
    ):
        key = str(source_row[JOIN_KEY])
        if index == 1 or index % 25 == 0 or index == total:
            logger.info(
                "Fusion reconciliation discovery %s/%s", index, total
            )
        try:
            bodacc_id = resolve_bodacc_announcement_id(source_row)
        except BodaccLookupResolutionError as error:
            errors.append(
                _technical_error(
                    key=key,
                    stage="lookup_resolution",
                    code="unresolved_reference",
                    reason=str(error),
                )
            )
            continue
        try:
            raw = fetch_announcement(bodacc_id)
        except BodaccFetchError as error:
            errors.append(
                _technical_error(
                    key=key,
                    bodacc_id=bodacc_id,
                    stage="bodacc_fetch",
                    code=error.code,
                    reason=error.detail,
                )
            )
            continue
        except Exception as error:
            errors.append(
                _technical_error(
                    key=key,
                    bodacc_id=bodacc_id,
                    stage="bodacc_fetch",
                    code=type(error).__name__,
                    reason=(
                        "Unexpected fetch failure "
                        f"({type(error).__name__})"
                    ),
                )
            )
            continue
        if not isinstance(raw, Mapping):
            errors.append(
                _technical_error(
                    key=key,
                    bodacc_id=bodacc_id,
                    stage="bodacc_fetch",
                    code="invalid_fetch_result",
                    reason="Fetcher returned a non-mapping announcement",
                )
            )
            continue
        try:
            normalized = normalize_bodacc_announcement(raw)
        except BodaccNormalizationError as error:
            errors.append(
                _technical_error(
                    key=key,
                    bodacc_id=bodacc_id,
                    stage="source_normalization",
                    code=type(error).__name__,
                    reason=str(error),
                )
            )
            continue
        year, grouping_keys = _discovery_grouping_keys(key, normalized)
        discoveries[key] = _DiscoveredAnnouncement(
            ref_annonce_complet=key,
            bodacc_id=bodacc_id,
            raw=dict(raw),
            normalized=normalized,
            campaign_year=year,
            grouping_keys=grouping_keys,
        )
    return discoveries, errors


def _expand_linked_source_rows(
    discoveries: dict[str, _DiscoveredAnnouncement],
    search_announcements: Callable[
        [str, int], Sequence[Mapping[str, Any]]
    ],
) -> list[dict[str, Any]]:
    """Close source groups through campaign-scoped BODACC SIREN searches."""

    errors: list[dict[str, Any]] = []
    searched: set[tuple[int, str]] = set()
    pending = deque(
        sorted(
            {
                (discovery.campaign_year, siren)
                for discovery in discoveries.values()
                if discovery.campaign_year is not None
                for siren in fusion_semantic_source_sirens(
                    discovery.normalized
                )
            }
        )
    )
    while pending:
        year, siren = pending.popleft()
        if (year, siren) in searched:
            continue
        searched.add((year, siren))
        try:
            results = search_announcements(siren, year)
        except BodaccFetchError as error:
            errors.append(
                _technical_error(
                    key=f"SOURCE_SEARCH:{year}:{siren}",
                    stage="source_group_discovery",
                    code=error.code,
                    reason=error.detail,
                )
            )
            continue
        except Exception as error:
            errors.append(
                _technical_error(
                    key=f"SOURCE_SEARCH:{year}:{siren}",
                    stage="source_group_discovery",
                    code=type(error).__name__,
                    reason=(
                        "Unexpected linked-source search failure "
                        f"({type(error).__name__})"
                    ),
                )
            )
            continue
        if not isinstance(results, Sequence) or isinstance(
            results, (str, bytes)
        ):
            errors.append(
                _technical_error(
                    key=f"SOURCE_SEARCH:{year}:{siren}",
                    stage="source_group_discovery",
                    code="invalid_search_result",
                    reason="Linked-source search returned a non-sequence",
                )
            )
            continue
        ordered = sorted(
            (raw for raw in results if isinstance(raw, Mapping)),
            key=lambda raw: str(raw.get("id", "")),
        )
        for raw in ordered:
            candidate_id = raw.get("id")
            if not isinstance(candidate_id, str) or not candidate_id:
                continue
            if candidate_id in discoveries:
                continue
            try:
                normalized = normalize_bodacc_announcement(raw)
            except BodaccNormalizationError:
                continue
            candidate_year, grouping_keys = _discovery_grouping_keys(
                candidate_id, normalized
            )
            if candidate_year != year:
                continue
            candidate_keys = set(grouping_keys)
            linked = False
            for existing in discoveries.values():
                if existing.campaign_year != year:
                    continue
                shared = candidate_keys.intersection(existing.grouping_keys)
                exact_description = any(
                    "|description=" in key for key in shared
                )
                shared_participants = sum(
                    "|participant=" in key for key in shared
                )
                if exact_description or shared_participants >= 2:
                    linked = True
                    break
            if not linked:
                continue
            discoveries[candidate_id] = _DiscoveredAnnouncement(
                ref_annonce_complet=candidate_id,
                bodacc_id=candidate_id,
                raw=dict(raw),
                normalized=normalized,
                campaign_year=candidate_year,
                grouping_keys=grouping_keys,
            )
            for candidate_siren in sorted(
                fusion_semantic_source_sirens(normalized)
            ):
                if (year, candidate_siren) not in searched:
                    pending.append((year, candidate_siren))
    return errors


_EXPECTED_LEGAL_FAMILY_BY_REFERENCE = {
    "FU": LegalFamily.FUSION.value,
    "AB": LegalFamily.FUSION.value,
    "SP": LegalFamily.SCISSION.value,
    "ST": LegalFamily.SCISSION.value,
    "AP": None,
}


def _local_semantic_metrics(
    denominator_keys: Sequence[str],
    label_by_key: Mapping[str, str],
    semantic_by_key: Mapping[str, FusionSemanticResult],
    technical_errors: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    legal_families = tuple(family.value for family in LegalFamily)
    outcomes = (*legal_families, ERROR_OUTCOME)
    confusion = {
        expected: {outcome: 0 for outcome in outcomes}
        for expected in (LegalFamily.FUSION.value, LegalFamily.SCISSION.value)
    }
    correct = 0
    scored = 0
    valid = 0
    legal_family_counts: Counter[str] = Counter()
    scope_counts: Counter[str] = Counter()
    fate_counts: Counter[str] = Counter()
    creation_counts: Counter[str] = Counter()
    wording_counts: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()
    consistency_issue_counts: Counter[str] = Counter()
    participant_total = 0
    participant_siren_total = 0
    outputs_with_participants = 0
    outputs_with_participant_siren = 0
    contradictory_outputs = 0
    evidence_total = 0
    for key in denominator_keys:
        reference = label_by_key[key]
        expected = _EXPECTED_LEGAL_FAMILY_BY_REFERENCE[reference]
        result = semantic_by_key.get(key)
        predicted = ERROR_OUTCOME
        if result is not None:
            valid += 1
            predicted = result.legal_family.value
            legal_family_counts[predicted] += 1
            wording_counts[result.partial_asset_transfer_wording.value] += 1
            scope_counts[result.transfer_scope.value] += 1
            fate_counts[result.transferor_fate.value] += 1
            creation_counts[result.beneficiary_creation.value] += 1
            role_counts.update(
                participant.role.value
                for participant in result.participants
            )
            participant_total += len(result.participants)
            siren_count = sum(
                participant.siren is not None
                for participant in result.participants
            )
            participant_siren_total += siren_count
            outputs_with_participants += bool(result.participants)
            outputs_with_participant_siren += siren_count > 0
            evidence_total += len(result.evidence)
            issues = _semantic_consistency_issues(result)
            consistency_issue_counts.update(issues)
            contradictory_outputs += bool(issues)
        if expected is not None:
            scored += 1
            confusion[expected][predicted] += 1
            correct += predicted == expected

    denominator = len(denominator_keys)
    denominator_set = set(denominator_keys)
    output_validation_failures = sum(
        error.get("failure_stage") == "llm_output_validation"
        and error.get(JOIN_KEY) in denominator_set
        for error in technical_errors
    )
    formatting_failures = sum(
        error.get("failure_code") == "invalid_json"
        and error.get(JOIN_KEY) in denominator_set
        for error in technical_errors
    )
    schema_validation_failures = (
        output_validation_failures - formatting_failures
    )
    scope_unknown = scope_counts[TransferScope.UNKNOWN.value]
    fate_unknown = fate_counts[TransferorFate.UNKNOWN.value]
    creation_unknown = creation_counts[
        BeneficiaryCreation.MIXED_OR_UNKNOWN.value
    ]
    unknown = legal_family_counts[LegalFamily.UNKNOWN.value]
    wording_unknown = wording_counts[
        PartialAssetTransferWording.UNKNOWN.value
    ]
    return {
        "reference_projection": {
            "FU": "FUSION",
            "AB": "FUSION",
            "SP": "SCISSION",
            "ST": "SCISSION",
            "AP": "NOT_SCORED_FOR_LEGAL_FAMILY",
            "note": (
                "FU/AB and SP/ST are projected only after local parsing; AP "
                "has no unique legal_family projection and is not scored. "
                "Labels are never provided to the parser."
            ),
        },
        "eligible_rows": denominator,
        "valid_semantic_outputs": valid,
        "valid_semantic_output_rate": _ratio(valid, denominator),
        "technical_failures": denominator - valid,
        "technical_failure_rate": _ratio(denominator - valid, denominator),
        "formatting_failures": formatting_failures,
        "formatting_failure_rate": _ratio(formatting_failures, denominator),
        "formatting_failure_denominator": denominator,
        "output_validation_failures": output_validation_failures,
        "output_validation_failure_rate": _ratio(
            output_validation_failures, denominator
        ),
        "schema_validation_failures": schema_validation_failures,
        "schema_validation_failure_rate": _ratio(
            schema_validation_failures, denominator
        ),
        "legal_family_scored_rows": scored,
        "legal_family_correct": correct,
        "legal_family_accuracy": _ratio(correct, scored),
        "legal_family_unknown_count": unknown,
        "legal_family_unknown_rate": _ratio(unknown, valid),
        "legal_family_counts": dict(sorted(legal_family_counts.items())),
        "partial_asset_transfer_wording_counts": dict(
            sorted(wording_counts.items())
        ),
        "partial_asset_transfer_wording_unknown_count": wording_unknown,
        "partial_asset_transfer_wording_unknown_rate": _ratio(
            wording_unknown, valid
        ),
        "transfer_scope_counts": dict(sorted(scope_counts.items())),
        "transfer_scope_unknown_count": scope_unknown,
        "transfer_scope_unknown_rate": _ratio(scope_unknown, valid),
        "transferor_fate_counts": dict(sorted(fate_counts.items())),
        "transferor_fate_unknown_count": fate_unknown,
        "transferor_fate_unknown_rate": _ratio(fate_unknown, valid),
        "beneficiary_creation_counts": dict(
            sorted(creation_counts.items())
        ),
        "beneficiary_creation_unknown_count": creation_unknown,
        "beneficiary_creation_unknown_rate": _ratio(
            creation_unknown, valid
        ),
        "participant_occurrences": participant_total,
        "participants_with_siren": participant_siren_total,
        "outputs_with_participants": outputs_with_participants,
        "participant_population_rate": _ratio(
            outputs_with_participants, valid
        ),
        "outputs_with_participant_siren": outputs_with_participant_siren,
        "participant_siren_population_rate": _ratio(
            outputs_with_participant_siren, valid
        ),
        "participant_role_occurrences": dict(sorted(role_counts.items())),
        "role_population_rate": _ratio(outputs_with_participants, valid),
        "contradictory_outputs": contradictory_outputs,
        "semantic_contradiction_rate": _ratio(
            contradictory_outputs, valid
        ),
        "evidence_occurrences": evidence_total,
        "semantic_consistency_issue_occurrences": sum(
            consistency_issue_counts.values()
        ),
        "semantic_consistency_issues_by_code": dict(
            sorted(consistency_issue_counts.items())
        ),
        "legal_family_confusion_matrix": confusion,
    }


def _final_metrics(
    denominator_keys: Sequence[str],
    label_by_key: Mapping[str, str],
    reconciled_by_key: Mapping[str, FusionReconciledRecord],
) -> dict[str, Any]:
    confusion = {
        reference: {outcome: 0 for outcome in CONFUSION_OUTCOMES}
        for reference in FUSION_TYPES
    }
    correct = 0
    unknown = 0
    errors = 0
    for key in denominator_keys:
        reference = label_by_key[key]
        reconciled = reconciled_by_key.get(key)
        predicted = (
            reconciled.final_type.value
            if reconciled is not None
            else ERROR_OUTCOME
        )
        confusion[reference][predicted] += 1
        correct += predicted == reference
        unknown += predicted == FinalFusionType.UNKNOWN.value
        errors += predicted == ERROR_OUTCOME

    per_type: dict[str, dict[str, Any]] = {}
    recalls: list[float] = []
    f1_scores: list[float] = []
    for operation_type in FUSION_TYPES:
        support = sum(confusion[operation_type].values())
        type_correct = confusion[operation_type][operation_type]
        predicted_count = sum(
            confusion[reference][operation_type]
            for reference in FUSION_TYPES
        )
        recall = _ratio(type_correct, support)
        precision = _ratio(type_correct, predicted_count)
        f1 = (
            2 * type_correct / (support + predicted_count)
            if support + predicted_count
            else None
        )
        if support:
            recalls.append(recall or 0.0)
            f1_scores.append(f1 or 0.0)
        per_type[operation_type] = {
            "support": support,
            "predicted_count": predicted_count,
            "correct": type_correct,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "unknown_count": confusion[operation_type][
                FinalFusionType.UNKNOWN.value
            ],
            "technical_failure_count": confusion[operation_type][
                ERROR_OUTCOME
            ],
        }

    denominator = len(denominator_keys)
    non_unknown = denominator - unknown - errors
    return {
        "eligible_reference_rows": denominator,
        "successful_reconciled_outputs": denominator - errors,
        "technical_failures": errors,
        "correct": correct,
        "accuracy": _ratio(correct, denominator),
        "unknown_count": unknown,
        "unknown_rate": _ratio(unknown, denominator),
        "non_unknown_predictions": non_unknown,
        "non_unknown_coverage": _ratio(non_unknown, denominator),
        "selective_accuracy": _ratio(correct, non_unknown),
        "macro_recall": (
            sum(recalls) / len(recalls) if recalls else None
        ),
        "macro_f1": (
            sum(f1_scores) / len(f1_scores) if f1_scores else None
        ),
        "per_type": per_type,
        "confusion_matrix": confusion,
    }


def _transition_metrics(
    provisional_by_key: Mapping[str, FusionProvisionalRecord],
    reconciled_by_key: Mapping[str, FusionReconciledRecord],
    label_by_key: Mapping[str, str],
) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    inspectable: list[dict[str, Any]] = []
    comparable_rows = 0
    before_correct = 0
    after_correct = 0
    changed_from_default = 0
    default_projection = {
        ProvisionalType.AB: FinalFusionType.AB.value,
        ProvisionalType.FZ: FinalFusionType.FU.value,
        ProvisionalType.SP: FinalFusionType.SP.value,
        ProvisionalType.SZ: FinalFusionType.ST.value,
        ProvisionalType.AP: FinalFusionType.AP.value,
        ProvisionalType.UNKNOWN: FinalFusionType.UNKNOWN.value,
    }
    for key in sorted(provisional_by_key):
        provisional = provisional_by_key[key]
        reconciled = reconciled_by_key.get(key)
        if reconciled is None:
            continue
        transition = (
            f"{provisional.provisional_type.value}"
            f"->{reconciled.final_type.value}"
        )
        counts[transition] += 1
        reference = label_by_key.get(key)
        if reference is not None:
            before_prediction = default_projection[provisional.provisional_type]
            comparable_rows += 1
            before_correct += before_prediction == reference
            after_correct += reconciled.final_type.value == reference
            changed_from_default += (
                before_prediction != reconciled.final_type.value
            )
        if provisional.provisional_type in (
            ProvisionalType.FZ,
            ProvisionalType.SZ,
        ):
            inspectable.append(
                {
                    JOIN_KEY: key,
                    "transition": transition,
                    "reconciliation_rule": (
                        reconciled.reconciliation_rule
                    ),
                    "reconciliation_group_keys": list(
                        reconciled.reconciliation_group_keys
                    ),
                    "anchor_refs": list(reconciled.anchor_refs),
                    "diagnostics": list(reconciled.diagnostics),
                }
            )
    self_rows = [
        row for row in reconciled_by_key.values() if row.self_relation
    ]
    conflict_rows = [
        row
        for row in reconciled_by_key.values()
        if "local_global_sp_conflict" in row.diagnostics
    ]
    conflict_codes = Counter(
        diagnostic
        for row in conflict_rows
        for diagnostic in row.diagnostics
        if diagnostic.startswith("sp_anchor_conflicts_with_")
    )
    return {
        "counts": dict(sorted(counts.items())),
        "required_transition_counts": {
            transition: counts[transition]
            for transition in ("FZ->AB", "FZ->FU", "SZ->SP", "SZ->ST")
        },
        "pre_reconciliation_projection": {
            "AB": "AB",
            "FZ": "FU",
            "SP": "SP",
            "SZ": "ST",
            "AP": "AP",
            "UNKNOWN": "UNKNOWN",
        },
        "comparable_rows": comparable_rows,
        "accuracy_before_reconciliation": _ratio(
            before_correct, comparable_rows
        ),
        "accuracy_after_reconciliation": _ratio(
            after_correct, comparable_rows
        ),
        "provisional_to_final_changed_rows": sum(
            row.changed for row in reconciled_by_key.values()
        ),
        "changed_from_default_resolution_rows": changed_from_default,
        "inspectable_fz_sz_rows": inspectable,
        "self_relation_rows": len(self_rows),
        "self_relation_final_type_counts": dict(
            sorted(Counter(row.final_type.value for row in self_rows).items())
        ),
        "local_global_conflict_rows": len(conflict_rows),
        "local_global_conflicts_by_code": dict(
            sorted(conflict_codes.items())
        ),
        "source_rows_dropped_by_reconciliation": 0,
    }



def _grouping_linkage_metrics(
    provisional_by_key: Mapping[str, FusionProvisionalRecord],
) -> dict[str, Any]:
    records = list(provisional_by_key.values())
    total = len(records)
    fusion_rows = [
        row
        for row in records
        if row.provisional_type in (ProvisionalType.AB, ProvisionalType.FZ)
    ]
    scission_rows = [
        row
        for row in records
        if row.provisional_type in (ProvisionalType.SP, ProvisionalType.SZ)
    ]
    description_counts = Counter(
        row.description_group_key
        for row in records
        if row.description_group_key is not None
    )
    multi_description_keys = {
        key for key, count in description_counts.items() if count > 1
    }
    rows_in_multi_description_groups = sum(
        row.description_group_key in multi_description_keys
        for row in records
    )
    beneficiary_linked = sum(
        bool(row.beneficiary_link_keys) for row in fusion_rows
    )
    transferor_linked = sum(
        bool(row.transferor_link_keys) for row in scission_rows
    )
    rows_with_any_linkage = sum(
        bool(row.beneficiary_link_keys or row.transferor_link_keys)
        for row in records
    )
    return {
        "provisional_rows": total,
        "campaign_year_rows": sum(
            row.campaign_year is not None for row in records
        ),
        "campaign_year_coverage": _ratio(
            sum(row.campaign_year is not None for row in records), total
        ),
        "description_groupable_rows": sum(
            row.description_group_key is not None for row in records
        ),
        "description_grouping_coverage": _ratio(
            sum(row.description_group_key is not None for row in records),
            total,
        ),
        "exact_description_groups": len(description_counts),
        "multi_row_exact_description_groups": len(multi_description_keys),
        "rows_in_multi_row_exact_description_groups": (
            rows_in_multi_description_groups
        ),
        "rows_with_any_participant_linkage": rows_with_any_linkage,
        "participant_linkage_coverage": _ratio(
            rows_with_any_linkage, total
        ),
        "fusion_provisional_rows": len(fusion_rows),
        "fusion_rows_with_beneficiary_linkage": beneficiary_linked,
        "fusion_beneficiary_linkage_coverage": _ratio(
            beneficiary_linked, len(fusion_rows)
        ),
        "scission_provisional_rows": len(scission_rows),
        "scission_rows_with_transferor_linkage": transferor_linked,
        "scission_transferor_linkage_coverage": _ratio(
            transferor_linked, len(scission_rows)
        ),
        "ab_anchor_rows": sum(
            row.provisional_type is ProvisionalType.AB for row in records
        ),
        "sp_anchor_rows": sum(
            row.provisional_type is ProvisionalType.SP for row in records
        ),
    }
def _current_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _failure_summary(
    errors: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_stage = Counter(
        error["failure_stage"] for error in errors
    )
    by_code = Counter(
        error["failure_code"] for error in errors
    )
    technical = sum(
        error["failure_stage"] != "final_evaluation"
        for error in errors
    )
    return {
        "total_error_rows": len(errors),
        "technical_error_rows": technical,
        "final_evaluation_error_rows": len(errors) - technical,
        "by_stage": dict(sorted(by_stage.items())),
        "by_code": dict(sorted(by_code.items())),
    }


def _write_outputs(
    output_dir: Path,
    semantic_predictions: pl.DataFrame,
    provisional: pl.DataFrame,
    reconciled: pl.DataFrame,
    errors: pl.DataFrame,
    summary: Mapping[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    semantic_predictions.write_parquet(
        output_dir / "fusion_semantic_predictions.parquet"
    )
    provisional.write_parquet(
        output_dir / "fusion_provisional.parquet"
    )
    reconciled.write_parquet(
        output_dir / "fusion_reconciled.parquet"
    )
    errors.write_parquet(
        output_dir / "fusion_reconciliation_errors.parquet"
    )
    with (output_dir / "fusion_reconciliation_summary.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def run_fusion_reconciliation_benchmark(
    annotations_path: str | Path,
    output_dir: str | Path,
    *,
    max_seeds: int | None = DEFAULT_MAX_SEEDS,
    fetch_announcement: Callable[[str], Mapping[str, Any]] | None = None,
    search_linked_announcements: Callable[
        [str, int], Sequence[Mapping[str, Any]]
    ] | None = None,
    semantic_parser: SemanticParser | None = None,
    run_timestamp: datetime | None = None,
    git_commit: str | None = None,
    model_name: str | None = None,
) -> FusionReconciliationBenchmarkResult:
    """Run local semantics then label-free deterministic reconciliation."""

    corpus = load_fusion_reconciliation_reference(annotations_path)
    full_run = max_seeds is None
    if full_run:
        seed_rows = corpus.source_rows.head(0)
    else:
        seed_rows = select_fusion_seed_rows(
            corpus.source_rows, max_seeds=max_seeds
        )
    seed_keys = tuple(str(value) for value in seed_rows[JOIN_KEY].to_list())
    seed_set = set(seed_keys)

    api_client = bodacc_api()
    fetch = fetch_announcement or api_client.fetch_annonce_json
    discoveries, discovery_errors = _discover_source_rows(
        corpus.source_rows, fetch
    )
    search = search_linked_announcements
    if search is None and fetch_announcement is None:
        search = api_client.search_acte_siren_for_year
    if search is not None:
        discovery_errors.extend(
            _expand_linked_source_rows(discoveries, search)
        )
    all_source_keys = tuple(
        str(value) for value in corpus.source_rows[JOIN_KEY].to_list()
    )
    expanded_keys, selection_reasons, component_metrics = (
        _expanded_keys_and_reasons(
            discoveries,
            seed_keys,
            full_run=full_run,
            all_source_keys=all_source_keys,
        )
    )
    expanded_discovered_keys = tuple(
        key for key in expanded_keys if key in discoveries
    )
    expanded_source_rows = pl.DataFrame(
        [
            {"ref_annonce": key, "numero_annonce": None, JOIN_KEY: key}
            for key in expanded_discovered_keys
        ],
        schema=corpus.source_rows.schema,
        strict=False,
    ).sort(JOIN_KEY)
    if full_run:
        denominator_keys = all_source_keys
    else:
        denominator_keys = tuple(
            sorted(
                set(expanded_discovered_keys).intersection(all_source_keys)
                .union(seed_set)
            )
        )

    active_parser = semantic_parser or fusion_semantic_parser
    semantic_rows: list[dict[str, Any]] = []
    semantic_by_key: dict[str, FusionSemanticResult] = {}
    provisional_rows: list[dict[str, Any]] = []
    provisional_by_key: dict[str, FusionProvisionalRecord] = {}
    pipeline_errors: list[dict[str, Any]] = []
    total = len(expanded_discovered_keys)
    for index, key in enumerate(expanded_discovered_keys, start=1):
        discovery = discoveries[key]
        reasons = selection_reasons[key]
        if index == 1 or index % 25 == 0 or index == total:
            logger.info(
                "Fusion semantic parsing %s/%s", index, total
            )
        try:
            semantic = active_parser.parse(discovery.raw)
            if not isinstance(semantic, FusionSemanticResult):
                raise FusionSemanticOutputError(
                    "invalid_parser_result",
                    "Parser returned an unvalidated result object",
                )
        except FusionSemanticOutputError as error:
            pipeline_errors.append(
                _technical_error(
                    key=key,
                    bodacc_id=discovery.bodacc_id,
                    stage="llm_output_validation",
                    code=error.code,
                    reason=error.detail,
                    raw_llm_response=error.raw_response,
                )
            )
            continue
        except FusionSemanticLLMError as error:
            pipeline_errors.append(
                _technical_error(
                    key=key,
                    bodacc_id=discovery.bodacc_id,
                    stage="llm_execution",
                    code=error.code,
                    reason=error.detail,
                    raw_llm_response=error.raw_response,
                )
            )
            continue
        except BodaccNormalizationError as error:
            pipeline_errors.append(
                _technical_error(
                    key=key,
                    bodacc_id=discovery.bodacc_id,
                    stage="parser_preparation",
                    code=type(error).__name__,
                    reason=str(error),
                )
            )
            continue
        except Exception as error:
            pipeline_errors.append(
                _technical_error(
                    key=key,
                    bodacc_id=discovery.bodacc_id,
                    stage="parser_execution",
                    code=type(error).__name__,
                    reason=(
                        "Unexpected parser failure "
                        f"({type(error).__name__})"
                    ),
                )
            )
            continue

        semantic_by_key[key] = semantic
        semantic_rows.append(
            _semantic_artifact_row(
                key,
                semantic,
                is_seed=key in seed_set,
                selection_reasons=reasons,
            )
        )
        try:
            provisional = build_fusion_provisional(
                key, discovery.normalized, semantic
            )
        except Exception as error:
            pipeline_errors.append(
                _technical_error(
                    key=key,
                    bodacc_id=discovery.bodacc_id,
                    stage="provisional_construction",
                    code=type(error).__name__,
                    reason=(
                        "Unexpected provisional construction failure "
                        f"({type(error).__name__})"
                    ),
                )
            )
            continue
        provisional_by_key[key] = provisional
        provisional_rows.append(
            _provisional_artifact_row(
                provisional,
                is_seed=key in seed_set,
                selection_reasons=reasons,
            )
        )

    reconciled_records = reconcile_fusion_family(
        tuple(provisional_by_key.values())
    )
    reconciled_by_key = {
        row.ref_annonce_complet: row for row in reconciled_records
    }

    # Reference labels first re-enter here, after every executable decision.
    label_by_key = {
        str(row[JOIN_KEY]): str(row["reference_type"])
        for row in corpus.labels.iter_rows(named=True)
    }
    reconciled_rows = [
        _reconciled_artifact_row(
            record,
            reference_type=label_by_key.get(record.ref_annonce_complet),
            is_seed=record.ref_annonce_complet in seed_set,
            selection_reasons=selection_reasons[
                record.ref_annonce_complet
            ],
        )
        for record in reconciled_records
    ]

    all_errors: list[dict[str, Any]] = []
    for issue in corpus.reference_issues:
        all_errors.append(
            {
                **_technical_error(
                    key=_text_key(issue.get(JOIN_KEY)),
                    stage=str(issue["failure_stage"]),
                    code=str(issue["failure_code"]),
                    reason=str(issue["failure_reason"]),
                ),
                "reference_type": issue.get("reference_type"),
                "is_seed": False,
                "in_final_denominator": False,
                "selection_reasons": [],
            }
        )

    for error in (*discovery_errors, *pipeline_errors):
        key = _text_key(error.get(JOIN_KEY))
        in_denominator = key in set(denominator_keys)
        reasons = (
            selection_reasons.get(key, ())
            if key is not None
            else ()
        )
        if key in seed_set and not reasons:
            reasons = ("SEED",)
        elif full_run and key is not None and not reasons:
            reasons = ("FULL_RUN",)
        all_errors.append(
            {
                **error,
                "reference_type": (
                    label_by_key.get(key) if key is not None else None
                ),
                "is_seed": key in seed_set,
                "in_final_denominator": in_denominator,
                "selection_reasons": list(reasons),
            }
        )

    for record in reconciled_records:
        reference = label_by_key.get(record.ref_annonce_complet)
        if reference is None:
            continue
        if record.final_type.value == reference:
            continue
        all_errors.append(
            {
                JOIN_KEY: record.ref_annonce_complet,
                "reference_type": reference,
                "bodacc_id": discoveries[
                    record.ref_annonce_complet
                ].bodacc_id,
                "is_seed": record.ref_annonce_complet in seed_set,
                "in_final_denominator": True,
                "selection_reasons": list(
                    selection_reasons[record.ref_annonce_complet]
                ),
                "failure_stage": "final_evaluation",
                "failure_code": (
                    "final_unknown"
                    if record.final_type is FinalFusionType.UNKNOWN
                    else "final_misclassification"
                ),
                "failure_reason": (
                    "Reconciled output explicitly preserved UNKNOWN"
                    if record.final_type is FinalFusionType.UNKNOWN
                    else (
                        "Reconciled fusion type differs from the "
                        "reference type"
                    )
                ),
                "raw_llm_response": None,
                "provisional_type": record.provisional_type.value,
                "final_predicted_type": record.final_type.value,
                "reconciliation_rule": record.reconciliation_rule,
                "correct": False,
            }
        )

    semantic_predictions = _frame_from_rows(
        semantic_rows, SEMANTIC_SCHEMA
    ).sort(JOIN_KEY)
    provisional_frame = _frame_from_rows(
        provisional_rows, PROVISIONAL_SCHEMA
    ).sort(JOIN_KEY)
    reconciled_frame = _frame_from_rows(
        reconciled_rows, RECONCILED_SCHEMA
    ).sort(JOIN_KEY)
    errors_frame = _frame_from_rows(
        all_errors, ERROR_SCHEMA
    ).sort(JOIN_KEY, "failure_stage", "failure_code")

    timestamp = run_timestamp or datetime.now(UTC)
    final_metrics = _final_metrics(
        denominator_keys, label_by_key, reconciled_by_key
    )
    summary = {
        "metadata": {
            "run_timestamp": timestamp.astimezone(UTC).isoformat(),
            "git_commit": (
                git_commit if git_commit is not None else _current_git_commit()
            ),
            "annotations_file": Path(annotations_path).name,
            "full_run": full_run,
            "benchmark_authority": (
                "authoritative" if full_run else "diagnostic_sample"
            ),
            "max_seeds": max_seeds,
            "selection_mode": (
                "full_global_reference"
                if full_run
                else (
                    "global_sorted_seeds_with_source_graph_"
                    "transitive_expansion"
                )
            ),
            "llm_model_name": model_name or get_model_name(),
            "fusion_semantics_prompt_version": (
                FUSION_SEMANTICS_PROMPT_VERSION
            ),
            "fusion_semantics_schema_version": (
                FUSION_SEMANTICS_SCHEMA_VERSION
            ),
            "fusion_reconciliation_version": (
                FUSION_RECONCILIATION_VERSION
            ),
            "description_fingerprint_version": (
                DESCRIPTION_FINGERPRINT_VERSION
            ),
            "historical_isolated_ap_fallback_implemented": (
                HISTORICAL_ISOLATED_AP_FALLBACK_IMPLEMENTED
            ),
            "historical_isolated_ap_fallback_note": (
                HISTORICAL_ISOLATED_AP_FALLBACK_NOTE
            ),
            "label_isolation": (
                "type_op defines the reference cohort, is split before "
                "fetching/parsing/grouping/provisional/reconciliation, "
                "and is rejoined only for evaluation."
            ),
        },
        "coverage": {
            "total_annotation_rows_loaded": corpus.total_rows_loaded,
            "supported_reference_rows_available": (
                corpus.supported_rows_available
            ),
            "eligible_unique_source_rows": corpus.source_rows.height,
            "out_of_scope_counts_by_type": corpus.out_of_scope_counts,
            "reference_issue_rows": len(corpus.reference_issues),
            "source_rows_discovered": len(discoveries),
            "source_discovery_failures": len(discovery_errors),
            "seed_rows": len(seed_keys),
            "expanded_rows": len(expanded_discovered_keys),
            "rows_added_by_group_expansion": max(
                0,
                len(expanded_discovered_keys)
                - (
                    corpus.source_rows.height
                    if full_run
                    else len(seed_keys)
                ),
            ),
            "final_denominator_rows": len(denominator_keys),
            "valid_semantic_outputs": len(semantic_by_key),
            "provisional_rows": len(provisional_by_key),
            "reconciled_rows": len(reconciled_by_key),
        },
        "sampling": {
            "full_run": full_run,
            "seed_keys": list(seed_keys),
            "expanded_keys": list(expanded_discovered_keys),
            "group_discovery": component_metrics,
            "discovery_uses_labels": False,
            "grouping_primitives": [
                "campaign-scoped exact normalized-description SHA-256",
                "campaign-scoped validated source SIREN",
            ],
            "discovery_failures_may_hide_links": bool(discovery_errors),
        },
        "failures": _failure_summary(all_errors),
        "metrics": {
            "local_semantics": _local_semantic_metrics(
                denominator_keys, label_by_key, semantic_by_key, all_errors
            ),
            "final_classification": final_metrics,
            "transitions": _transition_metrics(
                provisional_by_key, reconciled_by_key, label_by_key
            ),
            "grouping_linkage": _grouping_linkage_metrics(
                provisional_by_key
            ),
        },
        "confusion_matrix": final_metrics["confusion_matrix"],
        "artifacts": {
            "semantic_predictions": (
                "fusion_semantic_predictions.parquet"
            ),
            "provisional": "fusion_provisional.parquet",
            "reconciled": "fusion_reconciled.parquet",
            "errors": "fusion_reconciliation_errors.parquet",
            "summary": "fusion_reconciliation_summary.json",
        },
    }
    _write_outputs(
        Path(output_dir),
        semantic_predictions,
        provisional_frame,
        reconciled_frame,
        errors_frame,
        summary,
    )
    return FusionReconciliationBenchmarkResult(
        source_rows=corpus.source_rows,
        reference_labels=corpus.labels,
        seed_rows=seed_rows,
        expanded_source_rows=expanded_source_rows,
        semantic_predictions=semantic_predictions,
        provisional=provisional_frame,
        reconciled=reconciled_frame,
        errors=errors_frame,
        summary=summary,
    )


def _non_negative_integer(value: str) -> int:
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "must be a non-negative integer"
        ) from error
    if result < 0:
        raise argparse.ArgumentTypeError(
            "must be a non-negative integer"
        )
    return result


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark announcement-level fusion semantics and deterministic "
            "global reconciliation without extraction"
        )
    )
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--max-seeds",
        type=_non_negative_integer,
        default=DEFAULT_MAX_SEEDS,
        help=(
            "number of globally sorted source seeds before deterministic "
            "linked-group expansion (default: 5)"
        ),
    )
    selection.add_argument(
        "--all",
        action="store_true",
        help="run the authoritative benchmark on the full reference cohort",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    run_fusion_reconciliation_benchmark(
        args.annotations,
        args.output_dir,
        max_seeds=None if args.all else args.max_seeds,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
