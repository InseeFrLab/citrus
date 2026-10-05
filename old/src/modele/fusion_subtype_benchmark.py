"""Real-data benchmark for the dedicated fusion-family subtype router."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import subprocess
from typing import Any, Protocol

import polars as pl

from src.bodacc import BodaccNormalizationError
from src.bodacc.api import BodaccFetchError, bodacc_api
from src.llm.client import get_model_name
from src.modele.benchmark import JOIN_KEY
from src.modele.bodacc_lookup import (
    BodaccLookupResolutionError,
    resolve_bodacc_announcement_id,
)
from src.routing.fusion_prompt import FUSION_SUBTYPE_PROMPT_VERSION
from src.routing.fusion_subtype import (
    FUSION_SUBTYPE_TAXONOMY_VERSION,
    FusionSubtype,
    FusionSubtypeLLMError,
    FusionSubtypeOutputError,
    FusionSubtypeResult,
    fusion_subtype_router,
    semantic_consistency_issues,
)


FUSION_TYPES = ("FU", "AB", "SP", "ST", "AP")
PREDICTION_OUTCOMES = (*FUSION_TYPES, FusionSubtype.UNKNOWN.value)
ERROR_OUTCOME = "__ERROR__"
CONFUSION_OUTCOMES = (*PREDICTION_OUTCOMES, ERROR_OUTCOME)
ANNOTATION_COLUMNS = (
    "ref_annonce",
    "numero_annonce",
    JOIN_KEY,
    "type_op",
)
PREDICTION_SCHEMA = {
    JOIN_KEY: pl.String,
    "reference_type": pl.String,
    "predicted_type": pl.String,
    "correct": pl.Boolean,
    "transfer_scope": pl.String,
    "transferor_fate": pl.String,
    "beneficiary_creation": pl.String,
    "beneficiary_count": pl.String,
    "semantically_consistent": pl.Boolean,
    "semantic_consistency_issues": pl.List(pl.String),
    "reason": pl.String,
    "evidence": pl.List(pl.String),
}
ERROR_SCHEMA = {
    JOIN_KEY: pl.String,
    "reference_type": pl.String,
    "predicted_type": pl.String,
    "bodacc_id": pl.String,
    "failure_stage": pl.String,
    "failure_code": pl.String,
    "failure_reason": pl.String,
    "correct": pl.Boolean,
    "transfer_scope": pl.String,
    "transferor_fate": pl.String,
    "beneficiary_creation": pl.String,
    "beneficiary_count": pl.String,
    "semantically_consistent": pl.Boolean,
    "semantic_consistency_issues": pl.List(pl.String),
    "reason": pl.String,
    "evidence": pl.List(pl.String),
}
ROUTER_FAILURE_STAGES = (
    "router_preparation",
    "llm_execution",
    "llm_output_validation",
    "router_execution",
)


class FusionRouter(Protocol):
    """Injectable boundary used by the offline and real-data runners."""

    def route(self, announcement: Mapping[str, Any]) -> FusionSubtypeResult:
        ...


@dataclass(frozen=True)
class FusionSubtypeBenchmarkResult:
    """In-memory results matching the three persisted artifacts."""

    annotations: pl.DataFrame
    selected_annotations: pl.DataFrame
    benchmark_annotations: pl.DataFrame
    predictions: pl.DataFrame
    errors: pl.DataFrame
    summary: dict[str, Any]


def reference_subtype_for_type(operation_type: Any) -> FusionSubtype:
    """Return one exact in-scope final type without coercion."""

    if not isinstance(operation_type, str):
        raise ValueError("reference type must be an in-scope non-empty string")
    if operation_type not in FUSION_TYPES:
        raise ValueError(f"unsupported fusion reference type: {operation_type!r}")
    return FusionSubtype(operation_type)


def load_fusion_annotations(path: str | Path) -> pl.DataFrame:
    """Load and immediately project lookup fields plus the reference type."""

    source = Path(path)
    if source.suffix.lower() == ".parquet":
        frame = pl.read_parquet(source)
    elif source.suffix.lower() == ".csv":
        frame = pl.read_csv(source)
    else:
        raise ValueError("annotations must be a .parquet or .csv file")
    missing = set(ANNOTATION_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(
            f"Missing fusion benchmark annotation columns: {sorted(missing)}"
        )
    return frame.select(ANNOTATION_COLUMNS)


def select_fusion_annotations(
    annotations: pl.DataFrame,
    *,
    max_per_type: int | None = 5,
) -> pl.DataFrame:
    """Sample deterministically within each in-scope final type."""

    if max_per_type is not None and max_per_type < 0:
        raise ValueError("max_per_type must be non-negative or None")
    selected: list[pl.DataFrame] = []
    for operation_type in FUSION_TYPES:
        type_rows = annotations.filter(
            pl.col("type_op") == operation_type
        ).sort(JOIN_KEY)
        if max_per_type is not None:
            type_rows = type_rows.head(max_per_type)
        selected.append(type_rows)
    return pl.concat(selected, how="vertical_relaxed")


def _out_of_scope_rows(annotations: pl.DataFrame) -> pl.DataFrame:
    in_scope = pl.col("type_op").is_in(FUSION_TYPES).fill_null(False)
    return annotations.filter(~in_scope)


def _string_value(value: Any) -> str | None:
    if value is None:
        return None
    return value if isinstance(value, str) else str(value)


def _technical_error(
    *,
    annotation: Mapping[str, Any],
    stage: str,
    code: str,
    reason: str,
    bodacc_id: str | None = None,
) -> dict[str, Any]:
    operation_type = annotation.get("type_op")
    return {
        JOIN_KEY: _string_value(annotation.get(JOIN_KEY)),
        "reference_type": (
            operation_type if isinstance(operation_type, str) else None
        ),
        "predicted_type": None,
        "bodacc_id": bodacc_id,
        "failure_stage": stage,
        "failure_code": code,
        "failure_reason": reason,
        "correct": None,
        "transfer_scope": None,
        "transferor_fate": None,
        "beneficiary_creation": None,
        "beneficiary_count": None,
        "semantically_consistent": None,
        "semantic_consistency_issues": None,
        "reason": None,
        "evidence": None,
    }


def _partition_benchmark_annotations(
    selected: pl.DataFrame,
) -> tuple[pl.DataFrame, list[dict[str, Any]]]:
    """Exclude null/blank keys and every selected duplicate occurrence."""

    normalized_key = pl.col(JOIN_KEY).cast(pl.String)
    invalid_key = normalized_key.str.strip_chars().eq("").fill_null(True)
    invalid_rows = selected.filter(invalid_key)
    candidates = selected.filter(~invalid_key).with_columns(
        normalized_key.alias("_benchmark_join_key")
    )
    duplicate_counts = (
        candidates.group_by("_benchmark_join_key")
        .len()
        .filter(pl.col("len") > 1)
    )
    duplicate_keys = duplicate_counts["_benchmark_join_key"].to_list()
    duplicate_rows = candidates.filter(
        pl.col("_benchmark_join_key").is_in(duplicate_keys)
    )
    eligible = candidates.filter(
        ~pl.col("_benchmark_join_key").is_in(duplicate_keys)
    ).drop("_benchmark_join_key")

    errors: list[dict[str, Any]] = []
    for row in invalid_rows.iter_rows(named=True):
        errors.append(
            _technical_error(
                annotation=row,
                stage="lookup_resolution",
                code="invalid_join_key",
                reason="ref_annonce_complet must be non-null and non-empty",
            )
        )
    duplicate_sizes = dict(
        duplicate_counts.select("_benchmark_join_key", "len").iter_rows()
    )
    for row in duplicate_rows.iter_rows(named=True):
        key = str(row["_benchmark_join_key"])
        errors.append(
            _technical_error(
                annotation=row,
                stage="lookup_resolution",
                code="duplicate_join_key",
                reason=(
                    f"ref_annonce_complet occurs {duplicate_sizes[key]} times "
                    "in the selected sample"
                ),
            )
        )
    return eligible, errors


def _frame_from_rows(
    rows: Sequence[Mapping[str, Any]], schema: Mapping[str, pl.DataType]
) -> pl.DataFrame:
    return (
        pl.DataFrame(rows, schema=schema, strict=False)
        if rows
        else pl.DataFrame(schema=schema)
    )


def _prediction_by_key(
    predictions: pl.DataFrame,
) -> dict[str, Mapping[str, Any]]:
    by_key: dict[str, Mapping[str, Any]] = {}
    for row in predictions.iter_rows(named=True):
        key = row[JOIN_KEY]
        if key in by_key:
            raise ValueError(f"duplicate fusion prediction key: {key}")
        predicted = row["predicted_type"]
        if predicted not in PREDICTION_OUTCOMES:
            raise ValueError(f"unsupported predicted subtype: {predicted!r}")
        by_key[key] = row
    return by_key


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _semantic_consistency_metrics(
    predictions: pl.DataFrame,
) -> dict[str, Any]:
    rows = list(predictions.iter_rows(named=True))
    consistent_rows = [
        row for row in rows if bool(row["semantically_consistent"])
    ]
    consistent_count = len(consistent_rows)
    inconsistent_count = len(rows) - consistent_count
    correct_consistent_count = sum(
        bool(row["correct"]) for row in consistent_rows
    )
    issue_counts: Counter[str] = Counter()
    axis_counts: Counter[str] = Counter()
    for row in rows:
        for issue in row["semantic_consistency_issues"] or []:
            issue_counts[issue] += 1
            axis_counts[issue.partition(":")[0]] += 1

    def breakdown(field: str, labels: Sequence[str]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for label in labels:
            label_rows = [row for row in rows if row[field] == label]
            label_consistent = sum(
                bool(row["semantically_consistent"]) for row in label_rows
            )
            result[label] = {
                "evaluated_outputs": len(label_rows),
                "consistent_count": label_consistent,
                "inconsistent_count": len(label_rows) - label_consistent,
                "consistency_rate": _ratio(label_consistent, len(label_rows)),
            }
        return result

    return {
        "evaluated_outputs": len(rows),
        "semantically_consistent_outputs": consistent_count,
        "inconsistent_valid_outputs": inconsistent_count,
        "consistent_count": consistent_count,
        "inconsistent_count": inconsistent_count,
        "consistency_rate": _ratio(consistent_count, len(rows)),
        "inconsistency_rate": _ratio(inconsistent_count, len(rows)),
        "correct_consistent_outputs": correct_consistent_count,
        "accuracy_conditional_on_consistent_outputs": _ratio(
            correct_consistent_count, consistent_count
        ),
        "issue_occurrences": sum(issue_counts.values()),
        "issues_by_code": dict(sorted(issue_counts.items())),
        "issues_by_axis": dict(sorted(axis_counts.items())),
        "by_predicted_type": breakdown(
            "predicted_type", PREDICTION_OUTCOMES
        ),
        "by_reference_type": breakdown("reference_type", FUSION_TYPES),
    }


def summarize_fusion_subtype_metrics(
    benchmark_annotations: pl.DataFrame,
    predictions: pl.DataFrame,
    *,
    technical_router_failures: int | None = None,
) -> dict[str, Any]:
    """Compute subtype, abstention, technical, and consistency metrics."""

    predicted_by_key = _prediction_by_key(predictions)
    rows: list[dict[str, str]] = []
    for annotation in benchmark_annotations.iter_rows(named=True):
        key = str(annotation[JOIN_KEY])
        reference = reference_subtype_for_type(annotation["type_op"]).value
        prediction = predicted_by_key.get(key)
        predicted = (
            prediction["predicted_type"]
            if prediction is not None
            else ERROR_OUTCOME
        )
        rows.append(
            {
                "reference_type": reference,
                "predicted_type": predicted,
            }
        )

    eligible = len(rows)
    successful = predictions.height
    technical_failures = eligible - successful
    if technical_failures < 0:
        raise ValueError("fusion predictions exceed eligible reference rows")
    unknown_count = sum(
        row["predicted_type"] == FusionSubtype.UNKNOWN.value for row in rows
    )
    non_unknown_count = sum(
        row["predicted_type"]
        not in (FusionSubtype.UNKNOWN.value, ERROR_OUTCOME)
        for row in rows
    )
    correct = sum(
        row["reference_type"] == row["predicted_type"] for row in rows
    )

    confusion = {
        reference: {outcome: 0 for outcome in CONFUSION_OUTCOMES}
        for reference in FUSION_TYPES
    }
    for row in rows:
        confusion[row["reference_type"]][row["predicted_type"]] += 1

    fu_as_ab = confusion["FU"]["AB"]
    ab_as_fu = confusion["AB"]["FU"]
    sp_as_ap = confusion["SP"]["AP"]
    ap_as_sp = confusion["AP"]["SP"]
    st_as_sp = confusion["ST"]["SP"]
    st_as_ap = confusion["ST"]["AP"]
    sp_as_st = confusion["SP"]["ST"]
    ap_as_st = confusion["AP"]["ST"]
    priority_confusions = {
        "FU_AB": {
            "FU_as_AB": fu_as_ab,
            "AB_as_FU": ab_as_fu,
            "total": fu_as_ab + ab_as_fu,
        },
        "SP_AP": {
            "SP_as_AP": sp_as_ap,
            "AP_as_SP": ap_as_sp,
            "total": sp_as_ap + ap_as_sp,
        },
        "ST_SP_AP": {
            "ST_as_SP": st_as_sp,
            "ST_as_AP": st_as_ap,
            "SP_as_ST": sp_as_st,
            "AP_as_ST": ap_as_st,
            "total": st_as_sp + st_as_ap + sp_as_st + ap_as_st,
        },
    }

    per_type: dict[str, dict[str, Any]] = {}
    recalls: list[float] = []
    f1_scores: list[float] = []
    for operation_type in FUSION_TYPES:
        reference_rows = [
            row for row in rows if row["reference_type"] == operation_type
        ]
        support = len(reference_rows)
        type_correct = sum(
            row["predicted_type"] == operation_type
            for row in reference_rows
        )
        predicted_total = sum(
            row["predicted_type"] == operation_type for row in rows
        )
        type_unknown = sum(
            row["predicted_type"] == FusionSubtype.UNKNOWN.value
            for row in reference_rows
        )
        type_errors = sum(
            row["predicted_type"] == ERROR_OUTCOME
            for row in reference_rows
        )
        recall = _ratio(type_correct, support)
        precision = _ratio(type_correct, predicted_total)
        f1 = (
            2 * type_correct / (support + predicted_total)
            if support + predicted_total
            else None
        )
        if support:
            recalls.append(recall if recall is not None else 0.0)
            f1_scores.append(f1 if f1 is not None else 0.0)
        per_type[operation_type] = {
            "support": support,
            "predicted_count": predicted_total,
            "correct": type_correct,
            "accuracy": recall,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "predicted_unknown_count": type_unknown,
            "predicted_unknown_rate": _ratio(type_unknown, support),
            "technical_failure_count": type_errors,
        }

    return {
        "eligible_reference_rows": eligible,
        "successful_valid_router_outputs": successful,
        "technical_failures": technical_failures,
        "technical_router_failures": (
            technical_failures
            if technical_router_failures is None
            else technical_router_failures
        ),
        "correct": correct,
        "accuracy": _ratio(correct, eligible),
        "unknown_count": unknown_count,
        "unknown_rate": _ratio(unknown_count, eligible),
        "non_unknown_predictions": non_unknown_count,
        "non_unknown_coverage": _ratio(non_unknown_count, eligible),
        "coverage": _ratio(non_unknown_count, eligible),
        "selective_accuracy": _ratio(correct, non_unknown_count),
        "macro_recall": sum(recalls) / len(recalls) if recalls else None,
        "macro_f1": sum(f1_scores) / len(f1_scores) if f1_scores else None,
        "macro_supported_type_count": len(recalls),
        "per_type": per_type,
        "confusion_matrix": confusion,
        "priority_confusions": priority_confusions,
        "semantic_consistency": _semantic_consistency_metrics(predictions),
    }


def _counts_by_type(frame: pl.DataFrame) -> dict[str, int]:
    return {
        operation_type: frame.filter(
            pl.col("type_op") == operation_type
        ).height
        for operation_type in FUSION_TYPES
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
    technical_errors: Sequence[Mapping[str, Any]],
    semantic_errors: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    technical_stages = Counter(
        error["failure_stage"] for error in technical_errors
    )
    technical_codes = Counter(
        error["failure_code"] for error in technical_errors
    )
    semantic_codes = Counter(
        error["failure_code"] for error in semantic_errors
    )
    return {
        "technical_total": len(technical_errors),
        "technical_by_stage": dict(sorted(technical_stages.items())),
        "technical_by_code": dict(sorted(technical_codes.items())),
        "semantic_error_rows": len(semantic_errors),
        "semantic_by_code": dict(sorted(semantic_codes.items())),
    }


def _coverage_summary(
    annotations: pl.DataFrame,
    selected: pl.DataFrame,
    benchmark_annotations: pl.DataFrame,
    predictions: pl.DataFrame,
    technical_errors: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_type: dict[str, dict[str, int]] = {}
    for operation_type in FUSION_TYPES:
        type_predictions = predictions.filter(
            pl.col("reference_type") == operation_type
        )
        eligible_technical_failures = sum(
            error["reference_type"] == operation_type
            and (
                error["failure_stage"] not in (
                    "reference_validation",
                    "lookup_resolution",
                )
                or error["failure_code"] == "unresolved_reference"
            )
            for error in technical_errors
        )
        by_type[operation_type] = {
            "available_rows": annotations.filter(
                pl.col("type_op") == operation_type
            ).height,
            "selected_rows": selected.filter(
                pl.col("type_op") == operation_type
            ).height,
            "benchmark_eligible_rows": benchmark_annotations.filter(
                pl.col("type_op") == operation_type
            ).height,
            "successful_valid_outputs": type_predictions.height,
            "semantic_unknown_outputs": type_predictions.filter(
                pl.col("predicted_type") == FusionSubtype.UNKNOWN.value
            ).height,
            "semantically_inconsistent_outputs": type_predictions.filter(
                ~pl.col("semantically_consistent")
            ).height,
            "technical_failure_rows": eligible_technical_failures,
        }
    out_of_scope_rows = _out_of_scope_rows(annotations)
    out_of_scope = out_of_scope_rows.height
    out_of_scope_counts = Counter(
        "__NULL__" if row["type_op"] is None else str(row["type_op"])
        for row in out_of_scope_rows.iter_rows(named=True)
    )
    return {
        "total_annotation_rows_loaded": annotations.height,
        "supported_reference_rows_available": sum(
            _counts_by_type(annotations).values()
        ),
        "out_of_scope_reference_rows": out_of_scope,
        "unsupported_reference_rows": out_of_scope,
        "out_of_scope_counts_by_type": dict(sorted(out_of_scope_counts.items())),
        "rows_selected_after_sampling": selected.height,
        "benchmark_eligible_rows": benchmark_annotations.height,
        "invalid_join_key_failures": sum(
            error["failure_code"] == "invalid_join_key"
            for error in technical_errors
        ),
        "duplicate_join_key_failures": sum(
            error["failure_code"] == "duplicate_join_key"
            for error in technical_errors
        ),
        "successful_valid_outputs": predictions.height,
        "by_type": by_type,
    }


def _semantic_error_rows(
    predictions: pl.DataFrame,
) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    for row in predictions.iter_rows(named=True):
        base = {
            JOIN_KEY: row[JOIN_KEY],
            "reference_type": row["reference_type"],
            "predicted_type": row["predicted_type"],
            "bodacc_id": row[JOIN_KEY],
            "correct": row["correct"],
            "transfer_scope": row["transfer_scope"],
            "transferor_fate": row["transferor_fate"],
            "beneficiary_creation": row["beneficiary_creation"],
            "beneficiary_count": row["beneficiary_count"],
            "semantically_consistent": row["semantically_consistent"],
            "semantic_consistency_issues": row[
                "semantic_consistency_issues"
            ],
            "reason": row["reason"],
            "evidence": row["evidence"],
        }
        if not row["correct"]:
            is_unknown = (
                row["predicted_type"] == FusionSubtype.UNKNOWN.value
            )
            errors.append(
                {
                    **base,
                    "failure_stage": "fusion_evaluation",
                    "failure_code": (
                        "semantic_unknown" if is_unknown else "misclassification"
                    ),
                    "failure_reason": (
                        "Router explicitly abstained with UNKNOWN"
                        if is_unknown
                        else "Predicted fusion subtype differs from reference type"
                    ),
                }
            )
        if not row["semantically_consistent"]:
            errors.append(
                {
                    **base,
                    "failure_stage": "semantic_consistency",
                    "failure_code": "semantic_inconsistency",
                    "failure_reason": (
                        "Subtype and semantic axes are inconsistent: "
                        + "; ".join(row["semantic_consistency_issues"])
                    ),
                }
            )
    return errors


def _write_outputs(
    output_dir: Path,
    predictions: pl.DataFrame,
    errors: pl.DataFrame,
    summary: Mapping[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.write_parquet(output_dir / "fusion_predictions.parquet")
    errors.write_parquet(output_dir / "fusion_errors.parquet")
    with (output_dir / "fusion_summary.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def run_fusion_subtype_benchmark(
    annotations_path: str | Path,
    output_dir: str | Path,
    *,
    max_per_type: int | None = 5,
    fetch_announcement: Callable[[str], Mapping[str, Any]] | None = None,
    router: FusionRouter | None = None,
    run_timestamp: datetime | None = None,
    git_commit: str | None = None,
    model_name: str | None = None,
) -> FusionSubtypeBenchmarkResult:
    """Run fusion subtype classification only, without any field extraction."""

    annotations = load_fusion_annotations(annotations_path)
    selected = select_fusion_annotations(
        annotations, max_per_type=max_per_type
    )
    benchmark_annotations, join_key_errors = _partition_benchmark_annotations(
        selected
    )
    technical_errors: list[dict[str, Any]] = []
    technical_errors.extend(join_key_errors)
    fetch = fetch_announcement or bodacc_api().fetch_annonce_json
    active_router = router or fusion_subtype_router

    prediction_rows: list[dict[str, Any]] = []
    for annotation in benchmark_annotations.iter_rows(named=True):
        key = str(annotation[JOIN_KEY])
        operation_type = annotation["type_op"]
        reference_subtype = reference_subtype_for_type(operation_type)
        try:
            bodacc_id = resolve_bodacc_announcement_id(annotation)
        except BodaccLookupResolutionError as error:
            technical_errors.append(
                _technical_error(
                    annotation=annotation,
                    stage="lookup_resolution",
                    code="unresolved_reference",
                    reason=str(error),
                )
            )
            continue

        try:
            announcement = fetch(bodacc_id)
        except BodaccFetchError as error:
            technical_errors.append(
                _technical_error(
                    annotation=annotation,
                    bodacc_id=bodacc_id,
                    stage="bodacc_fetch",
                    code=error.code,
                    reason=error.detail,
                )
            )
            continue
        except Exception as error:
            technical_errors.append(
                _technical_error(
                    annotation=annotation,
                    bodacc_id=bodacc_id,
                    stage="bodacc_fetch",
                    code=type(error).__name__,
                    reason=f"Unexpected fetch failure ({type(error).__name__})",
                )
            )
            continue

        try:
            result = active_router.route(announcement)
            if not isinstance(result, FusionSubtypeResult):
                raise FusionSubtypeOutputError(
                    "invalid_router_result",
                    "Router returned an unvalidated result object",
                )
            consistency_issues = semantic_consistency_issues(result)
        except FusionSubtypeOutputError as error:
            technical_errors.append(
                _technical_error(
                    annotation=annotation,
                    bodacc_id=bodacc_id,
                    stage="llm_output_validation",
                    code=error.code,
                    reason=error.detail,
                )
            )
            continue
        except FusionSubtypeLLMError as error:
            technical_errors.append(
                _technical_error(
                    annotation=annotation,
                    bodacc_id=bodacc_id,
                    stage="llm_execution",
                    code=error.code,
                    reason=error.detail,
                )
            )
            continue
        except BodaccNormalizationError as error:
            technical_errors.append(
                _technical_error(
                    annotation=annotation,
                    bodacc_id=bodacc_id,
                    stage="router_preparation",
                    code=type(error).__name__,
                    reason=str(error),
                )
            )
            continue
        except Exception as error:
            technical_errors.append(
                _technical_error(
                    annotation=annotation,
                    bodacc_id=bodacc_id,
                    stage="router_execution",
                    code=type(error).__name__,
                    reason=f"Unexpected router failure ({type(error).__name__})",
                )
            )
            continue

        prediction_rows.append(
            {
                JOIN_KEY: key,
                "reference_type": operation_type,
                "predicted_type": result.subtype.value,
                "correct": result.subtype is reference_subtype,
                "transfer_scope": result.transfer_scope.value,
                "transferor_fate": result.transferor_fate.value,
                "beneficiary_creation": result.beneficiary_creation.value,
                "beneficiary_count": result.beneficiary_count.value,
                "semantically_consistent": not consistency_issues,
                "semantic_consistency_issues": list(consistency_issues),
                "reason": result.reason,
                "evidence": list(result.evidence),
            }
        )

    predictions = _frame_from_rows(prediction_rows, PREDICTION_SCHEMA)
    technical_router_failures = sum(
        error["failure_stage"] in ROUTER_FAILURE_STAGES
        for error in technical_errors
    )
    metrics = summarize_fusion_subtype_metrics(
        benchmark_annotations,
        predictions,
        technical_router_failures=technical_router_failures,
    )
    semantic_errors = _semantic_error_rows(predictions)
    all_errors = [*technical_errors, *semantic_errors]
    errors = _frame_from_rows(all_errors, ERROR_SCHEMA)

    timestamp = run_timestamp or datetime.now(UTC)
    summary = {
        "metadata": {
            "run_timestamp": timestamp.astimezone(UTC).isoformat(),
            "git_commit": (
                git_commit if git_commit is not None else _current_git_commit()
            ),
            "annotations_file": Path(annotations_path).name,
            "max_per_type": max_per_type,
            "selection_mode": "stratified_by_final_fusion_type",
            "selected_counts_by_type": _counts_by_type(selected),
            "benchmark_eligible_counts_by_type": _counts_by_type(
                benchmark_annotations
            ),
            "llm_model_name": model_name or get_model_name(),
            "fusion_subtype_prompt_version": FUSION_SUBTYPE_PROMPT_VERSION,
            "fusion_subtype_taxonomy_version": (
                FUSION_SUBTYPE_TAXONOMY_VERSION
            ),
        },
        "coverage": _coverage_summary(
            annotations,
            selected,
            benchmark_annotations,
            predictions,
            technical_errors,
        ),
        "failures": _failure_summary(technical_errors, semantic_errors),
        "metrics": metrics,
        "confusion_matrix": metrics["confusion_matrix"],
        "semantic_consistency": metrics["semantic_consistency"],
        "artifacts": {
            "predictions": "fusion_predictions.parquet",
            "errors": "fusion_errors.parquet",
            "summary": "fusion_summary.json",
        },
    }
    _write_outputs(Path(output_dir), predictions, errors, summary)
    return FusionSubtypeBenchmarkResult(
        annotations=annotations,
        selected_annotations=selected,
        benchmark_annotations=benchmark_annotations,
        predictions=predictions,
        errors=errors,
        summary=summary,
    )


def _sample_limit(value: str) -> int | None:
    if value.lower() == "all":
        return None
    try:
        limit = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "must be a non-negative integer or 'all'"
        ) from error
    if limit < 0:
        raise argparse.ArgumentTypeError(
            "must be a non-negative integer or 'all'"
        )
    return limit


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark FU/AB/SP/ST/AP routing without running extraction"
        )
    )
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--max-per-type",
        type=_sample_limit,
        default=5,
        help=(
            "maximum rows per fusion type in stable id order "
            "(default: 5; use 'all' for the full dataset)"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    run_fusion_subtype_benchmark(
        args.annotations,
        args.output_dir,
        max_per_type=args.max_per_type,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
