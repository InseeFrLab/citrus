"""Real-data benchmark for semantic routing, independent from extraction."""

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
from src.modele.benchmark import CANONICAL_TYPES, JOIN_KEY
from src.modele.bodacc_lookup import (
    BodaccLookupResolutionError,
    resolve_bodacc_announcement_id,
)
from src.routing import (
    ROUTER_PROMPT_VERSION,
    ROUTING_TAXONOMY_VERSION,
    RoutingFamily,
    RoutingLLMError,
    RoutingOutputError,
    RoutingResult,
    family_router,
)


FINAL_TYPES = CANONICAL_TYPES
REFERENCE_FAMILIES = (
    RoutingFamily.VE,
    RoutingFamily.LG,
    RoutingFamily.TP,
    RoutingFamily.FUSION_FAMILY,
)
PREDICTION_OUTCOMES = (
    *REFERENCE_FAMILIES,
    RoutingFamily.UNKNOWN,
)
FUSION_FINAL_TYPES = ("FU", "AB", "SP", "ST", "AP")
REFERENCE_FAMILY_BY_TYPE = {
    "VE": RoutingFamily.VE,
    "LG": RoutingFamily.LG,
    "TP": RoutingFamily.TP,
    **{
        operation_type: RoutingFamily.FUSION_FAMILY
        for operation_type in FUSION_FINAL_TYPES
    },
}
ANNOTATION_COLUMNS = (
    "ref_annonce",
    "numero_annonce",
    JOIN_KEY,
    "type_op",
)
PREDICTION_COLUMNS = (
    JOIN_KEY,
    "reference_type",
    "reference_family",
    "predicted_family",
    "correct",
    "reason",
    "evidence",
)
PREDICTION_SCHEMA = {
    JOIN_KEY: pl.String,
    "reference_type": pl.String,
    "reference_family": pl.String,
    "predicted_family": pl.String,
    "correct": pl.Boolean,
    "reason": pl.String,
    "evidence": pl.List(pl.String),
}
ERROR_SCHEMA = {
    JOIN_KEY: pl.String,
    "reference_type": pl.String,
    "reference_family": pl.String,
    "predicted_family": pl.String,
    "bodacc_id": pl.String,
    "failure_stage": pl.String,
    "failure_code": pl.String,
    "failure_reason": pl.String,
    "correct": pl.Boolean,
    "reason": pl.String,
    "evidence": pl.List(pl.String),
}
ROUTER_FAILURE_STAGES = (
    "router_preparation",
    "llm_execution",
    "llm_output_validation",
    "router_execution",
)


class Router(Protocol):
    """Small injectable boundary used by the real-data runner."""

    def route(self, announcement: Mapping[str, Any]) -> RoutingResult:
        ...


@dataclass(frozen=True)
class RoutingBenchmarkResult:
    """In-memory routing results matching the persisted artifacts."""

    annotations: pl.DataFrame
    selected_annotations: pl.DataFrame
    benchmark_annotations: pl.DataFrame
    predictions: pl.DataFrame
    errors: pl.DataFrame
    summary: dict[str, Any]


def reference_family_for_type(operation_type: Any) -> RoutingFamily:
    """Map one exact final annotation type to its routing family."""

    if not isinstance(operation_type, str):
        raise ValueError("reference type must be a canonical non-empty string")
    try:
        return REFERENCE_FAMILY_BY_TYPE[operation_type]
    except KeyError as error:
        raise ValueError(
            f"unsupported reference type: {operation_type!r}"
        ) from error


def load_routing_annotations(path: str | Path) -> pl.DataFrame:
    """Load and immediately project routing lookup/reference fields only."""

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
            f"Missing routing annotation columns: {sorted(missing)}"
        )
    return frame.select(ANNOTATION_COLUMNS)


def select_routing_annotations(
    annotations: pl.DataFrame,
    *,
    max_per_type: int | None = 10,
) -> pl.DataFrame:
    """Sample deterministically within each of the eight final types."""

    if max_per_type is not None and max_per_type < 0:
        raise ValueError("max_per_type must be non-negative or None")
    selected: list[pl.DataFrame] = []
    for operation_type in FINAL_TYPES:
        type_rows = annotations.filter(
            pl.col("type_op") == operation_type
        ).sort(JOIN_KEY)
        if max_per_type is not None:
            type_rows = type_rows.head(max_per_type)
        selected.append(type_rows)
    return pl.concat(selected, how="vertical_relaxed")


def _unsupported_reference_rows(annotations: pl.DataFrame) -> pl.DataFrame:
    supported = pl.col("type_op").is_in(FINAL_TYPES).fill_null(False)
    return annotations.filter(~supported)


def _string_key(value: Any) -> str | None:
    return value if isinstance(value, str) else None if value is None else str(value)


def _technical_error(
    *,
    annotation: Mapping[str, Any],
    stage: str,
    code: str,
    reason: str,
    bodacc_id: str | None = None,
) -> dict[str, Any]:
    operation_type = annotation.get("type_op")
    reference_family = REFERENCE_FAMILY_BY_TYPE.get(operation_type)
    return {
        JOIN_KEY: _string_key(annotation.get(JOIN_KEY)),
        "reference_type": (
            operation_type if isinstance(operation_type, str) else None
        ),
        "reference_family": (
            reference_family.value if reference_family is not None else None
        ),
        "predicted_family": None,
        "bodacc_id": bodacc_id,
        "failure_stage": stage,
        "failure_code": code,
        "failure_reason": reason,
        "correct": None,
        "reason": None,
        "evidence": None,
    }


def _partition_benchmark_annotations(
    selected: pl.DataFrame,
) -> tuple[pl.DataFrame, list[dict[str, Any]]]:
    """Exclude null/blank and every duplicate join-key occurrence."""

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


def _prediction_by_key(predictions: pl.DataFrame) -> dict[str, Mapping[str, Any]]:
    by_key: dict[str, Mapping[str, Any]] = {}
    for row in predictions.iter_rows(named=True):
        key = row[JOIN_KEY]
        if key in by_key:
            raise ValueError(f"duplicate routing prediction key: {key}")
        by_key[key] = row
    return by_key


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def summarize_routing_metrics(
    benchmark_annotations: pl.DataFrame,
    predictions: pl.DataFrame,
    *,
    technical_router_failures: int | None = None,
) -> dict[str, Any]:
    """Compute routing-only metrics with UNKNOWN and __ERROR__ separated."""

    predicted_by_key = _prediction_by_key(predictions)
    rows: list[dict[str, Any]] = []
    for annotation in benchmark_annotations.iter_rows(named=True):
        key = str(annotation[JOIN_KEY])
        reference_type = annotation["type_op"]
        reference_family = reference_family_for_type(reference_type)
        prediction = predicted_by_key.get(key)
        predicted_family = (
            prediction["predicted_family"] if prediction is not None else "__ERROR__"
        )
        rows.append(
            {
                "reference_type": reference_type,
                "reference_family": reference_family.value,
                "predicted_family": predicted_family,
            }
        )

    eligible = len(rows)
    successful = predictions.height
    technical_failures = eligible - successful
    if technical_failures < 0:
        raise ValueError("routing predictions exceed eligible reference rows")
    unknown_count = sum(
        row["predicted_family"] == RoutingFamily.UNKNOWN.value for row in rows
    )
    non_unknown_count = sum(
        row["predicted_family"]
        not in (RoutingFamily.UNKNOWN.value, "__ERROR__")
        for row in rows
    )
    correct = sum(
        row["reference_family"] == row["predicted_family"] for row in rows
    )

    confusion_columns = (
        *(family.value for family in PREDICTION_OUTCOMES),
        "__ERROR__",
    )
    confusion = {
        family.value: {outcome: 0 for outcome in confusion_columns}
        for family in REFERENCE_FAMILIES
    }
    for row in rows:
        confusion[row["reference_family"]][row["predicted_family"]] += 1

    per_family: dict[str, dict[str, Any]] = {}
    recalls: list[float] = []
    f1_scores: list[float] = []
    for family in REFERENCE_FAMILIES:
        label = family.value
        reference_rows = [row for row in rows if row["reference_family"] == label]
        support = len(reference_rows)
        family_correct = sum(
            row["predicted_family"] == label for row in reference_rows
        )
        predicted_total = sum(
            row["predicted_family"] == label for row in rows
        )
        predicted_unknown = sum(
            row["predicted_family"] == RoutingFamily.UNKNOWN.value
            for row in reference_rows
        )
        family_technical_failures = sum(
            row["predicted_family"] == "__ERROR__" for row in reference_rows
        )
        recall = _ratio(family_correct, support)
        precision = _ratio(family_correct, predicted_total)
        f1 = (
            2 * family_correct / (support + predicted_total)
            if support + predicted_total
            else None
        )
        if support:
            recalls.append(recall if recall is not None else 0.0)
            f1_scores.append(f1 if f1 is not None else 0.0)
        per_family[label] = {
            "support": support,
            "correct": family_correct,
            "recall": recall,
            "precision": precision,
            "f1": f1,
            "predicted_unknown_count": predicted_unknown,
            "predicted_unknown_rate": _ratio(predicted_unknown, support),
            "technical_failure_count": family_technical_failures,
        }

    fusion_breakdown: dict[str, dict[str, Any]] = {}
    for operation_type in FUSION_FINAL_TYPES:
        type_rows = [row for row in rows if row["reference_type"] == operation_type]
        support = len(type_rows)
        type_correct = sum(
            row["predicted_family"] == RoutingFamily.FUSION_FAMILY.value
            for row in type_rows
        )
        type_unknown = sum(
            row["predicted_family"] == RoutingFamily.UNKNOWN.value
            for row in type_rows
        )
        type_errors = sum(
            row["predicted_family"] == "__ERROR__" for row in type_rows
        )
        fusion_breakdown[operation_type] = {
            "support": support,
            "correct": type_correct,
            "recall": _ratio(type_correct, support),
            "predicted_unknown_count": type_unknown,
            "predicted_unknown_rate": _ratio(type_unknown, support),
            "technical_failure_count": type_errors,
            "prediction_counts": {
                outcome: sum(
                    row["predicted_family"] == outcome for row in type_rows
                )
                for outcome in confusion_columns
            },
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
        "coverage": _ratio(non_unknown_count, eligible),
        "selective_accuracy": _ratio(correct, non_unknown_count),
        "macro_recall": (
            sum(recalls) / len(recalls) if recalls else None
        ),
        "macro_f1": (
            sum(f1_scores) / len(f1_scores) if f1_scores else None
        ),
        "macro_supported_family_count": len(recalls),
        "per_reference_family": per_family,
        "confusion_matrix": confusion,
        "fusion_family_by_final_type": fusion_breakdown,
    }


def _counts_by_type(frame: pl.DataFrame) -> dict[str, int]:
    return {
        operation_type: frame.filter(
            pl.col("type_op") == operation_type
        ).height
        for operation_type in FINAL_TYPES
    }


def _counts_by_family(frame: pl.DataFrame) -> dict[str, int]:
    counts = {family.value: 0 for family in REFERENCE_FAMILIES}
    for operation_type, count in _counts_by_type(frame).items():
        counts[REFERENCE_FAMILY_BY_TYPE[operation_type].value] += count
    return counts


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
    semantic_error_count: int,
) -> dict[str, Any]:
    stages = Counter(error["failure_stage"] for error in technical_errors)
    codes = Counter(error["failure_code"] for error in technical_errors)
    return {
        "technical_total": len(technical_errors),
        "technical_by_stage": dict(sorted(stages.items())),
        "technical_by_code": dict(sorted(codes.items())),
        "semantic_error_rows": semantic_error_count,
    }


def _coverage_summary(
    annotations: pl.DataFrame,
    selected: pl.DataFrame,
    benchmark_annotations: pl.DataFrame,
    predictions: pl.DataFrame,
    technical_errors: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_final_type: dict[str, dict[str, int]] = {}
    for operation_type in FINAL_TYPES:
        type_predictions = predictions.filter(
            pl.col("reference_type") == operation_type
        )
        typed_errors = [
            error
            for error in technical_errors
            if error["reference_type"] == operation_type
        ]
        by_final_type[operation_type] = {
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
                pl.col("predicted_family") == RoutingFamily.UNKNOWN.value
            ).height,
            "technical_failure_rows": sum(
                error["failure_stage"] not in (
                    "reference_validation",
                    "lookup_resolution",
                )
                or error["failure_code"] == "unresolved_reference"
                for error in typed_errors
            ),
        }
    return {
        "total_annotation_rows_loaded": annotations.height,
        "supported_reference_rows_available": sum(
            _counts_by_type(annotations).values()
        ),
        "unsupported_reference_rows": _unsupported_reference_rows(
            annotations
        ).height,
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
        "by_final_type": by_final_type,
    }


def _semantic_error_rows(
    predictions: pl.DataFrame,
) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    for row in predictions.filter(~pl.col("correct")).iter_rows(named=True):
        is_unknown = row["predicted_family"] == RoutingFamily.UNKNOWN.value
        errors.append(
            {
                JOIN_KEY: row[JOIN_KEY],
                "reference_type": row["reference_type"],
                "reference_family": row["reference_family"],
                "predicted_family": row["predicted_family"],
                "bodacc_id": row[JOIN_KEY],
                "failure_stage": "routing_evaluation",
                "failure_code": (
                    "semantic_unknown" if is_unknown else "misclassification"
                ),
                "failure_reason": (
                    "Router explicitly abstained with UNKNOWN"
                    if is_unknown
                    else "Predicted routing family differs from reference family"
                ),
                "correct": False,
                "reason": row["reason"],
                "evidence": row["evidence"],
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
    predictions.write_parquet(output_dir / "routing_predictions.parquet")
    errors.write_parquet(output_dir / "routing_errors.parquet")
    with (output_dir / "routing_summary.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def run_routing_benchmark(
    annotations_path: str | Path,
    output_dir: str | Path,
    *,
    max_per_type: int | None = 10,
    fetch_announcement: Callable[[str], Mapping[str, Any]] | None = None,
    router: Router | None = None,
    run_timestamp: datetime | None = None,
    git_commit: str | None = None,
    model_name: str | None = None,
) -> RoutingBenchmarkResult:
    """Run family classification only; no operation extraction skill is called."""

    annotations = load_routing_annotations(annotations_path)
    selected = select_routing_annotations(
        annotations, max_per_type=max_per_type
    )
    benchmark_annotations, join_key_errors = _partition_benchmark_annotations(
        selected
    )
    technical_errors: list[dict[str, Any]] = []
    for row in _unsupported_reference_rows(annotations).iter_rows(named=True):
        technical_errors.append(
            _technical_error(
                annotation=row,
                stage="reference_validation",
                code="unsupported_reference_type",
                reason="type_op is null, blank, or non-canonical",
            )
        )
    technical_errors.extend(join_key_errors)
    fetch = fetch_announcement or bodacc_api().fetch_annonce_json
    active_router = router or family_router

    prediction_rows: list[dict[str, Any]] = []
    for annotation in benchmark_annotations.iter_rows(named=True):
        key = str(annotation[JOIN_KEY])
        operation_type = annotation["type_op"]
        reference_family = reference_family_for_type(operation_type)
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
            if not isinstance(result, RoutingResult):
                raise RoutingOutputError(
                    "invalid_router_result",
                    "Router returned an unvalidated result object",
                )
        except RoutingOutputError as error:
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
        except RoutingLLMError as error:
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
                "reference_family": reference_family.value,
                "predicted_family": result.family.value,
                "correct": result.family is reference_family,
                "reason": result.reason,
                "evidence": list(result.evidence),
            }
        )

    predictions = _frame_from_rows(prediction_rows, PREDICTION_SCHEMA)
    technical_router_failures = sum(
        error["failure_stage"] in ROUTER_FAILURE_STAGES
        for error in technical_errors
    )
    metrics = summarize_routing_metrics(
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
            "selection_mode": "stratified_by_final_annotated_type",
            "selected_counts_by_final_type": _counts_by_type(selected),
            "benchmark_eligible_counts_by_final_type": _counts_by_type(
                benchmark_annotations
            ),
            "selected_reference_family_counts": _counts_by_family(selected),
            "benchmark_eligible_reference_family_counts": _counts_by_family(
                benchmark_annotations
            ),
            "llm_model_name": model_name or get_model_name(),
            "router_prompt_version": ROUTER_PROMPT_VERSION,
            "routing_taxonomy_version": ROUTING_TAXONOMY_VERSION,
        },
        "coverage": _coverage_summary(
            annotations,
            selected,
            benchmark_annotations,
            predictions,
            technical_errors,
        ),
        "failures": _failure_summary(
            technical_errors, len(semantic_errors)
        ),
        "metrics": metrics,
        "artifacts": {
            "predictions": "routing_predictions.parquet",
            "errors": "routing_errors.parquet",
            "summary": "routing_summary.json",
        },
    }
    _write_outputs(Path(output_dir), predictions, errors, summary)
    return RoutingBenchmarkResult(
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
            "Benchmark LLM family routing without running extraction skills"
        )
    )
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--max-per-type",
        type=_sample_limit,
        default=10,
        help=(
            "maximum rows per final annotated type in stable id order "
            "(default: 10; use 'all' for the full dataset)"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    run_routing_benchmark(
        args.annotations,
        args.output_dir,
        max_per_type=args.max_per_type,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
