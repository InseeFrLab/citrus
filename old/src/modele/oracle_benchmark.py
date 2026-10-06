"""Real-data VE/LG/TP extraction benchmark selected by annotated type."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import subprocess
from typing import Any

import polars as pl

from src.bodacc.api import BodaccFetchError, bodacc_api
from src.modele.benchmark import (
    JOIN_KEY,
    TARGET_FIELDS,
    BenchmarkComparison,
    compare_predictions,
    normalize_predictions,
    summarize_metrics,
)
from src.modele.bodacc_lookup import (
    BodaccLookupResolutionError,
    resolve_bodacc_announcement_id,
)
from src.operation import (
    OperationSkill,
    location_gerance_skill,
    transmission_patrimoine_skill,
    vente_skill,
)


SUPPORTED_TYPES = ("VE", "LG", "TP")
DEFAULT_AMOUNT_TOLERANCE = 0.1
ANNOTATION_COLUMNS = (
    "ref_annonce",
    "numero_annonce",
    JOIN_KEY,
    "type_op",
    "siren_cedante",
    "siren_beneficiaire",
    "date_effet_comptable_op",
    "date_realisation_juridique_op",
    "montant",
)
PREDICTION_COLUMNS = (
    JOIN_KEY,
    "oracle_type",
    "anneeCampagne",
    "typeOperation",
    "sirenCedant",
    "raisonSocialeCedant",
    "sirenBeneficiaire",
    "raisonSocialeBeneficiaire",
    "dateEffetComptable",
    "dateRealisationJuridique",
    "montantNet",
    "source",
)
PREDICTION_SCHEMA = {
    JOIN_KEY: pl.String,
    "oracle_type": pl.String,
    "anneeCampagne": pl.Int64,
    "typeOperation": pl.String,
    "sirenCedant": pl.String,
    "raisonSocialeCedant": pl.String,
    "sirenBeneficiaire": pl.String,
    "raisonSocialeBeneficiaire": pl.String,
    "dateEffetComptable": pl.String,
    "dateRealisationJuridique": pl.String,
    "montantNet": pl.Float64,
    "source": pl.String,
}
ERROR_SCHEMA = {
    JOIN_KEY: pl.String,
    "type_op": pl.String,
    "bodacc_id": pl.String,
    "failure_stage": pl.String,
    "failure_code": pl.String,
    "failure_reason": pl.String,
    "failing_fields": pl.List(pl.String),
    **{
        f"expected_{field}": (
            pl.Date
            if field in ("dateEffetComptable", "dateRealisationJuridique")
            else pl.Float64 if field == "montantNet" else pl.String
        )
        for field in TARGET_FIELDS
    },
    **{
        f"predicted_{field}": (
            pl.Date
            if field in ("dateEffetComptable", "dateRealisationJuridique")
            else pl.Float64 if field == "montantNet" else pl.String
        )
        for field in TARGET_FIELDS
    },
    **{f"{field}_correct": pl.Boolean for field in TARGET_FIELDS},
    "exact_row_correct": pl.Boolean,
}

@dataclass(frozen=True)
class OracleBenchmarkResult:
    """In-memory results matching the persisted benchmark artifacts."""

    selected_annotations: pl.DataFrame
    benchmark_annotations: pl.DataFrame
    predictions: pl.DataFrame
    comparison: BenchmarkComparison
    errors: pl.DataFrame
    summary: dict[str, Any]


def load_annotation_dataset(path: str | Path) -> pl.DataFrame:
    """Load external Parquet/CSV annotations and discard irrelevant columns.

    In particular, ``date_creation_op`` is never selected into the runner's
    in-memory dataset and cannot affect sampling, extraction, or evaluation.
    """

    source = Path(path)
    if source.suffix.lower() == ".parquet":
        frame = pl.read_parquet(source)
    elif source.suffix.lower() == ".csv":
        frame = pl.read_csv(source)
    else:
        raise ValueError("annotations must be a .parquet or .csv file")
    missing = set(ANNOTATION_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing runner annotation columns: {sorted(missing)}")
    return frame.select(ANNOTATION_COLUMNS)


def select_oracle_annotations(
    annotations: pl.DataFrame,
    *,
    max_ve: int | None = 50,
    max_lg: int | None = 50,
    max_tp: int | None = 0,
) -> pl.DataFrame:
    """Select VE/LG/TP rows in deterministic announcement-id order per type."""

    limits = {"VE": max_ve, "LG": max_lg, "TP": max_tp}
    selected: list[pl.DataFrame] = []
    for operation_type in SUPPORTED_TYPES:
        limit = limits[operation_type]
        if limit is not None and limit < 0:
            raise ValueError("sample limits must be non-negative or None")
        type_rows = annotations.filter(
            pl.col("type_op") == operation_type
        ).sort(JOIN_KEY)
        if limit is not None:
            type_rows = type_rows.head(limit)
        selected.append(type_rows)
    return pl.concat(selected) if selected else annotations.head(0)


def _empty_prediction_frame() -> pl.DataFrame:
    return pl.DataFrame(schema=PREDICTION_SCHEMA)


def _prediction_row(
    key: str, oracle_type: str, extracted: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        JOIN_KEY: key,
        "oracle_type": oracle_type,
        **{
            field: extracted.get(field)
            for field in PREDICTION_COLUMNS
            if field not in (JOIN_KEY, "oracle_type")
        },
    }


def _pipeline_error(
    *,
    key: str | None,
    operation_type: str,
    bodacc_id: str | None,
    stage: str,
    code: str,
    reason: str,
) -> dict[str, Any]:
    return {
        JOIN_KEY: key,
        "type_op": operation_type,
        "bodacc_id": bodacc_id,
        "failure_stage": stage,
        "failure_code": code,
        "failure_reason": reason,
        "failing_fields": None,
    }


def _partition_benchmark_annotations(
    selected: pl.DataFrame,
) -> tuple[pl.DataFrame, list[dict[str, Any]]]:
    """Exclude only rows whose join key violates generic benchmark safety."""

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
        raw_key = row[JOIN_KEY]
        errors.append(
            _pipeline_error(
                key=None if raw_key is None else str(raw_key),
                operation_type=row["type_op"],
                bodacc_id=None,
                stage="lookup_resolution",
                code="invalid_join_key",
                reason="ref_annonce_complet must be non-null and non-empty",
            )
        )
    duplicate_sizes = dict(
        duplicate_counts.select("_benchmark_join_key", "len").iter_rows()
    )
    for row in duplicate_rows.iter_rows(named=True):
        key = row["_benchmark_join_key"]
        errors.append(
            _pipeline_error(
                key=key,
                operation_type=row["type_op"],
                bodacc_id=None,
                stage="lookup_resolution",
                code="duplicate_join_key",
                reason=(
                    f"ref_annonce_complet occurs {duplicate_sizes[key]} times "
                    "in the selected sample"
                ),
            )
        )
    return eligible, errors


def _benchmark_error_rows(comparison: BenchmarkComparison) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    incorrect = comparison.rows.filter(~pl.col("exact_row_correct"))
    for row in incorrect.iter_rows(named=True):
        failing = [
            field for field in TARGET_FIELDS if not row[f"{field}_correct"]
        ]
        error = {
            JOIN_KEY: row[JOIN_KEY],
            "type_op": row["typeOperation"],
            "bodacc_id": row[JOIN_KEY],
            "failure_stage": "benchmark",
            "failure_code": "field_mismatch",
            "failure_reason": f"Incorrect fields: {', '.join(failing)}",
            "failing_fields": failing,
            "exact_row_correct": False,
        }
        for field in TARGET_FIELDS:
            error[f"expected_{field}"] = row[field]
            error[f"predicted_{field}"] = row[f"{field}_prediction"]
            error[f"{field}_correct"] = row[f"{field}_correct"]
        rows.append(error)
    return rows


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


def _type_counts(frame: pl.DataFrame) -> dict[str, int]:
    return {
        operation_type: frame.filter(
            pl.col("type_op") == operation_type
        ).height
        for operation_type in SUPPORTED_TYPES
    }


def _coverage_summary(
    annotations: pl.DataFrame,
    selected: pl.DataFrame,
    benchmark_annotations: pl.DataFrame,
    predictions: pl.DataFrame,
    pipeline_errors: Sequence[Mapping[str, Any]],
    benchmark_summary: Mapping[str, Any],
) -> dict[str, Any]:
    available = _type_counts(annotations)
    selected_by_type = _type_counts(selected)
    benchmark_by_type = _type_counts(benchmark_annotations)
    success_by_type = {
        operation_type: predictions.filter(
            pl.col("oracle_type") == operation_type
        ).height
        for operation_type in SUPPORTED_TYPES
    }
    stages = (
        "lookup_resolution",
        "bodacc_fetch",
        "skill_execution",
    )
    by_type: dict[str, dict[str, int]] = {}
    for operation_type in SUPPORTED_TYPES:
        typed_errors = [
            error
            for error in pipeline_errors
            if error["type_op"] == operation_type
        ]
        by_type[operation_type] = {
            "available_rows": available[operation_type],
            "selected_rows": selected_by_type[operation_type],
            "benchmark_eligible_rows": benchmark_by_type[operation_type],
            "invalid_join_key_failures": sum(
                error["failure_code"] == "invalid_join_key"
                for error in typed_errors
            ),
            "duplicate_join_key_failures": sum(
                error["failure_code"] == "duplicate_join_key"
                for error in typed_errors
            ),
            "lookup_resolution_failures": sum(
                error["failure_stage"] == "lookup_resolution"
                for error in typed_errors
            ),
            "bodacc_fetch_failures": sum(
                error["failure_stage"] == "bodacc_fetch"
                for error in typed_errors
            ),
            "skill_execution_failures": sum(
                error["failure_stage"] == "skill_execution"
                for error in typed_errors
            ),
            "successful_predictions": success_by_type[operation_type],
        }

    stage_counts = {
        stage: sum(error["failure_stage"] == stage for error in pipeline_errors)
        for stage in stages
    }
    return {
        "total_annotation_rows_loaded": annotations.height,
        "ve_rows_available": available["VE"],
        "lg_rows_available": available["LG"],
        "tp_rows_available": available["TP"],
        "other_types_skipped": annotations.height - sum(available.values()),
        "rows_selected_after_sampling": selected.height,
        "benchmark_eligible_rows": benchmark_annotations.height,
        "invalid_join_key_failures": sum(
            error["failure_code"] == "invalid_join_key"
            for error in pipeline_errors
        ),
        "duplicate_join_key_failures": sum(
            error["failure_code"] == "duplicate_join_key"
            for error in pipeline_errors
        ),
        "lookup_resolution_failures": stage_counts["lookup_resolution"],
        "bodacc_fetch_failures": stage_counts["bodacc_fetch"],
        "skill_execution_failures": stage_counts["skill_execution"],
        "successful_predictions": predictions.height,
        "by_type": by_type,
        "benchmark": dict(benchmark_summary["coverage"]),
    }


def _write_outputs(
    output_dir: Path,
    predictions: pl.DataFrame,
    comparison: BenchmarkComparison,
    errors: pl.DataFrame,
    summary: Mapping[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.write_parquet(output_dir / "predictions.parquet")
    comparison.rows.write_parquet(output_dir / "comparison.parquet")
    errors.write_parquet(output_dir / "errors.parquet")
    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def run_oracle_benchmark(
    annotations_path: str | Path,
    output_dir: str | Path,
    *,
    max_ve: int | None = 50,
    max_lg: int | None = 50,
    max_tp: int | None = 0,
    amount_tolerance: float = DEFAULT_AMOUNT_TOLERANCE,
    fetch_announcement: Callable[[str], Mapping[str, Any]] | None = None,
    skills: Mapping[str, OperationSkill] | None = None,
    run_timestamp: datetime | None = None,
    git_commit: str | None = None,
) -> OracleBenchmarkResult:
    """Run oracle-selected VE/LG/TP extraction and generic evaluation."""

    annotations = load_annotation_dataset(annotations_path)
    selected = select_oracle_annotations(
        annotations, max_ve=max_ve, max_lg=max_lg, max_tp=max_tp
    )
    benchmark_annotations, join_key_errors = _partition_benchmark_annotations(
        selected
    )
    if fetch_announcement is None:
        fetch_announcement = bodacc_api().fetch_annonce_json
    if skills is None:
        skills = {
            "VE": vente_skill,
            "LG": location_gerance_skill,
            "TP": transmission_patrimoine_skill,
        }

    prediction_rows: list[dict[str, Any]] = []
    pipeline_errors = list(join_key_errors)
    for annotation in benchmark_annotations.iter_rows(named=True):
        key = str(annotation[JOIN_KEY])
        operation_type = annotation["type_op"]
        try:
            bodacc_id = resolve_bodacc_announcement_id(annotation)
        except BodaccLookupResolutionError as error:
            pipeline_errors.append(
                _pipeline_error(
                    key=key,
                    operation_type=operation_type,
                    bodacc_id=None,
                    stage="lookup_resolution",
                    code="unresolved_reference",
                    reason=str(error),
                )
            )
            continue

        try:
            announcement = fetch_announcement(bodacc_id)
        except BodaccFetchError as error:
            pipeline_errors.append(
                _pipeline_error(
                    key=key,
                    operation_type=operation_type,
                    bodacc_id=bodacc_id,
                    stage="bodacc_fetch",
                    code=error.code,
                    reason=error.detail,
                )
            )
            continue
        except Exception as error:  # isolate caller/network adapters per row
            pipeline_errors.append(
                _pipeline_error(
                    key=key,
                    operation_type=operation_type,
                    bodacc_id=bodacc_id,
                    stage="bodacc_fetch",
                    code=type(error).__name__,
                    reason=f"Unexpected fetch failure ({type(error).__name__})",
                )
            )
            continue

        try:
            extracted = skills[operation_type].extract(dict(announcement))
            candidate = _prediction_row(key, operation_type, extracted)
            candidate_frame = pl.DataFrame(
                [candidate], schema=PREDICTION_SCHEMA, strict=False
            )
            normalize_predictions(candidate_frame)
        except Exception as error:  # one extraction must not abort the run
            pipeline_errors.append(
                _pipeline_error(
                    key=key,
                    operation_type=operation_type,
                    bodacc_id=bodacc_id,
                    stage="skill_execution",
                    code=type(error).__name__,
                    reason=f"Skill extraction failed ({type(error).__name__})",
                )
            )
            continue
        prediction_rows.append(candidate)

    predictions = (
        pl.DataFrame(prediction_rows, schema=PREDICTION_SCHEMA, strict=False)
        if prediction_rows
        else _empty_prediction_frame()
    )
    comparison = compare_predictions(
        benchmark_annotations,
        predictions,
        amount_tolerance=amount_tolerance,
    )
    benchmark_summary = summarize_metrics(comparison)
    all_error_rows = [*pipeline_errors, *_benchmark_error_rows(comparison)]
    errors = (
        pl.DataFrame(all_error_rows, schema=ERROR_SCHEMA, strict=False)
        if all_error_rows
        else pl.DataFrame(schema=ERROR_SCHEMA)
    )

    timestamp = run_timestamp or datetime.now(UTC)
    summary = {
        "metadata": {
            "run_timestamp": timestamp.astimezone(UTC).isoformat(),
            "git_commit": (
                git_commit if git_commit is not None else _current_git_commit()
            ),
            "annotations_file": Path(annotations_path).name,
            "requested_limits": {
                "VE": max_ve,
                "LG": max_lg,
                "TP": max_tp,
            },
            "selected_counts": _type_counts(selected),
            "benchmark_eligible_counts": _type_counts(benchmark_annotations),
            "amount_tolerance_keur": amount_tolerance,
            "selection_mode": "reference_type_oracle",
            "type_metric_interpretation": (
                "typeOperation is structurally oracle-driven by the selected skill; "
                "this run does not evaluate classification"
            ),
        },
        "coverage": _coverage_summary(
            annotations,
            selected,
            benchmark_annotations,
            predictions,
            pipeline_errors,
            benchmark_summary,
        ),
        "benchmark": benchmark_summary,
        "artifacts": {
            "predictions": "predictions.parquet",
            "comparison": "comparison.parquet",
            "errors": "errors.parquet",
            "summary": "summary.json",
        },
    }
    _write_outputs(
        Path(output_dir), predictions, comparison, errors, summary
    )
    return OracleBenchmarkResult(
        selected_annotations=selected,
        benchmark_annotations=benchmark_annotations,
        predictions=predictions,
        comparison=comparison,
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
            "Benchmark VE/LG/TP extraction using annotated type_op only as a skill oracle"
        )
    )
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--max-ve",
        type=_sample_limit,
        default=50,
        help="maximum VE rows in stable id order (default: 50; use 'all' for full)",
    )
    parser.add_argument(
        "--max-lg",
        type=_sample_limit,
        default=50,
        help="maximum LG rows in stable id order (default: 50; use 'all' for full)",
    )
    parser.add_argument(
        "--max-tp",
        type=_sample_limit,
        default=0,
        help="maximum TP rows in stable id order (default: 0; use 'all' for full)",
    )
    parser.add_argument(
        "--amount-tolerance",
        type=float,
        default=DEFAULT_AMOUNT_TOLERANCE,
        help="generic benchmark tolerance in kEUR (default: 0.1)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    run_oracle_benchmark(
        args.annotations,
        args.output_dir,
        max_ve=args.max_ve,
        max_lg=args.max_lg,
        max_tp=args.max_tp,
        amount_tolerance=args.amount_tolerance,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
