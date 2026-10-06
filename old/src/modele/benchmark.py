"""Offline comparison of Citrus annotations and already-produced predictions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import polars as pl


JOIN_KEY = "ref_annonce_complet"
CANONICAL_TYPES = ("FU", "AB", "TP", "SP", "AP", "ST", "VE", "LG")
TARGET_FIELDS = (
    "typeOperation",
    "sirenCedant",
    "sirenBeneficiaire",
    "dateEffetComptable",
    "dateRealisationJuridique",
    "montantNet",
)
ANNOTATION_COLUMNS = {
    "type_op": "typeOperation",
    "siren_cedante": "sirenCedant",
    "siren_beneficiaire": "sirenBeneficiaire",
    "date_effet_comptable_op": "dateEffetComptable",
    "date_realisation_juridique_op": "dateRealisationJuridique",
    "montant": "montantNet",
}


class BenchmarkValidationError(ValueError):
    """Raised when benchmark input cannot be compared safely."""


@dataclass(frozen=True)
class BenchmarkComparison:
    """Inspectable comparison rows and key-accounting information."""

    rows: pl.DataFrame
    annotated_count: int
    prediction_count: int
    extra_prediction_keys: tuple[str, ...]


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and value != value)


def _normalize_siren(value: Any) -> str | None:
    if _is_missing(value):
        return None
    if isinstance(value, bool):
        raise BenchmarkValidationError(f"Invalid SIREN {value!r}")
    if isinstance(value, int):
        digits = str(value)
    elif isinstance(value, float) and value.is_integer():
        digits = str(int(value))
    else:
        digits = str(value).strip()
    if not digits.isdigit() or len(digits) > 9:
        raise BenchmarkValidationError(
            f"SIREN must contain at most 9 digits before zero-padding: {value!r}"
        )
    return digits.zfill(9)


def _normalize_date(value: Any) -> date | None:
    if _is_missing(value):
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    for parser in (
        lambda: datetime.fromisoformat(text.replace("Z", "+00:00")).date(),
        lambda: datetime.strptime(text, "%d/%m/%Y %H:%M:%S").date(),
        lambda: datetime.strptime(text, "%d/%m/%Y").date(),
    ):
        try:
            return parser()
        except ValueError:
            pass
    raise BenchmarkValidationError(f"Unsupported benchmark date: {value!r}")


def _normalize_frame(df: pl.DataFrame, *, annotations: bool) -> pl.DataFrame:
    if annotations:
        missing = ({JOIN_KEY} | set(ANNOTATION_COLUMNS)) - set(df.columns)
        if missing:
            raise BenchmarkValidationError(f"Missing annotation columns: {sorted(missing)}")
        df = df.rename(ANNOTATION_COLUMNS).select(JOIN_KEY, *TARGET_FIELDS)
    else:
        missing = ({JOIN_KEY} | set(TARGET_FIELDS)) - set(df.columns)
        if missing:
            raise BenchmarkValidationError(f"Missing prediction columns: {sorted(missing)}")
        df = df.select(JOIN_KEY, *TARGET_FIELDS)

    return df.with_columns(
        pl.col(JOIN_KEY).cast(pl.String),
        pl.col("typeOperation").cast(pl.String),
        pl.col("sirenCedant")
        .map_elements(_normalize_siren, return_dtype=pl.String),
        pl.col("sirenBeneficiaire")
        .map_elements(_normalize_siren, return_dtype=pl.String),
        pl.col("dateEffetComptable")
        .map_elements(_normalize_date, return_dtype=pl.Date),
        pl.col("dateRealisationJuridique")
        .map_elements(_normalize_date, return_dtype=pl.Date),
        pl.col("montantNet").cast(pl.Float64, strict=True),
    )


def normalize_annotations(df: pl.DataFrame) -> pl.DataFrame:
    """Map source annotations, or re-normalize canonical annotations."""

    is_canonical = ({JOIN_KEY} | set(TARGET_FIELDS)).issubset(df.columns)
    return _normalize_frame(df, annotations=not is_canonical)


def normalize_predictions(df: pl.DataFrame) -> pl.DataFrame:
    """Normalize predictions that already respect the Citrus kEUR contract."""

    return _normalize_frame(df, annotations=False)


def load_annotations_csv(source: str | Path, **read_csv_options: Any) -> pl.DataFrame:
    """Load a local or Polars-supported CSV source and normalize annotations."""

    return normalize_annotations(pl.read_csv(source, **read_csv_options))


def _duplicate_keys(df: pl.DataFrame) -> list[str]:
    return (
        df.group_by(JOIN_KEY).len().filter(pl.col("len") > 1)[JOIN_KEY].to_list()
    )


def compare_predictions(
    reference_df: pl.DataFrame,
    predictions_df: pl.DataFrame,
    *,
    amount_tolerance: float = 0.1,
) -> BenchmarkComparison:
    """Normalize and compare frames without performing extraction or I/O."""

    reference = normalize_annotations(reference_df)
    predictions = normalize_predictions(predictions_df)
    for side, frame in (("annotations", reference), ("predictions", predictions)):
        if frame[JOIN_KEY].null_count():
            raise BenchmarkValidationError(f"Null {JOIN_KEY} value in {side}")
        duplicates = _duplicate_keys(frame)
        if duplicates:
            raise BenchmarkValidationError(
                f"Duplicate {JOIN_KEY} values in {side}: {sorted(duplicates)}"
            )

    reference_keys = set(reference[JOIN_KEY].to_list())
    prediction_keys = set(predictions[JOIN_KEY].to_list())
    extra_keys = tuple(sorted(prediction_keys - reference_keys))
    joined = reference.join(
        predictions.with_columns(pl.lit(True).alias("_prediction_present")),
        on=JOIN_KEY,
        how="left",
        suffix="_prediction",
    ).with_columns(pl.col("_prediction_present").fill_null(False))

    assertions = []
    for field in TARGET_FIELDS:
        reference_col = pl.col(field)
        prediction_col = pl.col(f"{field}_prediction")
        if field == "montantNet":
            equal = (reference_col - prediction_col).abs() <= amount_tolerance
        else:
            equal = reference_col == prediction_col
        # Explicit policy: two field-level nulls are equal, but an absent row is not.
        equal_or_both_null = equal.fill_null(
            reference_col.is_null() & prediction_col.is_null()
        )
        assertions.append(
            (pl.col("_prediction_present") & equal_or_both_null).alias(f"{field}_correct")
        )
    joined = joined.with_columns(assertions).with_columns(
        pl.all_horizontal([pl.col(f"{field}_correct") for field in TARGET_FIELDS])
        .alias("exact_row_correct")
    )
    return BenchmarkComparison(
        rows=joined,
        annotated_count=reference.height,
        prediction_count=predictions.height,
        extra_prediction_keys=extra_keys,
    )


def summarize_metrics(comparison: BenchmarkComparison) -> dict[str, Any]:
    """Return coverage, field, exact-row, and classification metrics."""

    rows = comparison.rows
    matched = rows.filter(pl.col("_prediction_present"))
    fields: dict[str, dict[str, float | int | None]] = {}
    for field in TARGET_FIELDS:
        correct = int(rows[f"{field}_correct"].sum())
        evaluated = rows.height
        fields[field] = {
            "correct": correct,
            "evaluated": evaluated,
            "accuracy": correct / evaluated if evaluated else None,
        }
    exact_correct = int(rows["exact_row_correct"].sum())

    per_type: dict[str, dict[str, float | int | None]] = {}
    for operation_type in CANONICAL_TYPES:
        type_rows = rows.filter(pl.col("typeOperation") == operation_type)
        correct = int(type_rows["typeOperation_correct"].sum())
        per_type[operation_type] = {
            "support": type_rows.height,
            "correct": correct,
            "accuracy": correct / type_rows.height if type_rows.height else None,
            "recall": correct / type_rows.height if type_rows.height else None,
        }

    labels = (*CANONICAL_TYPES, "__UNKNOWN__", "__MISSING__")
    confusion = {reference: {prediction: 0 for prediction in labels} for reference in CANONICAL_TYPES}
    for row in rows.select("typeOperation", "typeOperation_prediction", "_prediction_present").iter_rows(named=True):
        reference_type = row["typeOperation"]
        if reference_type not in CANONICAL_TYPES:
            continue
        predicted = row["typeOperation_prediction"]
        bucket = (
            "__MISSING__" if not row["_prediction_present"] or predicted is None
            else predicted if predicted in CANONICAL_TYPES
            else "__UNKNOWN__"
        )
        confusion[reference_type][bucket] += 1

    return {
        "coverage": {
            "annotated_rows": comparison.annotated_count,
            "prediction_rows": comparison.prediction_count,
            "matched_rows": matched.height,
            "missing_predictions": comparison.annotated_count - matched.height,
            "extra_predictions": len(comparison.extra_prediction_keys),
            "extra_prediction_keys": list(comparison.extra_prediction_keys),
        },
        "fields": fields,
        "exact_row": {
            "correct": exact_correct,
            "evaluated": rows.height,
            "accuracy": exact_correct / rows.height if rows.height else None,
        },
        "classification": {
            "overall_accuracy": fields["typeOperation"]["accuracy"],
            "per_reference_type": per_type,
            "confusion_matrix": confusion,
        },
    }
