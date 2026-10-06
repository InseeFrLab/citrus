"""Évaluation : lancer l'extraction en un appel sur des opérations annotées.

Tire un échantillon (ou la totalité) des opérations du fichier d'annotations,
éventuellement filtrées sur le type annoté, envoie chaque annonce à
`extract.run_extraction` avec le prompt de l'approche choisie (`--approche`,
`metier` par défaut, ou `juridique`), puis affiche les métriques principales et ouvre une
session pour examiner les résultats opération par opération.

Chaque résultat est écrit dès qu'il arrive dans
`artifacts/evaluation/<horodatage>/results.jsonl` : un batch interrompu garde
ce qui a été calculé, et `--load` rouvre un batch sans refaire d'appel LLM.

Le dossier est copié au fil de l'eau sur S3, sous
`s3://projet-citrus/evaluation/<horodatage>/` (`--s3-prefix`, `--no-s3`), pour
survivre à la suppression du service. Le batch est suivi par un run MLflow
(expérience `citrus-evaluation`, `--no-mlflow`) qui porte les paramètres, les
métriques, leur affichage console (artefact `metriques.txt`) et le chemin S3
(tag `s3_uri`) ; les données restent sur S3. Si
Langfuse est configuré, les traces des appels LLM sont regroupées dans la
session `<horodatage>`. S3 et MLflow sont vérifiés avant tout appel LLM.

Usage :
    uv run python evaluate.py                          # 20 opérations au hasard
    uv run python evaluate.py --types VE LG -n 50      # filtrer sur le type annoté
    uv run python evaluate.py --types FUSION --per-type -n 5   # 5 par type de fusion
    uv run python evaluate.py --types TP --all         # toutes les opérations TP
    uv run python evaluate.py --approche juridique     # prompt de lecture juridique
    uv run python evaluate.py --analyse courte         # analyse en 2-3 phrases (plus rapide)
    uv run python evaluate.py --analyse aucune         # pas d'analyse (le plus rapide)
    uv run python evaluate.py --no-reasoning           # sans raisonnement du modèle
    uv run python evaluate.py --load artifacts/evaluation/<horodatage>
    uv run python evaluate.py --load s3://projet-citrus/evaluation/<horodatage>
    uv run python evaluate.py -n 5 --no-s3 --no-mlflow     # essai local

Les fonctions sont aussi importables depuis une cellule VS Code :
    >>> from evaluate import load_batch, evaluate, compute_metrics, format_metrics
    >>> meta, records = load_batch("artifacts/evaluation/<horodatage>")
    >>> annonces, operations = evaluate(records)
    >>> operations.filter(~pl.col("montantNet_ok"))
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import logging

import polars as pl

from extract import (
    _ANNOTATION_COLUMNS,
    ANNOTATION_ID_COLUMN,
    ANALYSIS_MODES,
    ANNOTATIONS_PATH,
    APPROACHES,
    COMPARED_FIELDS,
    DEFAULT_ANALYSIS,
    DEFAULT_APPROACH,
    OPERATION_CODES,
    PROMPT_VERSIONS,
    _pair_operations,
    _reference_value,
    format_annonce,
    format_comparison,
    format_result,
    load_annotations,
    reasoning_label,
    run_extraction,
)
from src import console_handler
from src.bodacc.api import BodaccFetchError
from src.llm.client import flush_traces, get_model_name, langfuse_enabled
from src.s3 import S3Sync, S3SyncError, download_files
from src.tracking import MlflowCredentialsError, setup_mlflow


FUSION_FAMILY = ("FU", "AB", "SP", "AP", "ST")
TYPE_ALIASES = {"FUSION": FUSION_FAMILY, "FUSIONS": FUSION_FAMILY}

DEFAULT_SAMPLE_SIZE = 20
DEFAULT_SEED = 0
DEFAULT_WORKERS = 4
# Au-delà, le lancement demande confirmation (sauf --yes).
CONFIRMATION_THRESHOLD = 100
# Tolérance relative de l'indicateur « montant proche », en plus du taux exact.
AMOUNT_RELATIVE_TOLERANCE = 0.10

OUTPUT_ROOT = Path(__file__).resolve().parent / "artifacts" / "evaluation"
# Les batchs rouverts depuis S3 sont téléchargés ici, sans écraser un dossier local.
S3_DOWNLOAD_ROOT = OUTPUT_ROOT / "s3"
DEFAULT_S3_PREFIX = "s3://projet-citrus/evaluation"
DEFAULT_EXPERIMENT = "citrus-evaluation"
RESULTS_FILE = "results.jsonl"
META_FILE = "meta.json"
# Les métriques telles qu'affichées en console, gardées en artefact (S3 et MLflow).
METRICS_TEXT_FILE = "metriques.txt"

NON_RETENU = "non retenu"
SANS_TYPE = "sans type"

# Une lettre par champ comparé, pour la liste compacte de la session.
FIELD_LETTERS = dict(zip(COMPARED_FIELDS, "TCBERM"))
FIELD_ALIASES = {
    "type": "typeOperation",
    "cedant": "sirenCedant",
    "beneficiaire": "sirenBeneficiaire",
    "effet": "dateEffetComptable",
    "realisation": "dateRealisationJuridique",
    "montant": "montantNet",
    **{field.lower(): field for field in COMPARED_FIELDS},
    **{letter.lower(): field for field, letter in FIELD_LETTERS.items()},
}


# --------------------------------------------------------------------------
# Sélection des opérations
# --------------------------------------------------------------------------


def parse_types(values: Iterable[str] | None) -> tuple[str, ...]:
    """Codes de type demandés ; `FUSION` vaut FU, AB, SP, AP, ST. Aucun = tous."""

    if not values:
        return OPERATION_CODES
    selected: list[str] = []
    for value in values:
        for token in value.replace(",", " ").split():
            for code in TYPE_ALIASES.get(token.upper(), (token.upper(),)):
                if code not in OPERATION_CODES:
                    raise ValueError(
                        f"type inconnu : {token} "
                        f"(attendus : {', '.join(OPERATION_CODES)}, FUSION)"
                    )
                if code not in selected:
                    selected.append(code)
    return tuple(selected)


def _sample(frame: pl.DataFrame, size: int, seed: int) -> pl.DataFrame:
    return frame if frame.height <= size else frame.sample(n=size, seed=seed)


def select_annonces(
    annotations: pl.DataFrame,
    types: Sequence[str],
    sample_size: int | None,
    *,
    per_type: bool = False,
    seed: int = DEFAULT_SEED,
) -> list[str]:
    """Tirer des opérations annotées et renvoyer leurs annonces, triées.

    `sample_size=None` prend toutes les opérations des types demandés. Avec
    `per_type`, la taille s'applique à chaque type au lieu du total. Le tirage
    part d'un ordre stable : même graine, même échantillon.
    """

    pool = annotations.filter(pl.col("type_op").is_in(list(types))).sort(
        ANNOTATION_ID_COLUMN, maintain_order=True
    )
    if sample_size is not None:
        if per_type:
            pool = pl.concat(
                [
                    _sample(pool.filter(pl.col("type_op") == code), sample_size, seed)
                    for code in types
                ]
            )
        else:
            pool = _sample(pool, sample_size, seed)
    return sorted(set(pool[ANNOTATION_ID_COLUMN].to_list()))


def reference_rows(annotations: pl.DataFrame, annonce_id: str) -> list[dict[str, Any]]:
    """Lignes annotées d'une annonce, déjà ramenées au format de comparaison.

    Les valeurs sont stockées sous les noms de colonnes d'origine : les
    fonctions de comparaison de `extract` s'appliquent telles quelles, y compris
    sur un batch rechargé depuis le disque.
    """

    rows = annotations.filter(pl.col(ANNOTATION_ID_COLUMN) == annonce_id)
    return [
        {
            "id_operation": row.get("id_operation"),
            **{
                column: _reference_value(field, row)
                for field, column in _ANNOTATION_COLUMNS.items()
            },
        }
        for row in rows.iter_rows(named=True)
    ]


# --------------------------------------------------------------------------
# Exécution
# --------------------------------------------------------------------------


def _annotated_types(references: Sequence[dict[str, Any]]) -> str:
    return ",".join(sorted({row["type_op"] for row in references if row.get("type_op")}))


def run_one(
    annonce_id: str,
    references: list[dict[str, Any]],
    *,
    approach: str = DEFAULT_APPROACH,
    analysis: str = DEFAULT_ANALYSIS,
    fetch: Callable[[str], dict[str, Any]] | None = None,
    ask_fn: Callable[..., str] | None = None,
    trace_session: str | None = None,
    **ask_options: Any,
) -> dict[str, Any]:
    """Traiter une annonce ; une erreur est enregistrée au lieu d'arrêter le batch.

    `trace_session` regroupe les traces Langfuse du batch ; chaque trace porte
    aussi les types annotés de l'annonce.
    """

    record: dict[str, Any] = {
        "annonce_id": annonce_id,
        "references": references,
        "statut": "ok",
        "erreur": None,
        "result": None,
    }
    try:
        result = run_extraction(
            annonce_id,
            approach=approach,
            analysis=analysis,
            fetch=fetch,
            ask_fn=ask_fn,
            trace_session=trace_session,
            trace_metadata={"type_annote": _annotated_types(references)},
            **ask_options,
        )
    except BodaccFetchError as error:
        record.update(statut="erreur BODACC", erreur=f"[{error.code}] {error.detail}")
    except Exception as error:  # noqa: BLE001 - une annonce en échec n'arrête pas le batch
        record.update(statut="erreur", erreur=f"{type(error).__name__}: {error}")
    else:
        # Le prompt système est le même pour toutes les annonces : inutile de le stocker.
        result.pop("messages", None)
        record["result"] = result
    return record


def _progress_line(done: int, total: int, record: dict[str, Any]) -> str:
    prefix = f"  [{done:>{len(str(total))}}/{total}] {record['annonce_id']:<16}"
    if record["result"] is None:
        return f"{prefix} ✗ {record['statut']} : {record['erreur']}"
    annotated = record["references"][0]["type_op"] if record["references"] else "?"
    predicted = _predicted_type(record["result"]["envelope"])
    mark = "✓" if predicted == annotated else "✗"
    return (
        f"{prefix} {mark} annoté {annotated:<3} prédit {predicted:<10}"
        f" {record['result']['elapsed_seconds']} s"
    )


def run_batch(
    annonce_ids: Sequence[str],
    annotations: pl.DataFrame,
    *,
    approach: str = DEFAULT_APPROACH,
    analysis: str = DEFAULT_ANALYSIS,
    workers: int = DEFAULT_WORKERS,
    output_dir: Path | None = None,
    fetch: Callable[[str], dict[str, Any]] | None = None,
    ask_fn: Callable[..., str] | None = None,
    progress: Callable[[str], None] = print,
    on_record: Callable[[], None] | None = None,
    trace_session: str | None = None,
    **ask_options: Any,
) -> list[dict[str, Any]]:
    """Lancer `run_extraction` sur chaque annonce, en parallèle.

    Chaque résultat est ajouté à `output_dir/results.jsonl` dès réception, puis
    `on_record` est appelé (copie S3 au fil de l'eau), depuis le thread principal.
    Ctrl-C annule les annonces pas encore parties et renvoie ce qui est acquis.
    """

    results_file = None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        results_file = (output_dir / RESULTS_FILE).open("a", encoding="utf-8")

    records: list[dict[str, Any]] = []
    executor = ThreadPoolExecutor(max_workers=max(1, workers))
    futures = [
        executor.submit(
            run_one,
            annonce_id,
            reference_rows(annotations, annonce_id),
            approach=approach,
            analysis=analysis,
            fetch=fetch,
            ask_fn=ask_fn,
            trace_session=trace_session,
            **ask_options,
        )
        for annonce_id in annonce_ids
    ]
    try:
        for done, future in enumerate(as_completed(futures), 1):
            record = future.result()
            records.append(record)
            if results_file is not None:
                results_file.write(
                    json.dumps(record, ensure_ascii=False, default=str) + "\n"
                )
                results_file.flush()
                if on_record is not None:
                    on_record()
            progress(_progress_line(done, len(futures), record))
    except KeyboardInterrupt:
        progress(
            f"  interruption : {len(records)}/{len(futures)} annonces traitées ; "
            "les appels déjà partis se terminent en arrière-plan"
        )
        executor.shutdown(wait=False, cancel_futures=True)
    else:
        executor.shutdown()
    finally:
        if results_file is not None:
            results_file.close()
    return sorted(records, key=lambda record: record["annonce_id"])


def write_meta(output_dir: Path, meta: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / META_FILE).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )


def load_batch(directory: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Relire un batch enregistré : ses métadonnées et ses résultats."""

    directory = Path(directory)
    meta_path = directory / META_FILE
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    records = [
        json.loads(line)
        for line in (directory / RESULTS_FILE).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return meta, sorted(records, key=lambda record: record["annonce_id"])


# --------------------------------------------------------------------------
# Évaluation
# --------------------------------------------------------------------------


def _predicted_type(envelope: dict[str, Any]) -> str:
    if not envelope["retenu"]:
        return NON_RETENU
    return envelope["codeTypeOperation"] or SANS_TYPE


def _amount_close(expected: int | None, obtained: int | None) -> bool:
    if expected is None or obtained is None:
        return expected == obtained
    return abs(expected - obtained) <= AMOUNT_RELATIVE_TOLERANCE * abs(expected)


def _operation_row(
    annonce_id: str,
    operation: dict[str, Any] | None,
    reference: dict[str, Any] | None,
    statut: str,
) -> dict[str, Any]:
    compared = operation is not None and reference is not None
    row: dict[str, Any] = {
        "annonce_id": annonce_id,
        "statut": statut,
        "type_annote": reference["type_op"] if reference else None,
        "type_predit": operation["typeOperation"] if operation else None,
    }
    all_exact = compared
    for field in COMPARED_FIELDS:
        expected = _reference_value(field, reference) if reference else None
        obtained = operation.get(field) if operation else None
        row[f"{field}_annote"] = expected
        row[f"{field}_predit"] = obtained
        row[f"{field}_ok"] = expected == obtained if compared else None
        all_exact = all_exact and expected == obtained
    row["montant_proche"] = (
        _amount_close(row["montantNet_annote"], row["montantNet_predit"])
        if compared
        else None
    )
    row["tout_exact"] = all_exact if compared else None
    return row


def _operation_schema() -> dict[str, pl.DataType]:
    schema: dict[str, Any] = {
        "annonce_id": pl.String,
        "statut": pl.String,
        "type_annote": pl.String,
        "type_predit": pl.String,
    }
    for field in COMPARED_FIELDS:
        value_type = pl.Int64 if field == "montantNet" else pl.String
        schema[f"{field}_annote"] = value_type
        schema[f"{field}_predit"] = value_type
        schema[f"{field}_ok"] = pl.Boolean
    schema["montant_proche"] = pl.Boolean
    schema["tout_exact"] = pl.Boolean
    return schema


WARNING_KINDS_SHOWN = 12

ANNONCE_SCHEMA = {
    "annonce_id": pl.String,
    "type_annote": pl.String,
    "statut": pl.String,
    "erreur": pl.String,
    "retenu": pl.Boolean,
    "type_predit": pl.String,
    "n_ops_annotees": pl.Int64,
    "n_ops_predites": pl.Int64,
    "reponse_vide": pl.Boolean,
    "alertes": pl.Int64,
    "alertes_luhn": pl.Int64,
    "types_alertes": pl.List(pl.String),
    "duree_s": pl.Float64,
}


def warning_kind(warning: str) -> str:
    """Réduire un avertissement de `extract.py` à sa nature, pour les compter.

    Le contexte (lecture, numéro d'opération) et la valeur fautive sont retirés,
    le champ est gardé : « lecture juridique op.2 dateEffetComptable n'est pas
    une date reconnue : 'mars 2023' » devient « dateEffetComptable n'est pas une
    date reconnue ».
    """

    kind = re.sub(r"^.*?\bop\.\d+(?: :)? ", "", warning.strip())
    kind = re.sub(r"=\S+", "", kind)
    kind = re.sub(r" : (?:['\"\[{(\d-].*|\w+)$", "", kind)
    kind = re.sub(r"^[\w ]+ : ", "", kind)
    return re.sub(r" \(.*\)$", "", kind)


def evaluate(records: Sequence[dict[str, Any]]) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Deux tables : une ligne par annonce, une ligne par opération.

    Les opérations prédites sont appariées aux lignes
    annotées comme dans la session de `extract.py` : par couple de SIREN, sinon par ordre.
    """

    annonce_rows: list[dict[str, Any]] = []
    operation_rows: list[dict[str, Any]] = []
    for record in records:
        references = record["references"]
        row: dict[str, Any] = {
            "annonce_id": record["annonce_id"],
            "type_annote": references[0]["type_op"] if references else None,
            "statut": record["statut"],
            "erreur": record["erreur"],
            "n_ops_annotees": len(references),
            "alertes": 0,
            "alertes_luhn": 0,
            "types_alertes": [],
        }
        result = record["result"]
        if result is None:
            annonce_rows.append(row)
            operation_rows.extend(
                _operation_row(record["annonce_id"], None, reference, "erreur")
                for reference in references
            )
            continue

        envelope = result["envelope"]
        operations = envelope["operations"]
        warnings = result["warnings"]
        row.update(
            retenu=envelope["retenu"],
            type_predit=_predicted_type(envelope),
            n_ops_predites=len(operations),
            reponse_vide=any("réponse LLM vide" in warning for warning in warnings),
            alertes=len(warnings),
            alertes_luhn=sum("Luhn" in warning for warning in warnings),
            types_alertes=[warning_kind(warning) for warning in warnings],
            duree_s=result["elapsed_seconds"],
        )
        annonce_rows.append(row)
        for operation, reference in _pair_operations(operations, references):
            if operation is not None and reference is not None:
                statut = "appariée"
            elif reference is None:
                statut = "non annotée"
            else:
                statut = "non prédite"
            operation_rows.append(
                _operation_row(record["annonce_id"], operation, reference, statut)
            )

    annonces = pl.DataFrame(annonce_rows, schema=ANNONCE_SCHEMA)
    operations = pl.DataFrame(operation_rows, schema=_operation_schema())
    operations = operations.with_row_index("n", offset=1)
    return annonces, operations


def _warning_kinds(done: pl.DataFrame) -> list[dict[str, Any]]:
    """Par nature d'avertissement : occurrences et annonces touchées, du plus fréquent."""

    kinds = (
        done.select("annonce_id", "types_alertes")
        .explode("types_alertes")
        .drop_nulls("types_alertes")
        .group_by("types_alertes")
        .agg(pl.len().alias("occurrences"), pl.col("annonce_id").n_unique().alias("annonces"))
        .sort(["occurrences", "types_alertes"], descending=[True, False])
    )
    return [
        {"type": row["types_alertes"], "occurrences": row["occurrences"], "annonces": row["annonces"]}
        for row in kinds.iter_rows(named=True)
    ]


def compute_metrics(annonces: pl.DataFrame, operations: pl.DataFrame) -> dict[str, Any]:
    """Compter tout ce que `format_metrics` affiche."""

    done = annonces.filter(pl.col("statut") == "ok")
    paired = operations.filter(pl.col("statut") == "appariée")
    durations = done["duree_s"].drop_nulls().to_list()

    per_type = []
    for code in OPERATION_CODES:
        subset = annonces.filter(pl.col("type_annote") == code)
        if not subset.height:
            continue
        subset_done = subset.filter(pl.col("statut") == "ok")
        subset_paired = paired.filter(pl.col("type_annote") == code)
        per_type.append(
            {
                "type": code,
                "annonces": subset.height,
                "ok": subset_done.height,
                "retenu": int(subset_done["retenu"].sum()),
                "type_ok": int((subset_done["type_predit"] == code).sum()),
                "operations_appariees": subset_paired.height,
                "champs": {
                    field: int(subset_paired[f"{field}_ok"].sum())
                    for field in COMPARED_FIELDS
                },
            }
        )

    fields = {}
    for field in COMPARED_FIELDS:
        filled = paired.filter(pl.col(f"{field}_annote").is_not_null())
        empty = paired.filter(pl.col(f"{field}_annote").is_null())
        fields[field] = {
            "exact": int(paired[f"{field}_ok"].sum()),
            "annote_renseigne": filled.height,
            "exact_si_renseigne": int(filled[f"{field}_ok"].sum()),
            "annote_vide": empty.height,
            "vide_si_vide": int(empty[f"{field}_predit"].is_null().sum()),
        }

    confusion: dict[tuple[str, str], int] = {}
    for annotated, predicted in done.select("type_annote", "type_predit").iter_rows():
        confusion[(annotated, predicted)] = confusion.get((annotated, predicted), 0) + 1

    statuts = dict(annonces.group_by("statut").len().iter_rows())
    operation_statuts = dict(operations.group_by("statut").len().iter_rows())
    return {
        "annonces": annonces.height,
        "statuts": statuts,
        "ok": done.height,
        "reponses_vides": int(done["reponse_vide"].sum()),
        "retenu": int(done["retenu"].sum()),
        "type_ok": int((done["type_predit"] == done["type_annote"]).sum()),
        "nombre_operations_ok": int(
            (done["n_ops_predites"] == done["n_ops_annotees"]).sum()
        ),
        "operations": operations.height,
        "operation_statuts": operation_statuts,
        "operations_appariees": paired.height,
        "tout_exact": int(paired["tout_exact"].sum()),
        "montant_proche": int(paired["montant_proche"].sum()),
        "champs": fields,
        "par_type": per_type,
        "confusion": confusion,
        "alertes_luhn": int(done["alertes_luhn"].sum()),
        "annonces_avec_alertes": int((done["alertes"] > 0).sum()),
        "alertes": int(done["alertes"].sum()),
        "types_alertes": _warning_kinds(done),
        "duree_mediane_s": statistics.median(durations) if durations else None,
        "duree_totale_llm_s": sum(durations),
    }


# --------------------------------------------------------------------------
# Métriques MLflow
# --------------------------------------------------------------------------


def _share(count: int, total: int) -> float | None:
    return count / total if total else None


def durations_by_type(annonces: pl.DataFrame) -> pl.DataFrame:
    """Temps d'inférence des annonces traitées, par type annoté."""

    return (
        annonces.filter((pl.col("statut") == "ok") & pl.col("duree_s").is_not_null())
        .group_by("type_annote")
        .agg(
            pl.len().alias("n"),
            pl.col("duree_s").mean().alias("duree_moyenne_s"),
            pl.col("duree_s").median().alias("duree_mediane_s"),
        )
        .sort("type_annote")
    )


def mlflow_metrics(
    metrics: dict[str, Any], meta: dict[str, Any], annonces: pl.DataFrame
) -> dict[str, float]:
    """Aplatir `compute_metrics` en métriques numériques pour MLflow.

    Les taux ont les mêmes dénominateurs que `format_metrics` : les annonces
    traitées (`ok`) pour le niveau annonce, les opérations appariées pour les champs.
    Un taux sans dénominateur est omis plutôt que mis à 0.
    """

    ok = metrics["ok"]
    paired = metrics["operations_appariees"]
    values: dict[str, float | None] = {
        "annonces": metrics["annonces"],
        "annonces_ok": ok,
        "erreurs": metrics["annonces"] - ok,
        "reponses_vides": metrics["reponses_vides"],
        "alertes_luhn": metrics["alertes_luhn"],
        "alertes": metrics["alertes"],
        "annonces_avec_alertes": metrics["annonces_avec_alertes"],
        "taux_ok": _share(ok, metrics["annonces"]),
        "taux_retenu": _share(metrics["retenu"], ok),
        "taux_type_ok": _share(metrics["type_ok"], ok),
        "taux_nombre_operations_ok": _share(metrics["nombre_operations_ok"], ok),
        "operations_appariees": paired,
        "taux_tout_exact": _share(metrics["tout_exact"], paired),
        "taux_montant_proche": _share(metrics["montant_proche"], paired),
        "duree_mediane_s": metrics["duree_mediane_s"],
        "duree_totale_llm_s": metrics["duree_totale_llm_s"],
        "duree_totale_s": meta.get("duree_totale_s"),
    }
    for field in COMPARED_FIELDS:
        values[f"taux_{field}_exact"] = _share(metrics["champs"][field]["exact"], paired)
    for row in metrics["par_type"]:
        values[f"taux_type_ok_{row['type']}"] = _share(row["type_ok"], row["ok"])

    done = annonces.filter((pl.col("statut") == "ok") & pl.col("duree_s").is_not_null())
    values["duree_moyenne_s"] = done["duree_s"].mean() if done.height else None
    for row in durations_by_type(annonces).iter_rows(named=True):
        values[f"duree_moyenne_s_{row['type_annote']}"] = row["duree_moyenne_s"]

    return {name: float(value) for name, value in values.items() if value is not None}


# --------------------------------------------------------------------------
# Affichage
# --------------------------------------------------------------------------


def _ratio(count: int, total: int) -> str:
    if not total:
        return "—"
    return f"{count}/{total} ({100 * count / total:.1f} %)"


def _rate(count: int, total: int) -> str:
    return "—" if not total else f"{100 * count / total:.0f} %"


def format_confusion(confusion: dict[tuple[str, str], int]) -> str:
    """Matrice de confusion : type annoté en lignes, type prédit en colonnes."""

    if not confusion:
        return "  (aucune annonce traitée)"
    annotated = [code for code in OPERATION_CODES if any(a == code for a, _ in confusion)]
    predicted_seen = {p for _, p in confusion}
    predicted = [code for code in OPERATION_CODES if code in predicted_seen]
    predicted += sorted(predicted_seen - set(OPERATION_CODES))
    width = max(5, *(len(code) for code in predicted))
    lines = [
        "  annoté \\ prédit  " + " ".join(f"{code:>{width}}" for code in predicted)
    ]
    for row in annotated:
        cells = []
        for column in predicted:
            count = confusion.get((row, column), 0)
            cells.append(f"{(str(count) if count else '·'):>{width}}")
        lines.append(f"  {row:<16} " + " ".join(cells))
    return "\n".join(lines)


def format_metrics(metrics: dict[str, Any], meta: dict[str, Any] | None = None) -> str:
    meta = meta or {}
    ok = metrics["ok"]
    label = APPROACHES.get(meta.get("approche"), "?")
    lines = [
        "═" * 78,
        f"  Évaluation — approche {label}"
        f", prompt {meta.get('prompt_version', '?')}"
        f", analyse {ANALYSIS_MODES.get(meta.get('analyse'), '?')}"
        f", modèle {meta.get('modele', '?')}"
        f", raisonnement {meta.get('raisonnement', '?')}",
    ]
    if meta.get("types"):
        sampling = (
            "toutes les opérations"
            if meta.get("sample_size") is None
            else f"échantillon de {meta['sample_size']}"
            + (" par type" if meta.get("per_type") else "")
            + f", graine {meta.get('seed')}"
        )
        lines.append(f"  types {', '.join(meta['types'])} — {sampling}")
    lines.append("  (référence : annotations, produites selon les règles métier)")
    lines.append("═" * 78)

    statuts = ", ".join(f"{name} {count}" for name, count in sorted(metrics["statuts"].items()))
    lines.append(f"  Annonces                   {metrics['annonces']}  ({statuts})")
    if metrics["reponses_vides"]:
        lines.append(f"  Réponses non analysables   {metrics['reponses_vides']}")
    if metrics["duree_mediane_s"] is not None:
        wall = meta.get("duree_totale_s")
        lines.append(
            f"  Durée LLM                  médiane {metrics['duree_mediane_s']:.1f} s"
            f", cumul {metrics['duree_totale_llm_s']:.0f} s"
            + (f", batch {wall:.0f} s" if wall else "")
        )

    lines += [
        "",
        "  Sur les annonces traitées :",
        f"  Retenue comme restructuration  {_ratio(metrics['retenu'], ok)}",
        f"  Type exact                     {_ratio(metrics['type_ok'], ok)}",
        f"  Nombre d'opérations exact      {_ratio(metrics['nombre_operations_ok'], ok)}",
        "",
        f"  Avertissements de normalisation : {metrics['alertes']}"
        f" sur {_ratio(metrics['annonces_avec_alertes'], ok)} annonce(s)",
    ]
    if metrics["types_alertes"]:
        lines.append("  occurrences  annonces  nature")
        for row in metrics["types_alertes"][:WARNING_KINDS_SHOWN]:
            lines.append(f"  {row['occurrences']:>11}  {row['annonces']:>8}  {row['type']}")
        hidden = metrics["types_alertes"][WARNING_KINDS_SHOWN:]
        if hidden:
            lines.append(
                f"  {sum(row['occurrences'] for row in hidden):>11}"
                f"  {'':>8}  … {len(hidden)} autre(s) nature(s)"
            )
    lines += [
        "",
        "  Par type annoté        n  retenu   type ok",
    ]
    for row in metrics["par_type"]:
        lines.append(
            f"  {row['type']:<16} {row['annonces']:>5}"
            f"  {_rate(row['retenu'], row['ok']):>6}"
            f"  {_rate(row['type_ok'], row['ok']):>8}"
        )

    lines += ["", f"  Matrice de confusion ({label})", format_confusion(metrics["confusion"])]

    paired = metrics["operations_appariees"]
    op_statuts = ", ".join(
        f"{name} {count}" for name, count in sorted(metrics["operation_statuts"].items())
    )
    lines += [
        "",
        f"  Champs — {paired} opération(s) appariée(s) sur {metrics['operations']}"
        f"  ({op_statuts})",
        "  champ                        exact            annoté renseigné   annoté vide",
    ]
    for field, counts in metrics["champs"].items():
        lines.append(
            f"  {field:<28} {_ratio(counts['exact'], paired):<17}"
            f" {_rate(counts['exact_si_renseigne'], counts['annote_renseigne']):>5}"
            f" sur {counts['annote_renseigne']:<8}"
            f" {_rate(counts['vide_si_vide'], counts['annote_vide']):>5}"
            f" sur {counts['annote_vide']}"
        )
    tolerance = int(AMOUNT_RELATIVE_TOLERANCE * 100)
    lines += [
        f"  {'montantNet à ±' + str(tolerance) + ' %':<28} {_ratio(metrics['montant_proche'], paired)}",
        f"  {'opération entièrement exacte':<28} {_ratio(metrics['tout_exact'], paired)}",
        "  (« annoté renseigné » : exact quand l'annotation a une valeur ;"
        " « annoté vide » : prédit vide aussi)",
    ]

    if metrics["par_type"]:
        header = "  exact par type    " + " ".join(
            f"{FIELD_LETTERS[field]:>5}" for field in COMPARED_FIELDS
        )
        lines += ["", header]
        for row in metrics["par_type"]:
            cells = " ".join(
                f"{_rate(row['champs'][field], row['operations_appariees']):>5}"
                for field in COMPARED_FIELDS
            )
            lines.append(f"  {row['type']:<16} {cells}   ({row['operations_appariees']} op.)")
        lines.append(
            "  " + "  ".join(f"{letter}={field}" for field, letter in FIELD_LETTERS.items())
        )
    lines.append("═" * 78)
    return "\n".join(lines)


def _mark(value: bool | None) -> str:
    return "·" if value is None else ("✓" if value else "✗")


def format_operation_list(operations: pl.DataFrame) -> str:
    header = "     n  annonce          annoté  prédit      " + " ".join(
        FIELD_LETTERS.values()
    ) + "  statut"
    lines = [header]
    for row in operations.iter_rows(named=True):
        marks = " ".join(_mark(row[f"{field}_ok"]) for field in COMPARED_FIELDS)
        statut = "" if row["statut"] == "appariée" else row["statut"]
        lines.append(
            f"  {row['n']:>4}  {row['annonce_id']:<16} {row['type_annote'] or '—':<7}"
            f" {row['type_predit'] or '—':<11} {marks}  {statut}"
        )
    lines.append(f"  ({operations.height} opération(s))")
    return "\n".join(lines)


def filter_operations(operations: pl.DataFrame, tokens: Sequence[str]) -> pl.DataFrame:
    """Filtres de `:liste` : un code de type, un nom de champ (erreur sur ce champ), `erreurs`."""

    for token in tokens:
        upper, lower = token.upper(), token.lower()
        if upper in OPERATION_CODES or upper in TYPE_ALIASES:
            codes = TYPE_ALIASES.get(upper, (upper,))
            operations = operations.filter(pl.col("type_annote").is_in(list(codes)))
        elif lower in FIELD_ALIASES:
            operations = operations.filter(pl.col(f"{FIELD_ALIASES[lower]}_ok") == False)  # noqa: E712
        elif lower in {"erreurs", "erreur", "ko"}:
            operations = operations.filter(pl.col("tout_exact").fill_null(False).not_())
        else:
            raise ValueError(f"filtre inconnu : {token}")
    return operations


BROWSE_HELP = """Commandes :
  <n> ou <id>        détail d'une opération (réponse du LLM, comparaison, analyse)
  :liste [filtres]   lister les opérations ; filtres cumulables :
                       un type (VE, LG, FUSION…), un champ en erreur
                       (type, cedant, beneficiaire, effet, realisation, montant,
                       ou sa lettre de colonne T C B E R M),
                       ou « erreurs » (toute opération pas entièrement exacte)
  :texte <n|id>      l'annonce en clair
  :payload <n|id>    le JSON BODACC de l'annonce
  :raw <n|id>        la réponse brute du LLM

  <id> : identifiant d'annonce, sans tenir compte de la casse ; un fragment
  suffit s'il ne désigne qu'une annonce (A202301491199, a202301491199,
  1491199…). Un nombre qui est un numéro d'opération de :liste désigne
  l'opération.
  :metriques         réafficher les métriques
  :help              cette aide
  :quit              quitter (ou Ctrl-D)"""


def _record_for(
    entry: str, operations: pl.DataFrame, records: dict[str, dict[str, Any]]
) -> tuple[dict[str, Any], int | None]:
    """Retrouver l'annonce désignée par un numéro d'opération ou un identifiant.

    Ordre de priorité : identifiant exact, numéro d'opération, puis identifiant
    sans tenir compte de la casse, entier ou fragment unique. `LookupError`
    porte le message à afficher quand rien ou plusieurs annonces correspondent.
    """

    entry = entry.strip()
    if not entry:
        raise LookupError("opération ou annonce manquante")
    if entry in records:
        return records[entry], None
    if entry.isdigit():
        rows = operations.filter(pl.col("n") == int(entry))
        if rows.height:
            return records[rows["annonce_id"][0]], int(entry)
    needle = entry.casefold()
    matches = [
        annonce_id for annonce_id in records if annonce_id.casefold() == needle
    ] or [annonce_id for annonce_id in records if needle in annonce_id.casefold()]
    if len(matches) == 1:
        return records[matches[0]], None
    if not matches:
        raise LookupError(f"opération ou annonce introuvable : {entry}")
    shown = ", ".join(sorted(matches)[:10]) + (" …" if len(matches) > 10 else "")
    raise LookupError(f"{len(matches)} annonces correspondent à {entry} : {shown}")


def format_detail(record: dict[str, Any], n: int | None = None) -> str:
    title = f"opération {n} — " if n is not None else ""
    lines = [f"── {title}annonce {record['annonce_id']} ({record['statut']})"]
    result = record["result"]
    if result is None:
        lines.append(f"  ✗ {record['erreur']}")
        lines.append(format_comparison([], record["references"]))
        return "\n".join(lines)
    if result.get("payload", {}).get("url_complete"):
        lines.append(f"  {result['payload']['url_complete']}")
    lines.append(format_result(result))
    lines.append(
        format_comparison(result["envelope"]["operations"], record["references"])
    )
    return "\n".join(lines)


def browse(
    records: Sequence[dict[str, Any]],
    operations: pl.DataFrame,
    metrics_text: str,
) -> int:
    """Session pour parcourir les résultats opération par opération."""

    by_id = {record["annonce_id"]: record for record in records}
    print("\nTape :liste pour voir les opérations, :help pour les commandes.\n")
    while True:
        try:
            entry = input("batch> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not entry:
            continue
        command, _, argument = entry.partition(" ")
        argument = argument.strip()
        if command in {":quit", ":q"}:
            return 0
        if command == ":help":
            print(BROWSE_HELP)
        elif command == ":metriques":
            print(metrics_text)
        elif command == ":liste":
            try:
                selected = filter_operations(operations, argument.split())
            except ValueError as error:
                print(f"  {error}")
                continue
            print(format_operation_list(selected))
        elif command in {":texte", ":payload", ":raw"}:
            try:
                record, _ = _record_for(argument, operations, by_id)
            except LookupError as error:
                print(f"  {error}")
                continue
            if record["result"] is None:
                print(f"  ✗ {record['erreur']}")
            elif command == ":texte":
                print(format_annonce(record["result"]["payload"]))
            elif command == ":payload":
                print(json.dumps(record["result"]["payload"], ensure_ascii=False, indent=2))
            else:
                print(record["result"]["raw_answer"])
        elif command.startswith(":"):
            print(f"  commande inconnue : {command}")
        else:
            try:
                record, n = _record_for(entry, operations, by_id)
            except LookupError as error:
                print(f"  {error}")
                continue
            print(format_detail(record, n))


# --------------------------------------------------------------------------
# Ligne de commande
# --------------------------------------------------------------------------


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Lancer l'extraction en un appel sur des opérations annotées et l'évaluer"
    )
    parser.add_argument(
        "--types",
        nargs="+",
        help="types annotés à garder (VE, LG, TP, FU, AB, SP, ST, AP, ou FUSION) ; défaut : tous",
    )
    size = parser.add_mutually_exclusive_group()
    size.add_argument(
        "-n",
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help=f"nombre d'opérations tirées (défaut : {DEFAULT_SAMPLE_SIZE})",
    )
    parser.add_argument(
        "--per-type",
        action="store_true",
        help="appliquer --sample-size à chaque type plutôt qu'au total",
    )
    size.add_argument(
        "--all", action="store_true", help="toutes les opérations, sans tirage (exclut -n)"
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED, help=f"graine du tirage (défaut : {DEFAULT_SEED})"
    )
    parser.add_argument(
        "--approche",
        choices=tuple(APPROACHES),
        default=DEFAULT_APPROACH,
        help=f"prompt à utiliser : règles métier ou lecture juridique (défaut : {DEFAULT_APPROACH})",
    )
    parser.add_argument(
        "--analyse",
        choices=tuple(ANALYSIS_MODES),
        default=DEFAULT_ANALYSIS,
        help=(
            "champ d'explication demandé au modèle : complete (étape par étape), "
            "courte (2-3 phrases) ou aucune ; moins de texte = réponse plus rapide, "
            f"mais moins de matière pour comprendre les erreurs (défaut : {DEFAULT_ANALYSIS})"
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"appels LLM simultanés (défaut : {DEFAULT_WORKERS})",
    )
    parser.add_argument(
        "--no-reasoning",
        action="store_true",
        help="désactiver le raisonnement du modèle (plus rapide, réponse directe)",
    )
    parser.add_argument("--temperature", type=float, default=None, help="température du LLM")
    parser.add_argument(
        "--annotations",
        default=ANNOTATIONS_PATH,
        help=f"fichier d'opérations vérifiées (défaut : {ANNOTATIONS_PATH})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="dossier des résultats (défaut : artifacts/evaluation/<horodatage>)",
    )
    parser.add_argument(
        "--load",
        default=None,
        help="rouvrir un batch enregistré, sans appel LLM : dossier local ou URI s3://…",
    )
    parser.add_argument(
        "--s3-prefix",
        default=DEFAULT_S3_PREFIX,
        help=f"copie des résultats sur S3, sous <préfixe>/<nom du dossier> (défaut : {DEFAULT_S3_PREFIX})",
    )
    parser.add_argument("--no-s3", action="store_true", help="ne pas copier les résultats sur S3")
    parser.add_argument(
        "--experiment",
        default=DEFAULT_EXPERIMENT,
        help=f"expérience MLflow du run (défaut : {DEFAULT_EXPERIMENT}, créée si absente)",
    )
    parser.add_argument("--no-mlflow", action="store_true", help="ne pas créer de run MLflow")
    parser.add_argument(
        "--yes", action="store_true", help=f"ne pas demander confirmation au-delà de {CONFIRMATION_THRESHOLD} annonces"
    )
    parser.add_argument(
        "--no-browse", action="store_true", help="afficher les métriques puis quitter"
    )
    return parser


def _confirm(count: int) -> bool:
    if not sys.stdin.isatty():
        return True
    answer = input(f"Lancer {count} appels LLM ? [o/N] ").strip().lower()
    return answer in {"o", "oui", "y", "yes"}


def batch_s3_uri(prefix: str, output_dir: Path) -> str:
    """URI S3 d'un batch : le préfixe suivi du nom de son dossier local."""

    return f"{prefix.rstrip('/')}/{Path(output_dir).name}/"


def _load(source: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Batch enregistré, depuis un dossier local ou une URI S3 (téléchargée d'abord)."""

    if not source.startswith("s3://"):
        return load_batch(source)
    local_dir = S3_DOWNLOAD_ROOT / source.rstrip("/").rsplit("/", 1)[-1]
    download_files(source, local_dir, (META_FILE, RESULTS_FILE))
    print(f"batch téléchargé dans {local_dir}")
    return load_batch(local_dir)


def _run_params(meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "approche": meta["approche"],
        "prompt_version": meta["prompt_version"],
        "analyse": meta["analyse"],
        "modele": meta["modele"],
        "raisonnement": meta["raisonnement"],
        "types": ",".join(meta["types"]),
        "sample_size": meta["sample_size"],
        "per_type": meta["per_type"],
        "seed": meta["seed"],
        "workers": meta["workers"],
        "temperature": meta["ask_options"].get("temperature", "défaut"),
        "annotations": meta["annotations"],
        "annonces": len(meta["annonces"]),
    }


def run_new_batch(
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], pl.DataFrame, str] | int:
    """Tirer l'échantillon, lancer le batch, copier sur S3 et suivre dans MLflow.

    Renvoie (records, operations, métriques affichables), ou un code de sortie si
    le batch n'a pas été lancé. S3 et MLflow sont vérifiés avant tout appel LLM.
    """

    try:
        types = parse_types(args.types)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 2
    annotations = load_annotations(args.annotations)
    sample_size = None if args.all else args.sample_size
    annonce_ids = select_annonces(
        annotations, types, sample_size, per_type=args.per_type, seed=args.seed
    )
    if not annonce_ids:
        print("aucune opération annotée pour ces types", file=sys.stderr)
        return 1
    output_dir = args.output_dir or OUTPUT_ROOT / datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S"
    )
    ask_options: dict[str, Any] = {}
    if args.temperature is not None:
        ask_options["temperature"] = args.temperature
    if args.no_reasoning:
        ask_options["reasoning"] = False
    sync = (
        None
        if args.no_s3
        else S3Sync(
            batch_s3_uri(args.s3_prefix, output_dir),
            output_dir,
            live_files=(RESULTS_FILE, META_FILE),
        )
    )
    print(
        f"{len(annonce_ids)} annonce(s) — types {', '.join(types)} — "
        f"approche {APPROACHES[args.approche]}, "
        f"prompt {PROMPT_VERSIONS[args.approche]}, "
        f"analyse {ANALYSIS_MODES[args.analyse]}, modèle {get_model_name()}, "
        f"raisonnement {reasoning_label(ask_options)}, "
        f"{args.workers} appel(s) simultané(s)\nrésultats : {output_dir}"
        + (f"\ncopie S3 : {sync.uri}" if sync else "")
    )
    if len(annonce_ids) > CONFIRMATION_THRESHOLD and not args.yes:
        if not _confirm(len(annonce_ids)):
            return 1

    mlflow = None
    if not args.no_mlflow:
        try:
            mlflow = setup_mlflow(args.experiment)
        except MlflowCredentialsError as error:
            print(f"{error}\n(--no-mlflow pour lancer sans run MLflow)", file=sys.stderr)
            return 2

    meta = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "approche": args.approche,
        "prompt_version": PROMPT_VERSIONS[args.approche],
        "analyse": args.analyse,
        "modele": get_model_name(),
        "raisonnement": reasoning_label(ask_options),
        "annotations": str(args.annotations),
        "types": list(types),
        "sample_size": sample_size,
        "per_type": args.per_type,
        "seed": args.seed,
        "workers": args.workers,
        "ask_options": ask_options,
        "s3_uri": sync.uri if sync else None,
        "langfuse_session": output_dir.name if langfuse_enabled() else None,
        "annonces": annonce_ids,
    }
    write_meta(output_dir, meta)
    if sync is not None:
        try:
            sync.check(META_FILE)
        except S3SyncError as error:
            print(f"{error}\n(--no-s3 pour lancer sans copie S3)", file=sys.stderr)
            return 2

    status = "FAILED"
    if mlflow is not None:
        run = mlflow.start_run(run_name=output_dir.name)
        meta["mlflow_run_id"] = run.info.run_id
        write_meta(output_dir, meta)
        print(f"run MLflow : {run.info.run_id} (expérience {args.experiment})")
    try:
        if mlflow is not None:
            mlflow.log_params(_run_params(meta))
            mlflow.set_tag("dossier_local", str(output_dir))
            if sync is not None:
                mlflow.set_tag("s3_uri", sync.uri)
            if meta["langfuse_session"]:
                mlflow.set_tag("langfuse_session", meta["langfuse_session"])
        started = time.monotonic()
        records = run_batch(
            annonce_ids,
            annotations,
            approach=args.approche,
            analysis=args.analyse,
            workers=args.workers,
            output_dir=output_dir,
            on_record=sync.maybe_push if sync else None,
            trace_session=output_dir.name,
            **ask_options,
        )
        meta["duree_totale_s"] = round(time.monotonic() - started, 1)
        meta["annonces_traitees"] = len(records)
        write_meta(output_dir, meta)

        annonces, operations = evaluate(records)
        annonces.write_parquet(output_dir / "annonces.parquet")
        operations.write_parquet(output_dir / "operations.parquet")
        metrics = compute_metrics(annonces, operations)
        metrics_text = format_metrics(metrics, meta)
        (output_dir / METRICS_TEXT_FILE).write_text(metrics_text, encoding="utf-8")
        if sync is not None:
            if sync.push_all():
                print(f"résultats copiés sur {sync.uri}")
            else:
                print(
                    f"copie S3 incomplète ; à refaire à la main :\n"
                    f"  mc cp --recursive {output_dir}/ s3/{sync.bucket}/{sync.prefix}/",
                    file=sys.stderr,
                )
        if mlflow is not None:
            mlflow.log_metrics(mlflow_metrics(metrics, meta, annonces))
            mlflow.log_artifact(str(output_dir / METRICS_TEXT_FILE))
            if sync is not None and sync.failures:
                mlflow.set_tag("s3_envois_en_echec", sync.failures)
        status = "FINISHED" if len(records) == len(annonce_ids) else "KILLED"
    except KeyboardInterrupt:
        status = "KILLED"
        raise
    finally:
        flush_traces()
        if mlflow is not None:
            mlflow.end_run(status=status)
    return records, operations, metrics_text


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    # Une ligne INFO par annonce noierait la progression ; le fichier de log garde tout.
    console_handler.setLevel(logging.WARNING)

    if args.load is not None:
        try:
            meta, records = _load(args.load)
        except S3SyncError as error:
            print(error, file=sys.stderr)
            return 2
        print(f"batch rechargé : {args.load} ({len(records)} annonces)")
        annonces, operations = evaluate(records)
        metrics_text = format_metrics(compute_metrics(annonces, operations), meta)
    else:
        outcome = run_new_batch(args)
        if isinstance(outcome, int):
            return outcome
        records, operations, metrics_text = outcome

    print(metrics_text)
    if args.no_browse:
        return 0
    return browse(records, operations, metrics_text)


if __name__ == "__main__":
    raise SystemExit(main())
