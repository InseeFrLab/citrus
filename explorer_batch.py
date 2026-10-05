"""Batch « bourrin » : lancer l'approche à prompt unique sur des opérations annotées.

Tire un échantillon (ou la totalité) des opérations du fichier d'annotations,
éventuellement filtrées sur le type annoté, envoie chaque annonce à
`bourrin.run_bourrin` avec le prompt de l'approche choisie (`--approche`,
`metier` par défaut, ou `juridique`), puis affiche les métriques principales et ouvre une
session pour examiner les résultats opération par opération.

Chaque résultat est écrit dès qu'il arrive dans
`artifacts/bourrin_batch/<horodatage>/results.jsonl` : un batch interrompu garde
ce qui a été calculé, et `--load` rouvre un batch sans refaire d'appel LLM.

Usage :
    uv run python explorer_batch.py                          # 20 opérations au hasard
    uv run python explorer_batch.py --types VE LG -n 50      # filtrer sur le type annoté
    uv run python explorer_batch.py --types FUSION --per-type -n 5   # 5 par type de fusion
    uv run python explorer_batch.py --types TP --all         # toutes les opérations TP
    uv run python explorer_batch.py --approche juridique     # prompt de lecture juridique
    uv run python explorer_batch.py --analyse courte         # analyse en 2-3 phrases (plus rapide)
    uv run python explorer_batch.py --analyse aucune         # pas d'analyse (le plus rapide)
    uv run python explorer_batch.py --no-reasoning           # sans raisonnement du modèle
    uv run python explorer_batch.py --load artifacts/bourrin_batch/<horodatage>

Les fonctions sont aussi importables depuis une cellule VS Code :
    >>> from explorer_batch import load_batch, evaluate, compute_metrics, format_metrics
    >>> meta, records = load_batch("artifacts/bourrin_batch/<horodatage>")
    >>> annonces, operations = evaluate(records)
    >>> operations.filter(~pl.col("montantNet_ok"))
"""
from __future__ import annotations

import argparse
import json
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

from bourrin import (
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
    run_bourrin,
)
from src import console_handler
from src.bodacc.api import BodaccFetchError
from src.llm.client import get_model_name


FUSION_FAMILY = ("FU", "AB", "SP", "AP", "ST")
TYPE_ALIASES = {"FUSION": FUSION_FAMILY, "FUSIONS": FUSION_FAMILY}

DEFAULT_SAMPLE_SIZE = 20
DEFAULT_SEED = 0
DEFAULT_WORKERS = 4
# Au-delà, le lancement demande confirmation (sauf --yes).
CONFIRMATION_THRESHOLD = 100
# Tolérance relative de l'indicateur « montant proche », en plus du taux exact.
AMOUNT_RELATIVE_TOLERANCE = 0.10

OUTPUT_ROOT = Path(__file__).resolve().parent / "artifacts" / "bourrin_batch"
RESULTS_FILE = "results.jsonl"
META_FILE = "meta.json"

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
    fonctions de comparaison de `bourrin` s'appliquent telles quelles, y compris
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


def run_one(
    annonce_id: str,
    references: list[dict[str, Any]],
    *,
    approach: str = DEFAULT_APPROACH,
    analysis: str = DEFAULT_ANALYSIS,
    fetch: Callable[[str], dict[str, Any]] | None = None,
    ask_fn: Callable[..., str] | None = None,
    **ask_options: Any,
) -> dict[str, Any]:
    """Traiter une annonce ; une erreur est enregistrée au lieu d'arrêter le batch."""

    record: dict[str, Any] = {
        "annonce_id": annonce_id,
        "references": references,
        "statut": "ok",
        "erreur": None,
        "result": None,
    }
    try:
        result = run_bourrin(
            annonce_id,
            approach=approach,
            analysis=analysis,
            fetch=fetch,
            ask_fn=ask_fn,
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
    **ask_options: Any,
) -> list[dict[str, Any]]:
    """Lancer `run_bourrin` sur chaque annonce, en parallèle.

    Chaque résultat est ajouté à `output_dir/results.jsonl` dès réception.
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
    "duree_s": pl.Float64,
}


def evaluate(records: Sequence[dict[str, Any]]) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Deux tables : une ligne par annonce, une ligne par opération.

    Les opérations prédites sont appariées aux lignes
    annotées comme dans la session bourrin : par couple de SIREN, sinon par ordre.
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
        "duree_mediane_s": statistics.median(durations) if durations else None,
        "duree_totale_llm_s": sum(durations),
    }


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
        f"  Batch bourrin — approche {label}"
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
        f"  Alertes Luhn sur un SIREN      {metrics['alertes_luhn']}"
        f"  ({metrics['annonces_avec_alertes']} annonce(s) avec avertissements)",
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
  :texte <n>         l'annonce en clair
  :payload <n>       le JSON BODACC de l'annonce
  :raw <n>           la réponse brute du LLM
  :metriques         réafficher les métriques
  :help              cette aide
  :quit              quitter (ou Ctrl-D)"""


def _record_for(
    entry: str, operations: pl.DataFrame, records: dict[str, dict[str, Any]]
) -> tuple[dict[str, Any] | None, int | None]:
    """Retrouver l'annonce désignée par un numéro d'opération ou un identifiant."""

    entry = entry.strip()
    if entry.isdigit():
        rows = operations.filter(pl.col("n") == int(entry))
        if not rows.height:
            return None, None
        return records.get(rows["annonce_id"][0]), int(entry)
    return records.get(entry), None


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
            record, _ = _record_for(argument, operations, by_id)
            if record is None:
                print(f"  opération ou annonce introuvable : {argument or '(rien)'}")
            elif record["result"] is None:
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
            record, n = _record_for(entry, operations, by_id)
            if record is None:
                print(f"  opération ou annonce introuvable : {entry}")
            else:
                print(format_detail(record, n))


# --------------------------------------------------------------------------
# Ligne de commande
# --------------------------------------------------------------------------


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Lancer l'approche bourrin sur des opérations annotées et l'évaluer"
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
        help="dossier des résultats (défaut : artifacts/bourrin_batch/<horodatage>)",
    )
    parser.add_argument(
        "--load",
        type=Path,
        default=None,
        help="rouvrir un batch enregistré, sans appel LLM",
    )
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


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    # Une ligne INFO par annonce noierait la progression ; le fichier de log garde tout.
    console_handler.setLevel(logging.WARNING)

    if args.load is not None:
        meta, records = load_batch(args.load)
        print(f"batch rechargé : {args.load} ({len(records)} annonces)")
    else:
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
        print(
            f"{len(annonce_ids)} annonce(s) — types {', '.join(types)} — "
            f"approche {APPROACHES[args.approche]}, "
            f"prompt {PROMPT_VERSIONS[args.approche]}, "
            f"analyse {ANALYSIS_MODES[args.analyse]}, modèle {get_model_name()}, "
            f"raisonnement {reasoning_label(ask_options)}, "
            f"{args.workers} appel(s) simultané(s)\nrésultats : {output_dir}"
        )
        if len(annonce_ids) > CONFIRMATION_THRESHOLD and not args.yes:
            if not _confirm(len(annonce_ids)):
                return 1

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
            "annonces": annonce_ids,
        }
        write_meta(output_dir, meta)
        started = time.monotonic()
        records = run_batch(
            annonce_ids,
            annotations,
            approach=args.approche,
            analysis=args.analyse,
            workers=args.workers,
            output_dir=output_dir,
            **ask_options,
        )
        meta["duree_totale_s"] = round(time.monotonic() - started, 1)
        meta["annonces_traitees"] = len(records)
        write_meta(output_dir, meta)

    annonces, operations = evaluate(records)
    if args.load is None:
        annonces.write_parquet(output_dir / "annonces.parquet")
        operations.write_parquet(output_dir / "operations.parquet")
    metrics_text = format_metrics(compute_metrics(annonces, operations), meta)
    print(metrics_text)
    if args.no_browse:
        return 0
    return browse(records, operations, metrics_text)


if __name__ == "__main__":
    raise SystemExit(main())
