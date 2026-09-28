"""Approche « bourrin » : un seul LLM, un seul prompt, routage + extraction.

Contrairement au pipeline `src/routing` + `src/operation`, ce script ne découpe
pas le problème : il envoie l'annonce BODACC brute à un LLM avec un prompt
générique, et lui demande à la fois le type d'opération et tous les
champs métier.

Deux approches, chacune avec son propre prompt et un appel LLM séparé :
  - `metier`    : application stricte des règles des gestionnaires
                  (`bourrin_prompt_metier.md`) ; ce sont ces règles qui ont
                  produit les annotations ;
  - `juridique` : lecture fondée sur la nature juridique des opérations
                  (`bourrin_prompt_juridique.md`), pour mesurer où les règles
                  s'en écartent.

Les deux renvoient la même structure : un type et une **liste** d'opérations
(une scission à plusieurs bénéficiaires en produit plusieurs, comme dans le
fichier d'annotations). Le champ libre `analyse` est paramétrable (`--analyse`) :
complète, courte ou absente — c'est l'essentiel du texte généré, donc du temps.

Usage :
    uv run python bourrin.py                    # boucle interactive
    uv run python bourrin.py A20230147853       # one-shot
    uv run python bourrin.py A20230147853 --json
    uv run python bourrin.py A20230147853 --approche juridique
    uv run python bourrin.py A20230147853 --analyse courte   # analyse en 2-3 phrases
    uv run python bourrin.py A20230147853 --no-reasoning   # sans raisonnement du modèle
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import textwrap
import time
from collections.abc import Callable, Sequence
from datetime import date, datetime
from pathlib import Path
from typing import Any

import polars as pl

# Confort REPL : permet de lancer les cellules depuis le dossier parent.
if os.path.isdir("citrus-ia-gen"):
    os.chdir("citrus-ia-gen")

from src import logger
from src.bodacc.api import BodaccFetchError, bodacc_api
from src.bodacc import normalize_bodacc_announcement
from src.llm.client import ask, get_model_name, parse_json_answer
from src.operation.vente import _eur_to_integer_keur
from src.utils import annuaire, is_luhn_valid


# Approche → libellé ; chacune a son prompt `bourrin_prompt_<approche>.md`.
APPROACHES = {
    "metier": "règles métier",
    "juridique": "lecture juridique",
}
DEFAULT_APPROACH = "metier"
PROMPT_VERSIONS = {
    "metier": "bourrin-metier-v2",
    "juridique": "bourrin-juridique-v2",
}

# Mode du champ `analyse` → libellé. Il change le prompt système (format de sortie).
ANALYSIS_MODES = {
    "complete": "complète",
    "courte": "courte (2-3 phrases)",
    "aucune": "sans analyse",
}
DEFAULT_ANALYSIS = "complete"

ANNOTATIONS_PATH = "s3://projet-citrus/data/operations_verifiees.parquet"
DEFAULT_S3_ENDPOINT = "https://minio.lab.sspcloud.fr"
ANNOTATION_ID_COLUMN = "ref_annonce_complet"

OPERATION_CODES = ("VE", "FU", "AB", "TP", "SP", "AP", "ST", "LG")

BUSINESS_FIELDS = (
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

COMPARED_FIELDS = (
    "typeOperation",
    "sirenCedant",
    "sirenBeneficiaire",
    "dateEffetComptable",
    "dateRealisationJuridique",
    "montantNet",
)

_DATE_INPUT_FORMATS = (
    "%Y-%m-%d",
    "%d-%m-%Y",
    "%d/%m/%Y",
    "%d.%m.%Y",
    "%Y/%m/%d",
)

# Gabarits de prompt système versionnés à part, à la racine du repo, pour rester
# lisibles. Les retours à la ligne des fichiers sont envoyés tels quels au LLM.
# Le corps des gabarits ne cite jamais `analyse` : tout ce qui le concerne passe
# par les emplacements {{ANALYSE_JSON}} (exemple JSON) et {{ANALYSE_REGLES}}
# (règles de sortie), remplis selon le mode.
PROMPT_DIR = Path(__file__).resolve().parent
_PROMPT_TEMPLATES = {
    approach: (PROMPT_DIR / f"bourrin_prompt_{approach}.md")
    .read_text(encoding="utf-8")
    .strip()
    for approach in APPROACHES
}

_ANALYSIS_JSON = ',\n  "analyse": "..."'

# Ce que les deux approches demandent d'expliquer en plus de leur propre démarche.
_COMPLETE_EXTRAS = """\
  Signale en particulier :
  le motif du rejet quand `retenu` vaut false,
  un cas presque réuni auquel il manque un élément, et le choix de "UNKNOWN" qui en découle,
  un montant de secours (par exemple la prime de fusion faute de mieux),
  et chaque champ laissé à null."""

ANALYSIS_RULES: dict[str, dict[str, str]] = {
    "complete": {
        "metier": f"""\
- `analyse` est rédigée en français et explique, étape par étape :
  le flux identifié (RCS-A ou RCS-B),
  pourquoi ce type a été retenu et quels cas ont été écartés,
  comment chaque champ a été extrait (quelle règle de priorité a joué, quelle ancre textuelle),
  et toute incertitude.
{_COMPLETE_EXTRAS}""",
        "juridique": f"""\
- `analyse` est rédigée en français et explique, étape par étape :
  le flux identifié (RCS-A ou RCS-B),
  quels éléments du texte établissent la nature juridique de l'opération
  et pourquoi les types voisins (§3) ont été écartés,
  comment chaque champ a été extrait (quelle ancre textuelle),
  et toute incertitude.
{_COMPLETE_EXTRAS}""",
    },
    "courte": dict.fromkeys(
        APPROACHES,
        """\
- `analyse` tient en 2 à 3 phrases au maximum, en français :
  le type retenu et l'indice décisif du texte,
  puis la principale incertitude s'il y en a une
  (motif du rejet, champ manquant, montant de secours).
  Pas de récapitulatif champ par champ.""",
    ),
    "aucune": dict.fromkeys(
        APPROACHES,
        """\
- Ne produis aucun champ `analyse` ni aucun texte d'explication :
  uniquement les champs de l'exemple ci-dessus.""",
    ),
}


def _check_approach(approach: str) -> str:
    if approach not in APPROACHES:
        raise ValueError(
            f"approche inconnue : {approach!r} (attendues : {', '.join(APPROACHES)})"
        )
    return approach


def _check_analysis(analysis: str) -> str:
    if analysis not in ANALYSIS_MODES:
        raise ValueError(
            f"mode d'analyse inconnu : {analysis!r} "
            f"(attendus : {', '.join(ANALYSIS_MODES)})"
        )
    return analysis


def system_prompt(
    approach: str = DEFAULT_APPROACH, analysis: str = DEFAULT_ANALYSIS
) -> str:
    """Prompt système d'une approche, avec les consignes du mode d'analyse."""

    prompt = (
        _PROMPT_TEMPLATES[_check_approach(approach)]
        .replace(
            "{{ANALYSE_JSON}}",
            "" if _check_analysis(analysis) == "aucune" else _ANALYSIS_JSON,
        )
        .replace("{{ANALYSE_REGLES}}", ANALYSIS_RULES[analysis][approach])
    )
    if "{{" in prompt:
        raise ValueError(f"emplacement non rempli dans le prompt {approach!r}")
    return prompt


def build_bourrin_messages(
    payload: dict[str, Any],
    approach: str = DEFAULT_APPROACH,
    analysis: str = DEFAULT_ANALYSIS,
) -> list[dict[str, str]]:
    """Construire l'unique paire de messages envoyée au LLM pour cette approche."""

    compact_payload = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, default=str
    )
    user = (
        "Annonce BODACC à classer et à extraire :\n"
        f"{compact_payload}"
    )
    return [
        {"role": "system", "content": system_prompt(approach, analysis)},
        {"role": "user", "content": user},
    ]


def _expand_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Déplier les balises BODACC stockées sous forme de chaîne JSON.

    Remplace `src.bodacc.api._clean_json`, qui suppose la présence des balises
    de vente et lève une TypeError sur les sept autres types d'opération.
    """

    expanded = dict(payload)
    for key, value in payload.items():
        if isinstance(value, str) and value.lstrip()[:1] in "{[":
            try:
                expanded[key] = json.loads(value)
            except json.JSONDecodeError:
                pass
    return expanded


def _normalized_siren(value: Any, warnings: list[str], field: str) -> str | None:
    if value is None:
        return None
    digits = str(value).replace(" ", "").replace(".", "").strip()
    if not digits or digits.lower() in {"null", "none"}:
        return None
    if not digits.isdigit() or len(digits) > 9:
        warnings.append(f"{field} n'est pas un SIREN exploitable : {value!r}")
        return None
    siren = digits.zfill(9)
    if not is_luhn_valid(siren):
        warnings.append(f"{field}={siren} échoue le contrôle de Luhn")
    return siren


def _normalized_date(value: Any, warnings: list[str], field: str) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"null", "none"}:
        return None
    for date_format in _DATE_INPUT_FORMATS:
        try:
            return datetime.strptime(text, date_format).date().isoformat()
        except ValueError:
            continue
    warnings.append(f"{field} n'est pas une date reconnue : {value!r}")
    return None


def _normalized_amount(value: Any, warnings: list[str], field: str) -> int | None:
    """Convertir le montant en euros renvoyé par le LLM vers le contrat kEUR."""

    if value is None:
        return None
    text = str(value).replace(" ", "").replace(",", ".").strip()
    if not text or text.lower() in {"null", "none"}:
        return None
    try:
        return _eur_to_integer_keur(float(text))
    except (ArithmeticError, TypeError, ValueError):
        warnings.append(f"{field} inexploitable : {value!r}")
        return None


def _normalized_type(value: Any, warnings: list[str], context: str) -> str | None:
    if value is None:
        return None
    code = str(value).strip().upper()
    if code in {"", "NULL", "NONE"}:
        return None
    if code not in (*OPERATION_CODES, "UNKNOWN"):
        warnings.append(f"{context} : type hors taxonomie ({value!r})")
        return "UNKNOWN"
    return code


def _campaign_year(payload: dict[str, Any]) -> int | None:
    """Année de campagne : 4 premiers chiffres de `parution`, sinon parution."""

    parution = str(payload.get("parution") or "").strip()
    if parution[:4].isdigit():
        return int(parution[:4])
    published = _normalized_date(payload.get("dateparution"), [], "dateparution")
    return int(published[:4]) if published is not None else None


def _normalized_operation(
    raw: dict[str, Any],
    payload: dict[str, Any],
    default_type: str | None,
    warnings: list[str],
    prefix: str,
) -> dict[str, Any]:
    """Ramener une opération libre du LLM au contrat OperationResult."""

    operation_type = (
        _normalized_type(raw.get("codeTypeOperation"), warnings, prefix)
        or default_type
        or "UNKNOWN"
    )
    return {
        "anneeCampagne": _campaign_year(payload),
        "typeOperation": operation_type,
        "sirenCedant": _normalized_siren(
            raw.get("sirenCedant"), warnings, f"{prefix} sirenCedant"
        ),
        "raisonSocialeCedant": raw.get("raisonSocialeCedant") or None,
        "sirenBeneficiaire": _normalized_siren(
            raw.get("sirenBeneficiaire"), warnings, f"{prefix} sirenBeneficiaire"
        ),
        "raisonSocialeBeneficiaire": raw.get("raisonSocialeBeneficiaire") or None,
        "dateEffetComptable": _normalized_date(
            raw.get("dateEffetComptable"), warnings, f"{prefix} dateEffetComptable"
        ),
        "dateRealisationJuridique": _normalized_date(
            raw.get("dateRealisationJuridique"),
            warnings,
            f"{prefix} dateRealisationJuridique",
        ),
        "montantNet": _normalized_amount(
            raw.get("montantNetEuros"), warnings, f"{prefix} montantNetEuros"
        ),
        "source": payload.get("url_complete"),
    }


def _normalized_reading(
    raw: Any, payload: dict[str, Any], warnings: list[str], label: str
) -> dict[str, Any]:
    """Normaliser une lecture : son code de type et sa liste d'opérations."""

    if not isinstance(raw, dict):
        if raw is not None:
            warnings.append(f"lecture « {label} » malformée : {type(raw).__name__}")
        raw = {}
    code = _normalized_type(raw.get("codeTypeOperation"), warnings, label)

    operations = raw.get("operations")
    if operations is None:
        operations = []
    elif not isinstance(operations, list):
        warnings.append(f"{label} : « operations » n'est pas une liste")
        operations = []

    normalized = [
        _normalized_operation(
            operation if isinstance(operation, dict) else {},
            payload,
            code,
            warnings,
            f"{label} op.{index}",
        )
        for index, operation in enumerate(operations, 1)
    ]
    return {"codeTypeOperation": code, "operations": normalized}


def normalize_answer(
    answer: dict[str, Any],
    payload: dict[str, Any],
    approach: str = DEFAULT_APPROACH,
) -> tuple[dict[str, Any], list[str]]:
    """Ramener la réponse libre du LLM à l'enveloppe normalisée, à une lecture."""

    warnings: list[str] = []
    if not answer:
        warnings.append("réponse LLM vide ou non analysable en JSON")
    reading = _normalized_reading(answer, payload, warnings, APPROACHES[approach])
    envelope: dict[str, Any] = {
        "id": answer.get("id") or payload.get("id"),
        "approche": approach,
        **reading,
        "analyse": answer.get("analyse") or None,
    }

    retenu = answer.get("retenu")
    has_operations = bool(envelope["operations"])
    if not isinstance(retenu, bool):
        if answer:
            warnings.append("champ « retenu » absent : déduit de la présence d'opérations")
        retenu = has_operations
    elif retenu and not has_operations:
        warnings.append("annonce déclarée retenue mais aucune opération produite")
    elif not retenu and has_operations:
        warnings.append("annonce déclarée non retenue mais des opérations sont produites")
    envelope["retenu"] = retenu
    return envelope, warnings


def _storage_options() -> dict[str, str] | None:
    """Identifiants S3 : variables d'environnement si présentes, sinon profil AWS.

    Le datalab SSP Cloud expose les clés en variables d'environnement ; le reste
    du dépôt suppose un profil AWS nommé ``service-account``. On accepte les deux.
    """

    endpoint = os.environ.get("AWS_ENDPOINT_URL")
    if endpoint is None and os.environ.get("AWS_S3_ENDPOINT"):
        endpoint = f"https://{os.environ['AWS_S3_ENDPOINT']}"
    options = {
        "aws_endpoint_url": endpoint or DEFAULT_S3_ENDPOINT,
        "aws_region": os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
    }
    if not os.environ.get("AWS_ACCESS_KEY_ID"):
        return None
    options["aws_access_key_id"] = os.environ["AWS_ACCESS_KEY_ID"]
    options["aws_secret_access_key"] = os.environ["AWS_SECRET_ACCESS_KEY"]
    if os.environ.get("AWS_SESSION_TOKEN"):
        options["aws_session_token"] = os.environ["AWS_SESSION_TOKEN"]
    return options


def load_annotations(source: str = ANNOTATIONS_PATH) -> pl.DataFrame:
    """Charger le fichier d'opérations vérifiées (S3 ou chemin local).

    Utilisable directement en REPL :
        >>> from bourrin import load_annotations, describe_annotations
        >>> df = load_annotations()
        >>> print(describe_annotations(df))
    """

    if not str(source).startswith("s3://"):
        return pl.read_parquet(source)
    options = _storage_options()
    if options is not None:
        return pl.read_parquet(source, storage_options=options)
    return pl.read_parquet(
        source,
        storage_options={
            "aws_endpoint_url": DEFAULT_S3_ENDPOINT,
            "aws_region": "us-east-1",
        },
        credential_provider=pl.CredentialProviderAWS(
            profile_name="service-account", region_name="us-east-1"
        ),
    )


def describe_annotations(annotations: pl.DataFrame) -> str:
    """Résumé lisible du contenu du fichier d'annotations."""

    counts = annotations.group_by("type_op").len().sort("len", descending=True)
    repartition = "  ".join(
        f"{row['type_op']}={row['len']}" for row in counts.iter_rows(named=True)
    )
    nulls = dict(zip(annotations.columns, annotations.null_count().row(0)))
    manquants = "  ".join(f"{col}={n}" for col, n in nulls.items() if n)
    duplicates = int(annotations[ANNOTATION_ID_COLUMN].is_duplicated().sum())
    return "\n".join(
        [
            f"  lignes                     {annotations.height}"
            f"  |  annonces uniques {annotations[ANNOTATION_ID_COLUMN].n_unique()}"
            + (f"  |  ⚠ {duplicates} lignes partagent une annonce" if duplicates else ""),
            f"  colonnes                   {', '.join(annotations.columns)}",
            f"  répartition type_op        {repartition}",
            f"  valeurs manquantes         {manquants or 'aucune'}",
        ]
    )


def annotation_for(
    annotations: pl.DataFrame | None, annonce_id: str
) -> list[dict[str, Any]]:
    """Toutes les lignes annotées d'une annonce — une annonce peut en porter plusieurs."""

    if annotations is None:
        return []
    rows = annotations.filter(pl.col(ANNOTATION_ID_COLUMN) == annonce_id)
    return rows.rows(named=True)


_ANNOTATION_COLUMNS = {
    "typeOperation": "type_op",
    "sirenCedant": "siren_cedante",
    "sirenBeneficiaire": "siren_beneficiaire",
    "dateEffetComptable": "date_effet_comptable_op",
    "dateRealisationJuridique": "date_realisation_juridique_op",
    "montantNet": "montant",
}


def _reference_value(field: str, annotation: dict[str, Any]) -> Any:
    value = annotation.get(_ANNOTATION_COLUMNS[field])
    if value is None:
        return None
    if field.startswith("siren"):
        return str(value).zfill(9)
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if field == "montantNet":
        return round(float(value))
    return value


def _pair_operations(
    operations: Sequence[dict[str, Any]], annotations: Sequence[dict[str, Any]]
) -> list[tuple[dict[str, Any] | None, dict[str, Any] | None]]:
    """Apparier prédictions et annotations : par couple de SIREN, sinon par ordre."""

    remaining = list(annotations)
    pairs: list[list[Any]] = []
    for operation in operations:
        key = (operation.get("sirenCedant"), operation.get("sirenBeneficiaire"))
        matched = None
        if key != (None, None):
            for index, annotation in enumerate(remaining):
                reference = (
                    _reference_value("sirenCedant", annotation),
                    _reference_value("sirenBeneficiaire", annotation),
                )
                if reference == key:
                    matched = remaining.pop(index)
                    break
        pairs.append([operation, matched])
    for pair in pairs:
        if pair[1] is None and remaining:
            pair[1] = remaining.pop(0)
    pairs.extend([None, annotation] for annotation in remaining)
    return [(pair[0], pair[1]) for pair in pairs]


def format_comparison(
    operations: Sequence[dict[str, Any]] | dict[str, Any],
    annotations: Sequence[dict[str, Any]] | dict[str, Any] | None,
) -> str:
    """Confronter les opérations prédites aux lignes annotées, champ par champ."""

    if isinstance(operations, dict):
        operations = [operations]
    if annotations is None:
        annotations = []
    elif isinstance(annotations, dict):
        annotations = [annotations]

    lines = ["  — référence annotée —"]
    if len(operations) != len(annotations):
        lines.append(
            f"  ⚠ {len(operations)} opération(s) prédite(s) pour "
            f"{len(annotations)} ligne(s) annotée(s)"
        )
    pairs = _pair_operations(list(operations), list(annotations))
    for index, (operation, annotation) in enumerate(pairs, 1):
        if len(pairs) > 1:
            lines.append(f"  · opération {index}/{len(pairs)}")
        if annotation is None:
            lines.append("    ✗ aucune ligne annotée correspondante")
            continue
        if operation is None:
            reference_type = _reference_value("typeOperation", annotation)
            reference_siren = _reference_value("sirenBeneficiaire", annotation)
            lines.append(
                f"    ✗ ligne annotée sans prédiction "
                f"({reference_type}, bénéficiaire {reference_siren})"
            )
            continue
        for field in COMPARED_FIELDS:
            expected = _reference_value(field, annotation)
            obtained = operation.get(field)
            mark = "✓" if expected == obtained else "✗"
            rendered = "—" if expected is None else str(expected)
            if expected != obtained:
                rendered += f"   (prédit : {obtained if obtained is not None else '—'})"
            lines.append(f"  {mark} {field:<24} {rendered}")
    return "\n".join(lines)


def run_bourrin(
    annonce_id: str,
    *,
    approach: str = DEFAULT_APPROACH,
    analysis: str = DEFAULT_ANALYSIS,
    fetch: Callable[[str], dict[str, Any]] | None = None,
    ask_fn: Callable[..., str] | None = None,
    **ask_options: Any,
) -> dict[str, Any]:
    """Enchaîner fetch BODACC → appel LLM unique (prompt de l'approche) → normalisation."""

    _check_approach(approach)
    _check_analysis(analysis)
    fetch = fetch or bodacc_api().fetch_annonce_json
    ask_fn = ask_fn or ask

    payload = _expand_payload(dict(fetch(annonce_id)))
    messages = build_bourrin_messages(payload, approach, analysis)
    started = time.monotonic()
    raw_answer = ask_fn(messages, **ask_options)
    elapsed = time.monotonic() - started
    answer = parse_json_answer(raw_answer)
    if not answer:
        logger.warning("Réponse LLM non exploitable pour %s", annonce_id)
    envelope, warnings = normalize_answer(answer, payload, approach)
    operations = envelope["operations"]
    return {
        "annonce_id": annonce_id,
        "approche": approach,
        "prompt_version": PROMPT_VERSIONS[approach],
        "analyse_mode": analysis,
        "envelope": envelope,
        # Raccourci de confort : première opération prédite.
        "prediction": operations[0] if operations else None,
        "warnings": warnings,
        "elapsed_seconds": round(elapsed, 2),
        "payload": payload,
        "messages": messages,
        "raw_answer": raw_answer,
    }


def _format_operation(operation: dict[str, Any], indent: str = "    ") -> list[str]:
    lines = []
    for field in BUSINESS_FIELDS:
        value = operation.get(field)
        rendered = "—" if value is None else str(value)
        if field.startswith("siren") and value is not None:
            rendered = f"{value}  {annuaire(value)}"
        lines.append(f"{indent}{field:<26} {rendered}")
    return lines


def format_result(result: dict[str, Any]) -> str:
    """Rendu lisible d'un résultat pour la session interactive."""

    envelope = result["envelope"]
    label = APPROACHES.get(envelope.get("approche"), "?")
    operations = envelope["operations"]
    code = envelope["codeTypeOperation"] or "—"
    lines = [
        f"  {'retenu':<26} {'oui' if envelope['retenu'] else 'non'}",
        f"  {label:<26} {code:<9} {len(operations)} opération(s)",
    ]
    for index, operation in enumerate(operations, 1):
        lines.append(f"  · opération {index}/{len(operations)}")
        lines.extend(_format_operation(operation))

    if envelope.get("analyse"):
        lines.append("  · analyse")
        lines.extend(
            f"    {chunk}" for chunk in textwrap.wrap(envelope["analyse"], width=88)
        )
    for warning in result["warnings"]:
        lines.append(f"  ⚠  {warning}")
    lines.append(f"  ({result['elapsed_seconds']} s, modèle {get_model_name()})")
    return "\n".join(lines)


def format_annonce(payload: dict[str, Any], largeur: int = 90) -> str:
    """Rendu lisible d'une annonce BODACC : entête, descriptifs, origine du fonds."""

    normalized = normalize_bodacc_announcement(payload)
    lines = [
        f"{payload.get('id', '?')}  |  {normalized.dialect.value}"
        f"  |  parution {normalized.publication_date}"
        f"  |  {payload.get('familleavis_lib', '')}",
        str(payload.get("url_complete", "")),
    ]
    if normalized.main_name or normalized.main_siren:
        lines.append(
            f"\nSociété objet : {normalized.main_name} ({normalized.main_siren})"
        )
    for party, label in (
        (normalized.previous_owners, "Précédent propriétaire"),
        (normalized.previous_operators, "Précédent exploitant"),
    ):
        for entry in party:
            lines.append(f"{label} : {entry.name} ({entry.siren})")
    for index, description in enumerate(normalized.all_descriptions, 1):
        lines.append(f"\n--- descriptif {index} ---")
        lines.append(textwrap.fill(description, width=largeur))
    for origin in normalized.origin_funds:
        lines.append("\n--- origine du fonds ---")
        lines.append(textwrap.fill(origin, width=largeur))
    return "\n".join(lines)


def afficher_annonce(annonce_id: str, brut: bool = False) -> dict[str, Any]:
    """Afficher une annonce BODACC à partir de sa référence, et renvoyer le payload."""

    payload = _expand_payload(bodacc_api().fetch_annonce_json(annonce_id))
    if brut:
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    else:
        print(format_annonce(payload))
    return payload


HELP = """Commandes :
  <id>          classer et extraire une annonce (ex. A20230147853)
  :sample       tirer une annonce au hasard dans les annotations
  :sample VE    tirer une annonce annotée de ce type (VE, LG, TP, FU, AB, SP, ST, AP)
  :annotations  rappeler le contenu du fichier d'annotations
  :texte        afficher l'annonce en clair (entête, descriptifs)
  :payload      afficher le JSON BODACC de la dernière annonce
  :prompt       afficher les messages envoyés au LLM
  :raw          afficher la réponse brute du LLM
  :json         afficher la dernière réponse normalisée en JSON
  :help         afficher cette aide
  :quit         quitter (ou Ctrl-D)"""


def reasoning_label(ask_options: dict[str, Any]) -> str:
    """Libellé du mode de raisonnement, pour les en-têtes et les métadonnées."""

    return "désactivé" if ask_options.get("reasoning") is False else "activé"


def run_interactive(
    annotations: pl.DataFrame | None = None,
    approach: str = DEFAULT_APPROACH,
    analysis: str = DEFAULT_ANALYSIS,
    **ask_options: Any,
) -> int:
    """Boucle interactive : un identifiant d'annonce par ligne."""

    print(
        f"citrus bourrin — approche {APPROACHES[approach]}, "
        f"prompt {PROMPT_VERSIONS[approach]}, analyse {ANALYSIS_MODES[analysis]}, "
        f"modèle {get_model_name()}, "
        f"raisonnement {reasoning_label(ask_options)}"
    )
    if annotations is not None:
        print(f"annotations : {ANNOTATIONS_PATH}")
        print(describe_annotations(annotations))
    print("\nTape :help pour les commandes, :quit pour sortir.\n")
    last: dict[str, Any] | None = None
    while True:
        try:
            entry = input("annonce> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not entry:
            continue
        if entry in {":quit", ":q"}:
            return 0
        if entry == ":help":
            print(HELP)
            continue
        if entry in {":payload", ":prompt", ":raw", ":json", ":texte"}:
            if last is None:
                print("  (aucune annonce traitée pour l'instant)")
                continue
            if entry == ":payload":
                print(json.dumps(last["payload"], ensure_ascii=False, indent=2, default=str))
            elif entry == ":texte":
                print(format_annonce(last["payload"]))
            elif entry == ":prompt":
                for message in last["messages"]:
                    print(f"--- {message['role']} ---\n{message['content']}")
            elif entry == ":raw":
                print(last["raw_answer"])
            else:
                print(json.dumps(last["envelope"], ensure_ascii=False, indent=2))
            continue
        if entry == ":annotations":
            if annotations is None:
                print("  (annotations non chargées)")
            else:
                print(describe_annotations(annotations))
            continue
        if entry.startswith(":sample"):
            if annotations is None:
                print("  (annotations non chargées)")
                continue
            requested = entry.removeprefix(":sample").strip().upper()
            pool = (
                annotations.filter(pl.col("type_op") == requested)
                if requested
                else annotations
            )
            if not pool.height:
                print(f"  aucune annonce annotée de type {requested}")
                continue
            drawn = pool.row(random.randrange(pool.height), named=True)
            print(
                f"  {drawn[ANNOTATION_ID_COLUMN]}   (annoté {drawn['type_op']})"
                "  — colle cet identifiant pour le traiter"
            )
            continue
        if entry.startswith(":"):
            print(f"  commande inconnue : {entry}")
            continue
        try:
            last = run_bourrin(
                entry, approach=approach, analysis=analysis, **ask_options
            )
        except BodaccFetchError as error:
            print(f"  ✗ BODACC [{error.code}] {error.detail}")
            continue
        except Exception as error:  # noqa: BLE001 - session interactive
            print(f"  ✗ {type(error).__name__}: {error}")
            continue
        print(format_result(last))
        referenced = annotation_for(annotations, entry)
        if referenced:
            print(
                format_comparison(
                    last["envelope"]["operations"], referenced
                )
            )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Routage et extraction BODACC par un unique prompt LLM générique"
        )
    )
    parser.add_argument(
        "annonce_ids",
        nargs="*",
        help="identifiants BODACC à traiter ; aucun = session interactive",
    )
    parser.add_argument(
        "--approche",
        choices=tuple(APPROACHES),
        default=DEFAULT_APPROACH,
        help=f"prompt à utiliser (défaut : {DEFAULT_APPROACH})",
    )
    parser.add_argument(
        "--analyse",
        choices=tuple(ANALYSIS_MODES),
        default=DEFAULT_ANALYSIS,
        help=(
            "champ d'explication demandé au modèle : complete, courte (2-3 phrases) "
            f"ou aucune — moins de texte, réponse plus rapide (défaut : {DEFAULT_ANALYSIS})"
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="n'afficher que la réponse normalisée en JSON (mode one-shot)",
    )
    parser.add_argument(
        "--no-reasoning",
        action="store_true",
        help="désactiver le raisonnement du modèle (plus rapide, réponse directe)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="température transmise au LLM",
    )
    parser.add_argument(
        "--annotations",
        default=ANNOTATIONS_PATH,
        help=f"fichier d'opérations vérifiées (défaut : {ANNOTATIONS_PATH})",
    )
    parser.add_argument(
        "--no-annotations",
        action="store_true",
        help="ne pas charger les annotations (démarrage plus rapide, hors ligne)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    ask_options: dict[str, Any] = {}
    if args.temperature is not None:
        ask_options["temperature"] = args.temperature
    if args.no_reasoning:
        ask_options["reasoning"] = False

    annotations: pl.DataFrame | None = None
    if not args.no_annotations and not args.json:
        try:
            annotations = load_annotations(args.annotations)
        except Exception as error:  # noqa: BLE001 - l'outil reste utilisable sans
            print(
                f"⚠ annotations non chargées ({type(error).__name__}: {error})",
                file=sys.stderr,
            )

    if not args.annonce_ids:
        return run_interactive(
            annotations, args.approche, args.analyse, **ask_options
        )

    status = 0
    for annonce_id in args.annonce_ids:
        try:
            result = run_bourrin(
                annonce_id,
                approach=args.approche,
                analysis=args.analyse,
                **ask_options,
            )
        except BodaccFetchError as error:
            print(f"✗ {annonce_id} : BODACC [{error.code}] {error.detail}", file=sys.stderr)
            status = 1
            continue
        if args.json:
            print(json.dumps(result["envelope"], ensure_ascii=False))
            continue
        print(f"{annonce_id}")
        print(format_result(result))
        referenced = annotation_for(annotations, annonce_id)
        if referenced:
            print(
                format_comparison(
                    result["envelope"]["operations"], referenced
                )
            )
    return status


if __name__ == "__main__":
    raise SystemExit(main())
