"""Grille de batchs bourrin : toutes les configurations, suivies dans MLflow.

Lance `explorer_batch.main` une fois par combinaison de
    modèle × raisonnement (oui/non) × analyse (complete/courte/aucune) × approche (metier/juridique)
sur le même échantillon d'annonces annotées (mêmes --types, -n, --seed), puis logge
chaque batch comme un run MLflow, rangé sous un run parent qui représente la grille.

Chaque batch est écrit dans `artifacts/bourrin_grid/<horodatage>/<configuration>/`,
avec les fichiers habituels de `explorer_batch` (results.jsonl, meta.json, parquets),
plus `metriques.txt` et `durees_par_type.csv`, tous envoyés en artefacts MLflow.

MLflow : les credentials doivent être dans l'environnement (injectés par le service
MLflow du SSP Cloud) : `MLFLOW_TRACKING_URI`, et pour un serveur http(s)
`MLFLOW_TRACKING_USERNAME` + `MLFLOW_TRACKING_PASSWORD` ou `MLFLOW_TRACKING_TOKEN`.
Sans eux, ou si le serveur refuse la connexion, le script s'arrête avant tout appel LLM.

Usage :
    uv run python explorer_grid.py --dry-run                     # lister la grille, sans appel
    uv run python explorer_grid.py -n 20                         # les 36 configurations
    uv run python explorer_grid.py -n 20 --types VE LG --modeles gemma4-26b-moe --raisonnement non
    uv run python explorer_grid.py -n 20 --resume artifacts/bourrin_grid/<horodatage>
        (mêmes options d'échantillon qu'au lancement : c'est vérifié)
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

import explorer_batch
from bourrin import (
    ANALYSIS_MODES,
    ANNOTATIONS_PATH,
    APPROACHES,
    COMPARED_FIELDS,
    PROMPT_VERSIONS,
    load_annotations,
)
from explorer_batch import (
    DEFAULT_SAMPLE_SIZE,
    DEFAULT_SEED,
    DEFAULT_WORKERS,
    META_FILE,
    compute_metrics,
    evaluate,
    format_metrics,
    load_batch,
    parse_types,
    select_annonces,
)

MODELS = ("qwen3-8-27b", "qwen3-6-35b-moe", "gemma4-26b-moe")
REASONING = {"oui": True, "non": False}
DEFAULT_EXPERIMENT = "citrus-bourrin-grille"

ROOT = Path(__file__).resolve().parent
GRID_ROOT = ROOT / "artifacts" / "bourrin_grid"
MODEL_ENV = "LLM_MODEL_NAME"

# Durée LLM approximative par annonce, pour l'ordre de grandeur du --dry-run
# (mesurée sur qwen3-8-27b ; qwen3-6 et gemma non mesurés).
ESTIMATED_SECONDS = {True: 20.0, False: 5.0}
# Au-delà de ce nombre d'appels LLM, le lancement demande confirmation (sauf --yes).
CONFIRMATION_THRESHOLD = 200

METRICS_TEXT_FILE = "metriques.txt"
DURATIONS_FILE = "durees_par_type.csv"


# --------------------------------------------------------------------------
# Grille
# --------------------------------------------------------------------------


def configurations(
    models: Sequence[str] = MODELS,
    reasonings: Sequence[bool] = tuple(REASONING.values()),
    analyses: Sequence[str] = tuple(ANALYSIS_MODES),
    approaches: Sequence[str] = tuple(APPROACHES),
) -> list[dict[str, Any]]:
    """Produit cartésien des réglages, un dict par configuration."""

    grid = []
    for model, reasoning, analysis, approach in itertools.product(
        models, reasonings, analyses, approaches
    ):
        grid.append(
            {
                "name": "__".join(
                    (
                        model,
                        "raisonnement" if reasoning else "sans-raisonnement",
                        analysis,
                        approach,
                    )
                ),
                "model": model,
                "reasoning": reasoning,
                "analysis": analysis,
                "approach": approach,
            }
        )
    return grid


def selection_argv(args: argparse.Namespace) -> list[str]:
    """Options de sélection communes à tous les batchs de la grille."""

    argv = ["--seed", str(args.seed), "--workers", str(args.workers)]
    argv += ["--annotations", str(args.annotations)]
    if args.types:
        argv += ["--types", *args.types]
    if args.all:
        argv.append("--all")
    else:
        argv += ["-n", str(args.sample_size)]
    if args.per_type:
        argv.append("--per-type")
    if args.temperature is not None:
        argv += ["--temperature", str(args.temperature)]
    return argv


def batch_argv(config: dict[str, Any], selection: Sequence[str], output_dir: Path) -> list[str]:
    """Arguments de `explorer_batch.main` pour une configuration."""

    argv = ["--approche", config["approach"], "--analyse", config["analysis"]]
    if not config["reasoning"]:
        argv.append("--no-reasoning")
    return [*argv, *selection, "--output-dir", str(output_dir), "--no-browse", "--yes"]


def is_complete(output_dir: Path) -> bool:
    """Un batch est terminé quand son meta.json porte la durée totale."""

    meta_path = output_dir / META_FILE
    if not meta_path.exists():
        return False
    meta, _ = load_batch(output_dir)
    return "duree_totale_s" in meta


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
# Lancement
# --------------------------------------------------------------------------


class MlflowCredentialsError(RuntimeError):
    """Credentials MLflow absents ou refusés par le serveur."""


def missing_mlflow_credentials(env: Mapping[str, str] = os.environ) -> list[str]:
    """Variables d'environnement MLflow manquantes (liste vide si tout y est)."""

    uri = env.get("MLFLOW_TRACKING_URI", "")
    if not uri:
        return ["MLFLOW_TRACKING_URI"]
    if not uri.startswith(("http://", "https://")) or env.get("MLFLOW_TRACKING_TOKEN"):
        return []
    return [
        name
        for name in ("MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD")
        if not env.get(name)
    ]


def setup_mlflow(experiment: str):
    """Vérifier les credentials et la connexion, puis choisir l'expérience."""

    missing = missing_mlflow_credentials()
    if missing:
        raise MlflowCredentialsError(
            f"credentials MLflow absents : {', '.join(missing)} "
            "(lancer le service avec MLflow activé, ou les exporter dans l'environnement)"
        )
    import mlflow

    try:
        # Contacte le serveur : une URI ou des identifiants faux échouent ici.
        mlflow.set_experiment(experiment)
    except Exception as error:
        raise MlflowCredentialsError(
            f"connexion à MLflow impossible ({os.environ['MLFLOW_TRACKING_URI']}) : {error}"
        ) from error
    return mlflow


def run_configuration(mlflow, config: dict[str, Any], selection: Sequence[str], output_dir: Path) -> bool:
    """Un batch = un run MLflow imbriqué. Renvoie False si le batch a échoué."""

    mlflow.start_run(run_name=config["name"], nested=True)
    previous_model = os.environ.get(MODEL_ENV)
    status = "FAILED"
    try:
        mlflow.log_params(
            {
                "modele": config["model"],
                "raisonnement": "oui" if config["reasoning"] else "non",
                "analyse": config["analysis"],
                "approche": config["approach"],
                "prompt_version": PROMPT_VERSIONS[config["approach"]],
            }
        )
        mlflow.set_tag("dossier_batch", str(output_dir))
        os.environ[MODEL_ENV] = config["model"]
        code = explorer_batch.main(batch_argv(config, selection, output_dir))
        if code != 0:
            mlflow.set_tag("erreur", f"explorer_batch a renvoyé {code}")
            return False

        meta, records = load_batch(output_dir)
        annonces, operations = evaluate(records)
        metrics = compute_metrics(annonces, operations)
        (output_dir / METRICS_TEXT_FILE).write_text(
            format_metrics(metrics, meta), encoding="utf-8"
        )
        durations_by_type(annonces).write_csv(output_dir / DURATIONS_FILE)
        mlflow.log_params(
            {
                "types": ",".join(meta.get("types", [])),
                "sample_size": meta.get("sample_size"),
                "per_type": meta.get("per_type"),
                "seed": meta.get("seed"),
                "workers": meta.get("workers"),
                "temperature": meta.get("ask_options", {}).get("temperature", "défaut"),
            }
        )
        mlflow.log_metrics(mlflow_metrics(metrics, meta, annonces))
        mlflow.log_artifacts(str(output_dir))
        status = "FINISHED"
        return True
    except KeyboardInterrupt:
        status = "KILLED"
        raise
    except Exception as error:  # la grille continue sur la configuration suivante
        mlflow.set_tag("erreur", repr(error)[:5000])
        print(f"✗ {config['name']} : {error!r}", file=sys.stderr)
        return False
    finally:
        if previous_model is None:
            os.environ.pop(MODEL_ENV, None)
        else:
            os.environ[MODEL_ENV] = previous_model
        mlflow.end_run(status=status)


def _estimate(grid: Sequence[dict[str, Any]], annonces: int, workers: int) -> str:
    seconds = sum(ESTIMATED_SECONDS[config["reasoning"]] for config in grid) * annonces
    seconds /= max(workers, 1)
    return f"~{seconds / 60:.0f} min (ordre de grandeur, qwen3-8-27b)"


# --------------------------------------------------------------------------
# Ligne de commande
# --------------------------------------------------------------------------


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Lancer explorer_batch sur une grille de configurations, suivie dans MLflow"
    )
    selection = parser.add_argument_group("échantillon (commun à toute la grille)")
    selection.add_argument("--types", nargs="+", help="types annotés à garder ; défaut : tous")
    size = selection.add_mutually_exclusive_group()
    size.add_argument(
        "-n", "--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE,
        help=f"nombre d'opérations tirées (défaut : {DEFAULT_SAMPLE_SIZE})",
    )
    selection.add_argument("--per-type", action="store_true", help="--sample-size par type")
    size.add_argument("--all", action="store_true", help="toutes les opérations, sans tirage (exclut -n)")
    selection.add_argument("--seed", type=int, default=DEFAULT_SEED, help="graine du tirage")
    selection.add_argument("--annotations", default=ANNOTATIONS_PATH, help="fichier d'annotations")
    selection.add_argument(
        "--workers", type=int, default=DEFAULT_WORKERS,
        help=f"appels LLM simultanés dans un batch (défaut : {DEFAULT_WORKERS})",
    )
    selection.add_argument("--temperature", type=float, default=None, help="température du LLM")

    grid = parser.add_argument_group("grille (toutes les valeurs par défaut)")
    grid.add_argument("--modeles", nargs="+", default=list(MODELS), help=f"défaut : {' '.join(MODELS)}")
    grid.add_argument("--raisonnement", nargs="+", choices=tuple(REASONING), default=list(REASONING))
    grid.add_argument("--analyses", nargs="+", choices=tuple(ANALYSIS_MODES), default=list(ANALYSIS_MODES))
    grid.add_argument("--approches", nargs="+", choices=tuple(APPROACHES), default=list(APPROACHES))

    parser.add_argument(
        "--experiment", default=DEFAULT_EXPERIMENT,
        help=f"expérience MLflow (défaut : {DEFAULT_EXPERIMENT}, créée si absente)",
    )
    parser.add_argument(
        "--resume", type=Path, default=None,
        help="reprendre une grille : les configurations déjà terminées sont sautées",
    )
    parser.add_argument("--dry-run", action="store_true", help="lister la grille sans rien lancer")
    parser.add_argument("--yes", action="store_true", help="ne pas demander confirmation")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    missing = missing_mlflow_credentials()
    if missing and not args.dry_run:
        print(f"credentials MLflow absents : {', '.join(missing)}", file=sys.stderr)
        return 2
    try:
        types = parse_types(args.types)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 2

    grid = configurations(
        args.modeles,
        [REASONING[value] for value in dict.fromkeys(args.raisonnement)],
        list(dict.fromkeys(args.analyses)),
        list(dict.fromkeys(args.approches)),
    )
    annotations = load_annotations(args.annotations)
    annonce_ids = select_annonces(
        annotations, types, None if args.all else args.sample_size,
        per_type=args.per_type, seed=args.seed,
    )
    if not annonce_ids:
        print("aucune opération annotée pour ces types", file=sys.stderr)
        return 1

    grid_dir = args.resume or GRID_ROOT / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    saved_ids = grid_dir / "annonces.txt"
    if args.resume and saved_ids.exists() and saved_ids.read_text(encoding="utf-8").split() != annonce_ids:
        print(
            "l'échantillon diffère de celui de la grille à reprendre : "
            "relancer avec les mêmes --types, -n, --seed, --per-type, --all",
            file=sys.stderr,
        )
        return 2
    pending = [config for config in grid if not is_complete(grid_dir / config["name"])]
    calls = len(pending) * len(annonce_ids)
    print(
        f"{len(grid)} configuration(s), {len(pending)} à lancer — {len(annonce_ids)} annonce(s) "
        f"— types {', '.join(types)} — {calls} appels LLM, {_estimate(pending, len(annonce_ids), args.workers)}\n"
        f"dossier : {grid_dir}\nexpérience MLflow : {args.experiment}"
    )
    for config in grid:
        mark = "·" if config in pending else "✓"
        print(f"  {mark} {config['name']}")
    if args.dry_run or not pending:
        if missing:
            print(f"attention : credentials MLflow absents ({', '.join(missing)})", file=sys.stderr)
        return 0
    try:
        mlflow = setup_mlflow(args.experiment)
    except MlflowCredentialsError as error:
        print(error, file=sys.stderr)
        return 2
    if calls > CONFIRMATION_THRESHOLD and not args.yes and sys.stdin.isatty():
        if input(f"Lancer {calls} appels LLM ? [o/N] ").strip().lower() not in {"o", "oui", "y", "yes"}:
            return 1

    selection = selection_argv(args)
    failures = []
    grid_dir.mkdir(parents=True, exist_ok=True)
    (grid_dir / "annonces.txt").write_text("\n".join(annonce_ids) + "\n", encoding="utf-8")
    with mlflow.start_run(run_name=f"grille-{grid_dir.name}"):
        mlflow.set_tag("dossier_grille", str(grid_dir))
        mlflow.log_params(
            {
                "types": ",".join(types),
                "sample_size": None if args.all else args.sample_size,
                "per_type": args.per_type,
                "seed": args.seed,
                "annotations": str(args.annotations),
                "workers": args.workers,
                "configurations": len(pending),
                "annonces": len(annonce_ids),
            }
        )
        mlflow.log_artifact(str(grid_dir / "annonces.txt"))
        for index, config in enumerate(pending, start=1):
            print(f"\n{'#' * 78}\n# [{index}/{len(pending)}] {config['name']}\n{'#' * 78}")
            if not run_configuration(mlflow, config, selection, grid_dir / config["name"]):
                failures.append(config["name"])
        mlflow.log_metric("configurations_en_echec", len(failures))

    if failures:
        print(f"\n{len(failures)} configuration(s) en échec :", *failures, sep="\n  ", file=sys.stderr)
        print(f"reprendre : uv run python explorer_grid.py --resume {grid_dir} …", file=sys.stderr)
        return 1
    print(f"\ngrille terminée : {grid_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
