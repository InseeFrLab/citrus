"""Connexion à MLflow, partagée par `evaluate.py` et `grid.py`.

Les credentials doivent être dans l'environnement (injectés par le service MLflow
du SSP Cloud) : `MLFLOW_TRACKING_URI`, et pour un serveur http(s)
`MLFLOW_TRACKING_USERNAME` + `MLFLOW_TRACKING_PASSWORD` ou `MLFLOW_TRACKING_TOKEN`.
`mlflow` n'est importé qu'à la connexion (import lent).
"""
from __future__ import annotations

import os
from collections.abc import Mapping


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
