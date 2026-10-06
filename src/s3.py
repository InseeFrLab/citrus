"""Accès S3 (MinIO du SSP Cloud) et copie au fil de l'eau d'un dossier de batch.

Les identifiants viennent des variables d'environnement du datalab si elles sont
présentes, sinon du profil AWS ``service-account`` : la même règle sert à polars
(`storage_options`) et à boto3 (`s3_client`).
"""
from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from src import logger

DEFAULT_S3_ENDPOINT = "https://minio.lab.sspcloud.fr"
AWS_PROFILE = "service-account"


class S3SyncError(RuntimeError):
    """Envoi ou téléchargement S3 refusé (identifiants, droits, réseau)."""


def _endpoint() -> str:
    endpoint = os.environ.get("AWS_ENDPOINT_URL")
    if endpoint is None and os.environ.get("AWS_S3_ENDPOINT"):
        endpoint = f"https://{os.environ['AWS_S3_ENDPOINT']}"
    return endpoint or DEFAULT_S3_ENDPOINT


def storage_options() -> dict[str, str] | None:
    """Identifiants S3 au format polars : variables d'environnement, sinon None (profil AWS).

    Le datalab SSP Cloud expose les clés en variables d'environnement ; le reste
    du dépôt suppose un profil AWS nommé ``service-account``. On accepte les deux.
    """

    if not os.environ.get("AWS_ACCESS_KEY_ID"):
        return None
    options = {
        "aws_endpoint_url": _endpoint(),
        "aws_region": os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
        "aws_access_key_id": os.environ["AWS_ACCESS_KEY_ID"],
        "aws_secret_access_key": os.environ["AWS_SECRET_ACCESS_KEY"],
    }
    if os.environ.get("AWS_SESSION_TOKEN"):
        options["aws_session_token"] = os.environ["AWS_SESSION_TOKEN"]
    return options


def s3_client() -> Any:
    """Client boto3, avec les mêmes identifiants que `storage_options`."""

    import boto3

    options = storage_options()
    if options is None:
        return boto3.Session(profile_name=AWS_PROFILE).client(
            "s3", endpoint_url=DEFAULT_S3_ENDPOINT, region_name="us-east-1"
        )
    return boto3.client(
        "s3",
        endpoint_url=options["aws_endpoint_url"],
        region_name=options["aws_region"],
        aws_access_key_id=options["aws_access_key_id"],
        aws_secret_access_key=options["aws_secret_access_key"],
        aws_session_token=options.get("aws_session_token"),
    )


def split_s3_uri(uri: str) -> tuple[str, str]:
    """`s3://bucket/a/b/` → (`bucket`, `a/b`)."""

    if not uri.startswith("s3://"):
        raise ValueError(f"URI S3 attendue (s3://bucket/chemin) : {uri}")
    bucket, _, key = uri[len("s3://"):].partition("/")
    if not bucket:
        raise ValueError(f"bucket manquant dans l'URI S3 : {uri}")
    return bucket, key.strip("/")


def download_files(uri: str, local_dir: Path, names: Iterable[str], *, client: Any = None) -> Path:
    """Télécharger les fichiers `names` du préfixe `uri` dans `local_dir`."""

    bucket, prefix = split_s3_uri(uri)
    client = client or s3_client()
    local_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        key = f"{prefix}/{name}" if prefix else name
        try:
            client.download_file(bucket, key, str(local_dir / name))
        except Exception as error:
            raise S3SyncError(f"téléchargement de s3://{bucket}/{key} impossible : {error}") from error
    return local_dir


class S3Sync:
    """Copie sur S3 d'un dossier local, poussée régulièrement pendant un batch.

    Le dossier local reste la source de vérité. Un échec d'envoi en cours de batch
    ne l'arrête pas (le jeton S3 du SSP Cloud est temporaire et peut expirer) :
    il est compté dans `failures` et le prochain déclenchement réessaie.
    """

    def __init__(
        self,
        uri: str,
        local_dir: Path,
        *,
        live_files: Iterable[str] = (),
        client: Any = None,
        every: int = 20,
        interval_s: float = 120.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.bucket, self.prefix = split_s3_uri(uri)
        self.uri = f"s3://{self.bucket}/{self.prefix}/"
        self.local_dir = Path(local_dir)
        self.live_files = tuple(live_files)
        self.every = every
        self.interval_s = interval_s
        self.failures = 0
        self._client = client
        self._clock = clock
        self._pending = 0
        self._last_attempt = clock()

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = s3_client()
        return self._client

    def _upload(self, name: str) -> None:
        self.client.upload_file(str(self.local_dir / name), self.bucket, f"{self.prefix}/{name}")

    def _push(self, names: Iterable[str]) -> bool:
        try:
            for name in names:
                if (self.local_dir / name).is_file():
                    self._upload(name)
        except Exception as error:  # noqa: BLE001 - un envoi raté n'arrête pas le batch
            self.failures += 1
            logger.warning("envoi vers %s impossible : %s", self.uri, error)
            return False
        return True

    def check(self, name: str) -> None:
        """Envoyer un premier fichier ; lève `S3SyncError` si S3 le refuse."""

        try:
            self._upload(name)
        except Exception as error:
            raise S3SyncError(f"écriture sur {self.uri} impossible : {error}") from error

    def maybe_push(self) -> None:
        """À appeler après chaque résultat : envoie `live_files` tous les `every` ou `interval_s`."""

        self._pending += 1
        if self._pending < self.every and self._clock() - self._last_attempt < self.interval_s:
            return
        self._pending = 0
        self._last_attempt = self._clock()
        self._push(self.live_files)

    def push_all(self) -> bool:
        """Envoyer tous les fichiers du dossier (fin de batch)."""

        return self._push(sorted(path.name for path in self.local_dir.iterdir() if path.is_file()))
