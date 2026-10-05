from typing import Any, Mapping

import requests

from src import logger


class BodaccFetchError(RuntimeError):
    """Observable failure while fetching one exact BODACC announcement."""

    def __init__(self, code: str, detail: str):
        super().__init__(detail)
        self.code = code
        self.detail = detail


class bodacc_api:
    def __init__(self, dataset_id="annonces-commerciales", end_point="records"):
        hook = "https://bodacc-datadila.opendatasoft.com/api/explore/v2.1/catalog/datasets"

        if end_point:
            self.url = f"{hook}/{dataset_id}/{end_point}"
        else:
            self.url = f"{hook}/{dataset_id}"

    def fetch_annonce_json(
        self, annonce_id: str, *, timeout: float = 30.0
    ) -> dict[str, Any]:
        """Fetch one exact announcement or raise a categorized failure.

        Querying two rows makes an unexpected non-unique exact-id result
        observable. The legacy methods are archived in `old/src/bodacc/api.py`.
        """

        logger.info("Fetching exact BODACC announcement %s", annonce_id)
        try:
            response = requests.get(
                self.url,
                params={
                    "where": f'id="{annonce_id}"',
                    "limit": 2,
                    "offset": 0,
                    "timezone": "UTC",
                    "include_links": "false",
                    "include_app_metas": "false",
                },
                timeout=timeout,
            )
        except requests.exceptions.RequestException as error:
            raise BodaccFetchError(
                "network_exception", f"BODACC request failed: {error}"
            ) from error

        if not 200 <= response.status_code < 300:
            raise BodaccFetchError(
                "http_error",
                f"BODACC returned HTTP {response.status_code}",
            )

        try:
            payload = response.json()
        except ValueError as error:
            raise BodaccFetchError(
                "invalid_json", "BODACC response is not valid JSON"
            ) from error
        if not isinstance(payload, Mapping):
            raise BodaccFetchError(
                "malformed_payload", "BODACC response must be a JSON object"
            )

        results = payload.get("results")
        if not isinstance(results, list):
            raise BodaccFetchError(
                "malformed_payload", "BODACC response has no results list"
            )
        total_count = payload.get("total_count")
        if not results:
            raise BodaccFetchError(
                "zero_results", f"No BODACC result for id {annonce_id}"
            )
        if len(results) > 1 or (
            isinstance(total_count, int) and total_count > 1
        ):
            raise BodaccFetchError(
                "multiple_results",
                f"Expected one BODACC result for id {annonce_id}, got "
                f"{total_count if isinstance(total_count, int) else len(results)}",
            )
        if len(results) != 1 or not isinstance(results[0], Mapping):
            raise BodaccFetchError(
                "malformed_payload", "BODACC result is not a JSON object"
            )
        result = dict(results[0])
        if result.get("id") != annonce_id:
            raise BodaccFetchError(
                "malformed_payload",
                f"BODACC result id does not match requested id {annonce_id}",
            )
        return result
