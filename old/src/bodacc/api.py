import json
import re
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

    def requests_url(self, annonce_id: str) -> str:
        """
        Constructs url to fetch bodacc api from annonce_id

        Args:
            annonce_id (str)
        
        Returns: 
            url of the API requests, as a string
        """
        return f"{self.url}?where=id%3D%22{annonce_id}%22&limit=1&offset=0&timezone=UTC&include_links=false&include_app_metas=false"

    def get_annonce(self, annonce_id: str):
        """
        Fetch an annonce from bodacc api

        Args: 
            annonce_id (str): announce id to fetch
        
        Returns: 
            the response (a requests.response object)
        """
        request_url = self.requests_url(annonce_id)
        logger.info(f"Fetching info from {request_url} for annonce {annonce_id}")
        try:
            response = requests.get(request_url)
        except requests.exceptions.RequestException as e:
            logger.warning(f'Request failed with error {e}')
            response = None

        return response

    def get_annonce_json(self, annonce_id: str) -> dict:
        """
        Extract the first result of a call from Bodacc API

        Arg: 
            annonce_id (str): bodacc annonce id to fetch

        Returns: 
            First response, parsed as a json into a Python dict  
        """
        annonce_content = self.get_annonce(annonce_id).content
        logger.info("Extracting results from annonce content")
        return json.loads(annonce_content).get("results")[0]

    def fetch_annonce_json(
        self, annonce_id: str, *, timeout: float = 30.0
    ) -> dict[str, Any]:
        """Fetch one exact announcement or raise a categorized failure.

        This evaluation-oriented method intentionally lives alongside the
        legacy methods so their behavior remains unchanged. Querying two rows
        makes an unexpected non-unique exact-id result observable.
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

    def search_acte_siren_for_year(
        self,
        siren: str,
        campaign_year: int,
        *,
        timeout: float = 30.0,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Return deterministic source notices mentioning a SIREN in ``acte``."""

        if not isinstance(siren, str) or re.fullmatch(r"\d{9}", siren) is None:
            raise ValueError("siren must contain exactly 9 digits")
        if not isinstance(campaign_year, int) or not 1900 <= campaign_year <= 2100:
            raise ValueError("campaign_year must be between 1900 and 2100")
        if not isinstance(limit, int) or limit <= 0 or limit > 100:
            raise ValueError("limit must be between 1 and 100")
        where = (
            f'search(acte, "{siren}") '
            f"and dateparution >= date'{campaign_year}-01-01' "
            f"and dateparution < date'{campaign_year + 1}-01-01'"
        )
        try:
            response = requests.get(
                self.url,
                params={
                    "where": where,
                    "limit": limit,
                    "offset": 0,
                    "timezone": "UTC",
                    "include_links": "false",
                    "include_app_metas": "false",
                },
                timeout=timeout,
            )
        except requests.exceptions.RequestException as error:
            raise BodaccFetchError(
                "network_exception", f"BODACC search failed: {error}"
            ) from error
        if not 200 <= response.status_code < 300:
            raise BodaccFetchError(
                "http_error",
                f"BODACC search returned HTTP {response.status_code}",
            )
        try:
            payload = response.json()
        except ValueError as error:
            raise BodaccFetchError(
                "invalid_json", "BODACC search response is not valid JSON"
            ) from error
        if not isinstance(payload, Mapping) or not isinstance(
            payload.get("results"), list
        ):
            raise BodaccFetchError(
                "malformed_payload", "BODACC search has no results list"
            )
        if payload.get("total_count", 0) > limit:
            raise BodaccFetchError(
                "search_result_limit_exceeded",
                f"BODACC search for {siren}/{campaign_year} exceeds {limit}",
            )
        if not all(isinstance(item, Mapping) for item in payload["results"]):
            raise BodaccFetchError(
                "malformed_payload", "BODACC search result is not an object"
            )
        return sorted(
            (dict(item) for item in payload["results"]),
            key=lambda item: str(item.get("id", "")),
        )


def _keep_numero_immat(personnes) -> list:
    """
    Filters some keys to keep the one only with immatriculation numbers
    """
    personnes_with_immat = []
    if isinstance(personnes, list):
        logger.debug(f"Cleaning {personnes} as a list")
        for personne in personnes:
            if personne.get("numeroImmatriculation"):
                personnes_with_immat.append(personne)
    elif isinstance(personnes, dict):
        logger.debug(f"Cleaning {personnes} as a dict")
        if personnes.get("numeroImmatriculation"):
            personnes_with_immat.append(personnes)

    return personnes_with_immat


def _string_to_dict(json_dict: dict, key: str) -> dict:

    if "listeprecedentproprietaire" in json_dict.keys():
        logger.info(f"Cleaning {key}")
        json_dict[key] = json.loads(json_dict[key])

    return json_dict


def _clean_json(json_dict: dict) -> dict:
    """
    Clean the response from API : transforms dict stored as a string into a Python dict
    """

    json_dict = _string_to_dict(json_dict, "listeprecedentproprietaire")
    json_dict["listeprecedentproprietaire_filtered"] = _keep_numero_immat(json_dict["listeprecedentproprietaire"].get("personne", []))

    json_dict = _string_to_dict(json_dict, "listepersonnes")
    json_dict["listepersonnes_filtered"] = _keep_numero_immat(json_dict["listepersonnes"].get("personne", []))

    json_dict = _string_to_dict(json_dict, "listeetablissements")

    json_dict = _string_to_dict(json_dict, "acte")

    return json_dict


def _get_siren(json_dict: dict, key="listeprecedentproprietaire_filtered") -> str:
    """
    Extract from JSON dict the first Siren from defined key (either listeprecedentproprietaire or listepersonnes)
    """
    logger.debug(f"Extracting Sirene from {json_dict.get(key, {})}")
    sirene = (json_dict.get(key, {})[0]
        .get("numeroImmatriculation", {})
        .get("numeroIdentification", {})
        .replace(" ", "")
        )
    return sirene


def _get_rs(json_dict: dict, key="listeprecedentproprietaire_filtered") -> str:
    """
    Extract from JSON dict the first raison sociale from defined key (either listeprecedentproprietaire or listepersonnes)
    """
    logger.debug(f"Extracting raison sociale from {json_dict.get(key, {})}")
    return json_dict[key][0].get("denomination", "")


def _get_annee_annonce(json_dict: dict) -> int:
    """
    """
    logger.debug(f"Extraction date parution from {json_dict}")
    return int(json_dict.get("dateparution", "")[:4])
