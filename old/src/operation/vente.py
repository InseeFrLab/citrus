from decimal import Decimal, ROUND_HALF_UP
from typing import Any

from src.llm.prompt import _build_prompt_llm_amount, _build_prompt_llm_date_comptable
from src.llm.client import ask_json
from src import logger
from src.bodacc.api import _clean_json, _get_rs, _get_siren, _get_annee_annonce
from src.operation.base import OperationResult


def _build_prompt_llm_vente_amount(text_a_extraire) -> list[dict]:
    exemple_vente_amount = (
        "pour le texte 'Etablissement principal acquis " +
        "par achat au prix stipulé de 330000 EUR'," +
        " tu dois retourner {'montantNet': 330000}"
    )

    return _build_prompt_llm_amount(
        text_a_extraire=text_a_extraire,
        instructions_exemple=exemple_vente_amount
    )


def extract_amount_vente(json_dict):
    originefonds = json_dict.get("listeetablissements", {}).get("etablissement", {})
    if isinstance(originefonds, list):
        originefonds = originefonds[0].get("origineFonds", "")
    elif isinstance(originefonds, dict):
        originefonds = originefonds.get("origineFonds", "")
    else:
        originefonds = ""

    logger.info(f"Extracting amount from '{originefonds}'")
    return ask_json(
        _build_prompt_llm_vente_amount(originefonds)
    )


def extract_date_comptable_vente(json_dict):
    logger.info(f"Extracting dateEffetComptable")
    date_comptable = json_dict.get("acte", {}).get("vente", {}).get("publiciteLegale", {}).get("date")
    if date_comptable:
        logger.info(f"dateEffetComptable extracted from publiciteLegale: {date_comptable}")
    else:
        date_comptable = json_dict.get("acte", {}).get("dateCommencementActivite", {})
        if date_comptable:
            logger.info(f"dateEffetComptable extracted from dateCommencementActivite: {date_comptable}")
        if not date_comptable:
            descriptif = json_dict.get("acte", {}).get("vente", {}).get("descriptif", "")
            exemple = (
                "pour le texte 'Suivant acte reçu par Maître Dorothée COUCOU" +
                "Notaire à Roubaix, 1 rue de la Paix le 17-06-2026 enregistré au SERVICE DEPARTEMENATL DE L ENREGISTREMENT" +
                " DE LILLE le 22-06-226 dossier 2026 000118218 référence 9999P61 2026 N 01111. Domicile des anciens " +
                "propriétaires: 1 rue de la République 59290 Wasquehal. Siège social du nouveau propriétaire: " +
                "2 rue Tartampion 59100 Roubaix. Les oppositions seront reçues dans les dix jours de la dernière " +
                " en date des publications prévues par la loi pour la correspondance et la validité.'" +
                " Tu dois répondre {'dateEffetComptable': 17-06-2026}."
            )
            logger.info(f"Asking a LLM to extract dateEffetComptable from descriptif")
            date_comptable = ask_json(
                _build_prompt_llm_date_comptable(
                    descriptif,
                    instructions_exemple=exemple,
                    instructions_complementaires="Ne réponds rien si tu trouves plusieurs dates dans le texte.")
                ).get("dateEffetComptable")
            if date_comptable:
                logger.info(f"dateEffetComptable extracted with LLM from descriptif: {date_comptable}")
            else:
                date_comptable = json_dict.get("dateparution", "")
                logger.info(f"dateEffetComptable extracted from dateparution : {date_comptable}")

    return date_comptable


def parse_vente(json_dict):
    """
    Extract from JSON dict sireneCedant, raisonSocialeCedant, sirenBeneficiaire, raisonSocialeBeneficiaire
    """
    logger.info("Cleaning json_dict")
    logger.debug(f"json_dict is {json_dict}")
    json_dict = _clean_json(json_dict)
    sireneCedant = _get_siren(json_dict, key="listeprecedentproprietaire_filtered")
    raisonSocialeCedant = _get_rs(json_dict, key="listeprecedentproprietaire_filtered")
    sirenBeneficiaire = _get_siren(json_dict, key="listepersonnes_filtered")
    raisonSocialeBeneficiaire = _get_rs(json_dict, key="listepersonnes_filtered")
    datecomptable = extract_date_comptable_vente(json_dict)
    anneeCampagne = _get_annee_annonce(json_dict)

    return {
        "anneeCampagne": anneeCampagne, 
        "sirenCedant": sireneCedant,
        "raisonSocialeCedant": raisonSocialeCedant,
        "sirenBeneficiaire": sirenBeneficiaire,
        "raisonSocialeBeneficiaire": raisonSocialeBeneficiaire,
        "dateEffetComptable": datecomptable,
        "typeOperation": "VE",
        "source": json_dict["url_complete"]
    }


def _eur_to_integer_keur(amount: Any) -> int:
    """Apply the legacy evaluator's half-away-from-zero EUR to kEUR conversion."""
    return int((Decimal(str(amount)) / Decimal(1000)).quantize(
        Decimal("1"), rounding=ROUND_HALF_UP
    ))


class VenteSkill:
    """Canonical VE extraction entry point, backed by the legacy helpers."""

    operation_type = "VE"

    def extract(self, announcement: dict[str, Any]) -> OperationResult:
        parsed = parse_vente(announcement)
        amount = extract_amount_vente(announcement).get("montantNet")
        return {
            **parsed,
            "dateRealisationJuridique": None,
            "montantNet": (
                _eur_to_integer_keur(amount) if amount is not None else None
            ),
        }


vente_skill = VenteSkill()
