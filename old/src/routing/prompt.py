"""Versioned prompt for semantic BODACC family routing."""

from __future__ import annotations

from collections.abc import Mapping
import json
from typing import Any


ROUTER_PROMPT_VERSION = "family-router-v1"


def build_family_routing_messages(
    context: Mapping[str, Any],
) -> list[dict[str, str]]:
    """Build deterministic French routing messages from normalized facts only."""

    system = """Tu classes une annonce légale BODACC française dans exactement une famille sémantique.

Familles autorisées :
- VE : vente, cession ou acquisition d'un fonds de commerce ou d'actifs d'activité contre une contrepartie financière ; il y a transfert de propriété.
- LG : location-gérance ou mise en location-gérance ; seuls des droits temporaires d'exploitation sont confiés, sans transfert de propriété.
- TP : transmission universelle de patrimoine ; dissolution sans liquidation avec transfert universel au profit de l'associé unique personne morale.
- FUSION_FAMILY : fusion, absorption, scission totale ou partielle, apport partiel d'actifs, ou restructuration apparentée de cette famille.
- UNKNOWN : les faits sont insuffisants, ambigus, contradictoires ou sans rapport avec ces familles.

Appuie-toi uniquement sur le contexte source normalisé. Les faits d'ancien propriétaire peuvent soutenir VE ; les faits d'ancien exploitant peuvent soutenir LG. Ne transforme pas ces indices en règles mécaniques et ne devine jamais : réponds UNKNOWN si les éléments ne permettent pas une classification fiable.

Réponds uniquement par un objet JSON valide, sans Markdown ni texte autour, avec exactement les champs suivants :
{"family":"VE|LG|TP|FUSION_FAMILY|UNKNOWN","evidence":["indice source bref"],"reason":"une phrase concise fondée sur la source"}

family doit être une valeur exacte de la liste. evidence doit être une liste de zéro à trois chaînes brèves tirées du contexte. reason doit être une chaîne concise. Ne fournis ni raisonnement caché ni explication longue."""
    compact_context = json.dumps(
        dict(context),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    user = (
        "Contexte source normalisé à classifier :\n"
        f"{compact_context}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


__all__ = ("ROUTER_PROMPT_VERSION", "build_family_routing_messages")
