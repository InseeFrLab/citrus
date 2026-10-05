"""Versioned prompt for routing inside the fusion family."""

from __future__ import annotations

from collections.abc import Mapping
import json
from typing import Any


FUSION_SUBTYPE_PROMPT_VERSION = "fusion-subtype-v1"


def build_fusion_subtype_messages(
    context: Mapping[str, Any],
) -> list[dict[str, str]]:
    """Build deterministic French messages from normalized source facts."""

    system = """Tu classes une annonce légale BODACC déjà reconnue comme appartenant à la famille fusion, absorption, scission ou apport partiel d'actifs.

Tu dois renseigner les faits juridiques observables avant de choisir exactement un subtype Citrus. N'invente pas un fait absent et ne te fonde que sur le contexte source normalisé.

Axes sémantiques autorisés :
- transfer_scope : TOTAL si tout le patrimoine est transmis, PARTIAL si seule une branche ou une partie d'actif est transmise, UNKNOWN si la source ne permet pas de trancher.
- transferor_fate : DISAPPEARS si le cédant disparaît à l'issue de l'opération, SURVIVES s'il continue d'exister, UNKNOWN si ce sort n'est pas établi.
- beneficiary_creation : NEW si le bénéficiaire est créé pour l'opération, EXISTING s'il préexistait, MIXED_OR_UNKNOWN si plusieurs situations coexistent ou si la source ne permet pas de trancher.
- beneficiary_count : ONE pour un bénéficiaire, MULTIPLE pour plusieurs bénéficiaires, UNKNOWN si leur nombre n'est pas établi.

Subtypes autorisés :
- FU : fusion dans laquelle plusieurs cédants disparaissent et transmettent totalement leur patrimoine à un bénéficiaire nouvellement créé.
- AB : absorption dans laquelle un ou plusieurs cédants disparaissent et transmettent totalement leur patrimoine à un bénéficiaire absorbant préexistant.
- SP : scission partielle ; le cédant survit et ne transmet qu'une partie de son patrimoine à un ou plusieurs bénéficiaires nouvellement créés.
- ST : scission totale ; le cédant disparaît et son patrimoine est réparti entre plusieurs bénéficiaires, qu'ils soient nouveaux, préexistants ou mixtes.
- AP : apport partiel d'actifs ; le cédant survit et ne transmet qu'une partie de son patrimoine à un bénéficiaire préexistant.
- UNKNOWN : les faits sont insuffisants, ambigus ou contradictoires pour distinguer sûrement ces cinq subtypes.

Distinctions difficiles obligatoires :
- FU vs AB : les deux peuvent employer « fusion », « absorption », « société absorbante » ou « société absorbée ». Le critère décisif est un bénéficiaire créé pour l'opération (FU) ou déjà préexistant (AB). Le seul mot « fusion » ne suffit pas.
- SP vs AP : les deux sont des transferts partiels avec survie du cédant et peuvent employer « apport partiel d'actif » ou le régime des scissions. Le critère décisif est un bénéficiaire créé pour l'opération (SP) ou déjà préexistant (AP). Le seul intitulé « apport partiel d'actif » ne suffit pas.
- ST : le cédant disparaît et répartit son patrimoine entre plusieurs bénéficiaires. Ne confonds pas cette scission totale avec un transfert partiel où le cédant survit.

Les axes servent à rendre la décision inspectable. Réponds directement avec les axes, de brefs indices source et une justification concise ; ne fournis pas de raisonnement caché ou détaillé. Si un axe n'est pas étayé, utilise sa valeur d'incertitude. Si le subtype lui-même n'est pas fiable, réponds UNKNOWN au lieu de deviner.

Réponds uniquement par un objet JSON valide, sans Markdown ni texte autour, avec exactement les champs suivants :
{"subtype":"FU|AB|SP|ST|AP|UNKNOWN","transfer_scope":"TOTAL|PARTIAL|UNKNOWN","transferor_fate":"DISAPPEARS|SURVIVES|UNKNOWN","beneficiary_creation":"NEW|EXISTING|MIXED_OR_UNKNOWN","beneficiary_count":"ONE|MULTIPLE|UNKNOWN","evidence":["indice source bref"],"reason":"une phrase concise fondée sur la source"}

evidence doit contenir de zéro à cinq chaînes brèves tirées du contexte. reason doit être une chaîne concise."""
    compact_context = json.dumps(
        dict(context),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    user = (
        "Contexte source normalisé à classifier dans la famille fusion :\n"
        f"{compact_context}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


__all__ = (
    "FUSION_SUBTYPE_PROMPT_VERSION",
    "build_fusion_subtype_messages",
)
