from src import logger


def _build_prompt_llm(text_a_extraire, instructions) -> list[dict]:

    system = (
        "Tu es un assistant pour extraire des informations d'un journal d'annonces légales. "
        "Tu reponds UNIQUEMENT avec un objet JSON valide, sans aucun texte ni "
        "balise Markdown autour. Par exemple si je te demande une balise 'montantNet': "
        '{{"montantNet": 330000}}"'
    )

    user = f"""A partir du contenu donné ci-après, renvoie un objet JSON avec exactement cette clé :

{instructions}


**Texte dont il faut extraire l'information**
{text_a_extraire}"""

    llm_prompt = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    logger.debug(f"LLM prompt is {llm_prompt}")
    return llm_prompt


def _build_prompt_llm_amount(text_a_extraire, instructions_exemple) -> list[dict]:

    instr = (
        '- "montantNet" : extrait le montant en euros uniquement à partir du texte' +
        ' ci dessous. Ne met pas de centimes ou de virgules, juste le montant en euros. ' +
        'Par exemple, ' + str(instructions_exemple)
    )

    return _build_prompt_llm(text_a_extraire=text_a_extraire, instructions=instr)


def _build_prompt_llm_date_comptable(
    text_a_extraire,
    instructions_exemple,
    instructions_complementaires=""
) -> list[dict]:

    instr = (
        '- "dateEffetComptable" : extrait la date uniquement à partir du texte' +
        ' ci dessous. Renvoie la date dans le format JJ-MM-AAAA. ' + str(instructions_complementaires) +
        'Par exemple, ' + str(instructions_exemple)
    )

    return _build_prompt_llm(text_a_extraire=text_a_extraire, instructions=instr)
