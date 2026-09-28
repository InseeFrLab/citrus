"""Cellules d'exploration interactives — à exécuter une par une dans VS Code.

Séparé de `bourrin.py` pour que le module reste importable : du code au niveau
module s'exécuterait à chaque `import bourrin`, y compris les appels réseau et
les appels LLM.

Kernel : citrus (uv .venv).
"""
# %%
from pprint import pprint

import polars as pl
from dotenv import load_dotenv

load_dotenv(override=True)

from bourrin import (
    afficher_annonce,
    annotation_for,
    format_annonce,
    format_comparison,
    format_result,
    load_annotations,
    describe_annotations,
    run_bourrin,
)
from src.bodacc.api import bodacc_api

pl.Config.set_tbl_cols(-1)
annotations = load_annotations()
print(describe_annotations(annotations))

# %% l'annonce étudiée — la changer ici, puis réexécuter les cellules suivantes
ANNONCE_ID = "A20240102873"
RAISONNEMENT = True  # False : le modèle répond sans raisonner (plus rapide)
APPROCHE = "metier"  # ou "juridique" : chaque approche a son propre prompt
ANALYSE = "complete"  # "courte" (2-3 phrases) ou "aucune" : réponse plus rapide

# %% une annonce de bout en bout
res = run_bourrin(
    ANNONCE_ID, approach=APPROCHE, analysis=ANALYSE, reasoning=RAISONNEMENT
)
print(format_result(res))

# %% la réponse normalisée, et le raccourci de confort
res["envelope"]
res["prediction"]
pprint(res["payload"])

# %% confronter à la référence annotée
print(format_comparison(
    res["envelope"]["operations"],
    annotation_for(annotations, ANNONCE_ID),
))

# %% inspecter une annonce sans appeler le LLM
afficher_annonce(ANNONCE_ID)
annotation_for(annotations, ANNONCE_ID)
annotations.filter(pl.col("ref_annonce_complet") == ANNONCE_ID)

# %% le payload brut
pprint(bodacc_api().fetch_annonce_json(ANNONCE_ID))
