import os
if os.getcwd() != '/home/onyxia/work/citrus':
    os.chdir("citrus")

from src.bodacc.api import bodacc_api, _keep_numero_immat, parse_vente

api = bodacc_api()

# api.requests_url("A202601261236")

bodacc_annonce = api.get_annonce_json("A202601261236")

# l = []
# for personne in bodacc_annonce.get("listeprecedentproprietaire", {}).get("personne", {}):
#     if personne.get("numeroImmatriculation"):
#         l.append(personne)
# l

#         # .get("numeroImmatriculation", {})
#         # .get("numeroIdentification", {})
#         # .replace(" ", "")
#         # )
regex_dict = parse_vente(bodacc_annonce)
