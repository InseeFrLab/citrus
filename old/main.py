from src.modele.metrics import filter_metrics_bad_md
from src.modele.evaluate import evaluate_vente

###########################
# Parsing operation Vente
###########################
source = "s3://projet-citrus/data/202607_citrus_bodacc_300ventes.csv"
from_bodacc_df, res, citrus_apibodacc_df = evaluate_vente(source)

filter_metrics_bad_md(citrus_apibodacc_df, res, "raisonSocialeCedant")

filter_metrics_bad_md(citrus_apibodacc_df, res, "dateEffetComptable")

# source = "s3://projet-citrus/data/202607_citrus_bodacc_1800op.csv"