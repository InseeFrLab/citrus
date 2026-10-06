# citrus

Extraire des annonces légales du [BODACC](https://www.bodacc.fr/) les informations sur les
**restructurations d'entreprises** : le type d'opération parmi huit (`VE`, `FU`, `AB`, `TP`,
`SP`, `AP`, `ST`, `LG`) et les champs métier, en particulier le SIREN du cédant et celui du
bénéficiaire.

**Le dépôt est en cours de développement.** Des assistants d'IA ont été utilisés à divers stades
du processus.

## Approche

L'approche actuelle, l'extraction en un appel, envoie l'annonce BODACC brute à **un seul appel LLM** avec
un prompt générique, qui renvoie à la fois le type d'opération et les champs. Deux prompts
coexistent :

- `metier` (`prompts/metier.md`) : les règles appliquées par les gestionnaires, avec
  lesquelles les annotations de référence ont été produites ;
- `juridique` (`prompts/juridique.md`) : les définitions juridiques des huit types.

La normalisation (SIREN, dates, montant en k€) est faite en Python après la réponse du modèle.

Le code antérieur (ancienne chaîne d'évaluation des ventes, pipeline découpé
routing → extraction → benchmarks) est archivé dans [`old/`](./old/README.md).

## Installation

Python ≥ 3.13 et [uv](https://docs.astral.sh/uv/) :

```bash
uv sync
```

Le code lit sa configuration dans les variables d'environnement :

```bash
LLM_LAB_API_KEY=...        # obligatoire pour tout appel LLM
# optionnel :
LLM_LAB_ENDPOINT=https://llm.lab.sspcloud.fr/api
LLM_MODEL_NAME=qwen3-8-27b
# optionnel, tracing :
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
LANGFUSE_HOST=...
```

Un fichier `.env` à la racine n'est pas nécessaire : s'il existe, il est chargé automatiquement.

Les annotations sont lues sur S3/MinIO (`s3://projet-citrus/data/operations_verifiees.parquet`)
avec les variables `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN` et
`AWS_S3_ENDPOINT` exposées par le SSP Cloud, ou à défaut avec le profil AWS `service-account`.
`evaluate.py` et `grid.py` demandent en plus les identifiants MLflow (`MLFLOW_TRACKING_URI`, etc.)
et le droit d'écrire sous `s3://projet-citrus/evaluation/`.

## Utilisation

| Script | Rôle |
|---|---|
| `extract.py` | une annonce → un appel LLM → réponse normalisée (session interactive, one-shot, `--json`) |
| `notebook.py` | cellules `# %%` pour explorer une annonce dans VS Code |
| `evaluate.py` | évaluation sur un échantillon d'opérations annotées, avec métriques |
| `grid.py` | toutes les configurations (modèle × raisonnement × analyse × approche), suivies dans MLflow |

```bash
uv run python extract.py A20230147853
uv run python evaluate.py --types VE LG -n 50
uv run python grid.py --dry-run
```

### Sauvegarde des résultats

Un batch `evaluate.py` est écrit dans `artifacts/evaluation/<horodatage>/`, et ce dossier est
copié au fil de l'eau sur `s3://projet-citrus/evaluation/<horodatage>/` : il survit à la
suppression du service. Chaque batch crée aussi un run MLflow (expérience `citrus-evaluation`)
avec les paramètres, les métriques, le tableau de métriques affiché en console (artefact
`metriques.txt`) et le chemin S3 (tag `s3_uri`). Si Langfuse est configuré, les
appels LLM du batch y sont regroupés dans la session `<horodatage>`, avec l'identifiant de
l'annonce en métadonnée.

```bash
uv run python evaluate.py --load s3://projet-citrus/evaluation/<horodatage>   # rouvrir depuis S3
uv run python evaluate.py -n 5 --no-s3 --no-mlflow                             # essai local seul
```

## Licence

Voir [LICENSE](./LICENSE).
