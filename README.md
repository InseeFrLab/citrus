# citrus

Extraire des annonces légales du [BODACC](https://www.bodacc.fr/) les informations sur les
**restructurations d'entreprises** : le type d'opération parmi huit (`VE`, `FU`, `AB`, `TP`,
`SP`, `AP`, `ST`, `LG`) et les champs métier, en particulier le SIREN du cédant et celui du
bénéficiaire.

**Le dépôt est en cours de développement.** Des assistants d'IA ont été utilisés à divers stades
du processus.

## Approche

L'approche actuelle, dite « bourrin », envoie l'annonce BODACC brute à **un seul appel LLM** avec
un prompt générique, qui renvoie à la fois le type d'opération et les champs. Deux prompts
coexistent :

- `metier` (`bourrin_prompt_metier.md`) : les règles appliquées par les gestionnaires, avec
  lesquelles les annotations de référence ont été produites ;
- `juridique` (`bourrin_prompt_juridique.md`) : les définitions juridiques des huit types.

La normalisation (SIREN, dates, montant en k€) est faite en Python après la réponse du modèle.

Le code antérieur (ancienne chaîne d'évaluation des ventes, pipeline découpé
routing → extraction → benchmarks) est archivé dans [`old/`](./old/README.md).

## Installation

Python ≥ 3.13 et [uv](https://docs.astral.sh/uv/) :

```bash
uv sync
```

Fichier `.env` à la racine (chargé automatiquement) :

```dotenv
LLM_LAB_API_KEY=...
# optionnel :
LLM_LAB_ENDPOINT=https://llm.lab.sspcloud.fr/api
LLM_MODEL_NAME=qwen3-8-27b
# optionnel, tracing :
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
LANGFUSE_HOST=...
```

Les annotations sont lues sur S3/MinIO (`s3://projet-citrus/data/operations_verifiees.parquet`)
via le profil AWS `service-account` du SSP Cloud. `explorer_grid.py` demande en plus les
identifiants MLflow (`MLFLOW_TRACKING_URI`, etc.).

## Utilisation

| Script | Rôle |
|---|---|
| `bourrin.py` | une annonce → un appel LLM → réponse normalisée (session interactive, one-shot, `--json`) |
| `explorer.py` | cellules `# %%` pour explorer une annonce dans VS Code |
| `explorer_batch.py` | évaluation sur un échantillon d'opérations annotées, avec métriques |
| `explorer_grid.py` | toutes les configurations (modèle × raisonnement × analyse × approche), suivies dans MLflow |

```bash
uv run python bourrin.py A20230147853
uv run python explorer_batch.py --types VE LG -n 50
uv run python explorer_grid.py --dry-run
```

Tests (hors ligne) :

```bash
uv run python -m unittest discover -s tests -t .
```

## Licence

Voir [LICENSE](./LICENSE).
