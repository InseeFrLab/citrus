> **Archive.** Ce dossier contient le code antérieur à l'approche « bourrin » qui n'est plus
> utilisé : l'ancienne chaîne d'évaluation des ventes (`main.py`, `src/modele/evaluate.py`,
> `metrics.py`) et le pipeline découpé routing → extraction → benchmarks (`src/routing`,
> `src/operation`, `src/modele/*_benchmark.py`), avec leurs tests. `src/bodacc/api.py` et
> `src/utils.py` sont des copies intégrales des versions d'avant l'élagage.
> Les imports `src.*` n'ont pas été réécrits : rien ici ne s'exécute en l'état.
> Le README ci-dessous est celui d'origine.

# citrus

Extract and structure information about **company restructurings** from the French
[BODACC](https://www.bodacc.fr/) (*Bulletin Officiel des Annonces Civiles et
Commerciales*).

For each legal announcement, citrus fetches the raw record from the BODACC open
API, figures out **what kind of restructuring** it describes, and extracts the
structured fields that matter — in particular the SIREN of the transferor and
the SIREN of the beneficiary.

**The repo is still work in progress**. 

Des assistants d'IA ont par ailleurs été utilisés à divers stades du processus. 

## What it does

BODACC publishes legal announcements about the life of French companies. citrus
targets eight canonical restructuring categories:

| Code  | Category                                   | Business meaning / typical signal |
|-------|--------------------------------------------|-------------------------------------|
| `VE`  | Vente / cession                            | "acquis par", "achat au prix stipulé", "achat", "cession" |
| `FU`  | Fusion                                     | Several companies form a new beneficiary |
| `AB`  | Absorption                                 | Transfer into an existing beneficiary |
| `TP`  | Transmission universelle de patrimoine     | Commonly abbreviated TUP |
| `SP`  | Scission partielle                         | Partial transfer to newly created companies |
| `AP`  | Apport partiel d'actifs                    | Partial transfer to an existing beneficiary |
| `ST`  | Scission totale                            | Disappearing transferor split among beneficiaries |
| `LG`  | Location-gérance                           | Operation without ownership transfer |

Only `VE` currently has a complete POC extraction and evaluation pipeline. The
other types are targets, not implemented capabilities. Unknown or ambiguous
announcements must not silently default to `VE`.

## How it works

The pipeline mixes deterministic parsing with LLM calls:

1. **Fetch** — `bodacc_api` queries the opendatasoft catalog
   (`annonces-commerciales` dataset) for a given announcement id and returns the
   first result as a Python dict.
2. **Classify** — planned classification will gather the free-text description
   from the payload and choose the relevant operation logic. This is work in
   progress and does not yet implement the target taxonomy.
3. **Parse** — depending on the type, the matching parser extracts the
   structured fields. Structured/identifier fields (SIREN, raison sociale) are
   read directly from the JSON; free-form fields that vary from one greffe to
   the next (amount, accounting effect date) are extracted with an LLM.
4. **Evaluate** — the extraction is compared field by field against a reference
   file and per-field accuracy metrics are written out.

LLM access goes through an OpenAI-compatible client pointed at the SSP Cloud LLM
lab, wrapped with [Langfuse](https://langfuse.com/) for tracing. All prompts ask
the model to answer with a single JSON object, which is parsed back into a dict.

## Project structure

```
citrus/
├── main.py                     # entry point: run the vente evaluation pipeline
├── test.py                     # quick manual checks on a single announcement
└── src/
    ├── __init__.py             # loads .env, configures the "citrus" logger
    ├── utils.py                # annuaire URL, Luhn SIREN check, date parsing
    ├── bodacc/
    │   └── api.py              # bodacc_api client + JSON cleaning / field extraction
    ├── llm/
    │   ├── client.py           # OpenAI-compatible client, ask() / ask_json()
    │   └── prompt.py           # prompt builders (base, amount, date, type)
    ├── operation/
    │   └── vente.py            # vente parser + LLM amount / date extraction
    └── modele/
        ├── evaluate.py         # evaluate_vente: fetch → parse → compare → save
        └── metrics.py          # per-field comparison metrics
```

## Requirements

- Python **≥ 3.13**
- [uv](https://docs.astral.sh/uv/) for dependency management
- Access to the SSP Cloud LLM lab (for the LLM calls) and to the project's
  S3/MinIO bucket (for reading the input files and writing results)

Main dependencies: `polars`, `boto3`, `openai`, `langfuse`, `requests`,
`unidecode`, `python-dotenv`.

## Installation

```bash
git clone https://github.com/Krrcharles/citrus-ia-perso.git
cd citrus-ia-perso
uv sync
```

## Configuration

Create a `.env` file at the repository root (loaded automatically on import).

LLM lab access:

```dotenv
LLM_LAB_API_KEY=your-api-key
# optional overrides:
LLM_LAB_ENDPOINT=https://llm.lab.sspcloud.fr/api
LLM_MODEL_NAME=gemma4-26b-moe
```

Langfuse tracing (used by the LLM client):

```dotenv
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
LANGFUSE_HOST=...
```

S3 / MinIO access is read from an AWS profile named `service-account` pointing at
`https://minio.lab.sspcloud.fr` (the default SSP Cloud setup), so no extra
configuration is needed there when running on the platform.

The logger writes to `log/<timestamp>_citrus.log`. It creates a `log/` directory
if one doesn't already exist.

## Usage

### Run the vente evaluation pipeline

`main.py` runs the full fetch → parse → compare loop over a reference CSV stored
on S3 and prints where the extraction disagrees with the reference:

```bash
uv run main.py
```

It reads the source file (`s3://projet-citrus/data/...`), extracts fields for
every announcement, writes the extraction and the metrics back to S3, and shows
the mismatching rows for chosen fields (e.g. `raisonSocialeCedant`,
`dateEffetComptable`).

## Extraction details

For a **vente**, `parse_vente` returns:

- `sirenCedant`, `raisonSocialeCedant` — the company being sold (previous owner)
- `sirenBeneficiaire`, `raisonSocialeBeneficiaire` — the buyer
- `dateEffetComptable` — accounting effect date (from the structured field when
  present, otherwise extracted from the free text by the LLM, otherwise the
  publication date)
- `montantNet` — the amount, extracted by the LLM from the `origineFonds` text
- `anneeCampagne`, `typeOperation` (`"VE"`), `source`

SIREN numbers can be sanity-checked with `src.utils.is_luhn_valid`, and any SIREN
resolved to its public record via `src.utils.annuaire`.

## Status

- ✅ Fetching from the BODACC API
- ⏳ Classification across the eight-type target taxonomy — work in progress
- ✅ Full parsing + evaluation pipeline for **ventes**
- ⏳ Parsers for `FU`, `AB`, `TP`, `SP`, `AP`, `ST`, and `LG` (not yet implemented)

## License

See [LICENSE](./LICENSE).
