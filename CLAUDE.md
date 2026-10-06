# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`citrus` extracts structured information about **French company restructurings** from BODACC
legal announcements (opendatasoft `annonces-commerciales` dataset): *which* of eight canonical
operation types an announcement is (`VE`, `FU`, `AB`, `TP`, `SP`, `AP`, `ST`, `LG`), and the
business fields it carries — above all the SIREN of the transferor (*cédant*) and of the
beneficiary.

The active approach — single-call extraction (called "bourrin" in git history) — is one generic prompt and one LLM call doing
classification *and* extraction for all eight types: `extract.py`, plus `evaluate.py`, `grid.py`
and `notebook.py` layered on top. Code, comments, CLI flags and prompts are mostly in French.

**`old/` is a frozen archive** of everything earlier that this approach does not use: the legacy
`VE`-only S3 evaluation (`main.py`, `modele/evaluate.py`, `metrics.py`) and the decomposed
pipeline (LLM routing `src/routing`, per-type extraction `src/operation`, the four
`src/modele/*_benchmark.py` runners), plus their tests and full pre-trim copies of
`src/bodacc/api.py` and `src/utils.py`. Its `src.*` imports were not rewritten, so nothing in
`old/` runs as is, and `unittest discover -s tests` does not see `old/tests/`. Do not import from
or edit `old/` unless asked; read it for prior art (e.g. the fusion reconciliation logic).

## Commands

Dependencies are managed with `uv` (Python ≥ 3.13): `uv sync`.

Tests are plain `unittest` (no pytest), fully offline — LLM calls and BODACC fetches are
injected fakes:

```bash
uv run python -m unittest discover -s tests -t .        # whole suite (~80 tests, ~1.5 s)
uv run python -m unittest tests.test_evaluate           # one module
uv run python -m unittest tests.test_grid.GridTest.test_name  # one test
```

No linter, formatter or type checker is configured.

### `extract.py` — one announcement

```bash
uv run python extract.py                      # interactive session
uv run python extract.py A20230147853         # one-shot
uv run python extract.py A20230147853 --json  # normalized envelope only, for piping
uv run python extract.py A20230147853 --approche juridique  # legal-reading prompt (default metier)
uv run python extract.py A20230147853 --analyse courte      # complete|courte|aucune
uv run python extract.py A20230147853 --no-annotations      # skip loading annotations
```

By default it loads annotations from `s3://projet-citrus/data/operations_verifiees.parquet`
(`--annotations` to override; `--json` also skips the load). `--temperature` is forwarded to the
LLM and `--no-reasoning` disables model reasoning; `--approche`, `--analyse`, `--temperature` and
`--no-reasoning` exist on `evaluate.py` too. `extract.py` has no dedicated unit tests; it is
exercised through `tests/test_evaluate.py`.

### `evaluate.py` — batch scoring against annotations

```bash
uv run python evaluate.py                             # 20 random operations
uv run python evaluate.py --types VE LG -n 50         # filter on annotated type_op
uv run python evaluate.py --types FUSION --per-type -n 5   # FUSION = FU AB SP AP ST
uv run python evaluate.py --types TP --all            # no sampling
uv run python evaluate.py --analyse aucune            # no analyse field (fastest)
uv run python evaluate.py --load artifacts/evaluation/<timestamp>  # reopen, no LLM call
uv run python evaluate.py --load s3://projet-citrus/evaluation/<timestamp>  # reopen from S3
uv run python evaluate.py -n 5 --no-s3 --no-mlflow   # local only
```

It samples annotation *rows* (`--seed`, default 0, stable order; asks for confirmation above 100
announcements unless `--yes`), runs `run_extraction` over a thread pool (`--workers`, default 4),
and appends each record to `artifacts/evaluation/<timestamp>/results.jsonl` (`--output-dir`;
gitignored; Ctrl-C keeps partial results) as it arrives, then prints metrics and opens a `batch>`
browsing session (`--no-browse` to skip). `meta.json` records `approche`, `prompt_version`,
`analyse`, `s3_uri`, `mlflow_run_id` and `langfuse_session`.

Persistence (so a batch survives deleting the Onyxia service): the local folder stays the source of
truth, and `src/s3.py:S3Sync` copies it to `<--s3-prefix>/<folder name>/` (default
`s3://projet-citrus/evaluation/`) every 20 records or 2 min via `run_batch(on_record=…)`, then
fully at the end. A mid-batch upload failure (e.g. the SSP Cloud's temporary S3 token expiring)
only logs a warning and is retried on the next trigger. Each batch is also one MLflow run (experiment
`citrus-evaluation`; params, final metrics from `mlflow_metrics`, the console metrics text as artifact
`metriques.txt`, tag `s3_uri`). The data itself
stays on S3, not in MLflow artifacts. S3 write access and MLflow credentials are both checked
before any LLM call (exit 2); `--no-s3` / `--no-mlflow` disable them. There is no resume: an
interrupted batch keeps its partial `results.jsonl` (run status `KILLED`), and a relaunch starts over.
`--load s3://…` downloads into `artifacts/evaluation/s3/<name>/`, so it never overwrites a local
folder. `tests/test_evaluate.py:MainTest` covers `main` end to end with fake S3/MLflow clients.

Scoring details: the annotations (produced by the business rules) are the only reference for
both approaches. Reference rows are stored already normalized via `_reference_value` under their
annotation column names, so `_pair_operations` / `format_comparison` work unchanged on reloaded
batches. Field accuracy is computed on predicted/annotated operation pairs only; amounts count as correct within `AMOUNT_RELATIVE_TOLERANCE` (10 %) of the reference (`_amount_close`). Status
`non annotée` means an extra *predicted* operation left unmatched — the announcement itself is
always annotated.

### `grid.py` — full configuration grid in MLflow

Runs model × reasoning (oui/non) × analyse × approche — 36 configurations by default, models from
`MODELS` in `grid.py`; restrict with `--modeles`, `--raisonnement`, `--analyses`, `--approches`;
`--dry-run` lists without any call — on one shared sample. Each batch is a nested MLflow run under
a `grille-<timestamp>` parent (experiment `citrus-grid` by default).

It does not reimplement the batch: it calls `evaluate.main(argv)` per configuration and switches
the model by setting `LLM_MODEL_NAME`, which `get_model_name` reads on every call — so
configurations run sequentially. Output: `artifacts/grid/<timestamp>/<configuration>/` (plus
`metriques.txt`, `durees_par_type.csv`); `--resume <dir>` skips finished ones and refuses a
different sample. MLflow credentials are mandatory (`MLFLOW_TRACKING_URI`, plus
`MLFLOW_TRACKING_USERNAME`+`_PASSWORD` or `_TOKEN` for http(s)): without them, or if
`set_experiment` fails, it exits with code 2 before any LLM call (`--dry-run` only warns). The grid
owns the MLflow runs, so `batch_argv` passes `--no-mlflow` to `evaluate`, plus
`--s3-prefix <prefix>/grid/<timestamp>/` (S3 is checked by uploading `annonces.txt`).
`run_configuration` copies `meta.json`'s `s3_uri` onto the configuration's run as a tag. Tests use
a fake MLflow and patch `evaluate.main`.

`notebook.py` holds `# %%` cells for a VS Code interactive window. Importing `extract.py` runs
nothing except a REPL convenience: it `chdir`s into `citrus-ia-gen/` if launched from the parent.

## Configuration

`.env` at the repo root is loaded on `import src`. `LLM_LAB_API_KEY` is required for anything
touching an LLM; `LLM_LAB_ENDPOINT` (default `https://llm.lab.sspcloud.fr/api`) and
`LLM_MODEL_NAME` (default `DEFAULT_MODEL` = `qwen3-8-27b` in `src/llm/client.py`) override the
SSP Cloud lab defaults. The endpoint path is
case-sensitive (`/api`, not `/API`); a wrong one surfaces as an opaque `405 Method Not Allowed`.
Langfuse is optional: when both `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are set
(`langfuse_enabled`), `get_client` wraps with `langfuse.openai.OpenAI`, and `run_extraction`
labels each call via `trace_attributes` (langfuse v4 `propagate_attributes`): trace name
`extraction`, tags approach/prompt version/analyse, metadata `annonce_id` (+ `type_annote` in a
batch), and session = batch folder name. Otherwise `trace_attributes` is a no-op and `langfuse` is
never imported. S3/MinIO credentials (`src/s3.py`, shared by polars and boto3) come from the
datalab's `AWS_*` env vars, else the AWS profile `service-account`. MLflow connection helpers live
in `src/tracking.py`. Importing `src` also creates `log/` and opens `log/<timestamp>_citrus.log`.

## Architecture

**Source layer — `src/`**
- `src/__init__.py`: loads `.env`, configures the `citrus` logger (file + console handler).
- `src/bodacc/api.py`: `bodacc_api.fetch_annonce_json` fetches one exact announcement and raises
  categorized `BodaccFetchError`s.
- `src/bodacc/normalization.py`: turns a raw payload into a frozen `NormalizedBodaccAnnouncement`
  exposing **source facts only** — dialect (`RCS-A`/`RCS-B`), parties, descriptions, dates,
  origin-of-funds. No role inference, date cascade or amount normalization; keep it that way.
- `src/llm/client.py`: OpenAI-compatible client (`ask`, `ask_json`, `parse_json_answer`);
  `reasoning=False` sends `reasoning_effort="none"`.
- `src/utils.py`: `is_luhn_valid` (SIREN check), `annuaire` (public record URL).
- `src/s3.py`: S3 credentials, `S3Sync` (batch copy), `download_files`; `src/tracking.py`:
  MLflow credential check and `setup_mlflow` (imports `mlflow` lazily).

**Extraction — `extract.py`**
`run_extraction` sends the raw BODACC payload (string-encoded JSON fields expanded by
`_expand_payload`) to a single LLM call asking for the operation type *and* every business field.

Two approaches, each with its own prompt and its own LLM call (split so one call does not have to
produce both readings). Prompts live in `prompts/<approach>.md`, are loaded into
`_PROMPT_TEMPLATES`, and versioned in `PROMPT_VERSIONS`:

- `metier` (`prompts/metier.md`): the **business rules** operators actually apply — the ordered
  cascade LG → TP → fusion/scission ≤2 → >2 → VE → not retained, textual anchors for dates and
  amounts, one operation per secondary SIREN. The annotations were produced by this process.
- `juridique` (`prompts/juridique.md`): the **legal definitions** of the eight types and the
  confusions to avoid, with no business-rule cascade.

§1 (announcement structure) is duplicated in both prompt files, so edit both.

The prompt files are **templates**, rendered by `system_prompt(approach, analysis)`. The
`--analyse` mode (default `complete`) controls the free-text `analyse` field, which is most of the
generated text and therefore of the latency. It fills `{{ANALYSE_JSON}}` (the key in the JSON
example, dropped for `aucune`) and `{{ANALYSE_REGLES}}` (from `ANALYSIS_RULES[mode][approach]`).
The template bodies must never mention `analyse` themselves — anything the model should explain
goes into `ANALYSIS_RULES`, otherwise `aucune` would still ask for it. `system_prompt` raises if a
`{{` placeholder is left unfilled.

Both approaches return the same flat shape — `{id, retenu, codeTypeOperation, operations[],
analyse}` (`analyse` absent in mode `aucune`); the normalized envelope adds `approche`.
`operations` is a **list**, matching the per-operation granularity of the annotation file.
`approach` and `analysis` are explicit keywords on `run_extraction` / `run_batch` / `run_one`,
kept out of `**ask_options`, which go straight to the LLM client.

Determinism lives in Python, not in the model: `normalize_answer` coerces the free-form answer
(SIREN zero-padding + Luhn warning, several date formats → ISO, unknown type → `UNKNOWN`), and the
amount is asked in EUR then converted to the kEUR contract via `_eur_to_integer_keur`.
`_pair_operations` pairs predicted operations with annotated rows on the (cédant, bénéficiaire)
couple before falling back to order.

Annotation rows are expected to carry `ref_annonce_complet` (join key), `type_op`,
`siren_cedante`, `siren_beneficiaire`, `date_effet_comptable_op`,
`date_realisation_juridique_op`, `montant`.

## Conventions that matter here

- **Never invent a classification.** Unknown or unparseable types normalize to `UNKNOWN`; do not
  fall back to `VE`.
- **Reference labels stay out of the prediction.** `type_op` only selects the sample and is
  joined back for scoring.
- **Dependency injection over patching internals.** `fetch` / `ask_fn` are parameters — that is
  what keeps the test suite offline.
- **Prompt versions.** Bump `PROMPT_VERSIONS` in `extract.py` when editing a `prompts/*.md` file;
  batch `meta.json` and MLflow runs record it.
