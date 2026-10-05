# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`citrus` extracts structured information about **French company restructurings** from BODACC
legal announcements (opendatasoft `annonces-commerciales` dataset): *which* of eight canonical
operation types an announcement is (`VE`, `FU`, `AB`, `TP`, `SP`, `AP`, `ST`, `LG`), and the
business fields it carries — above all the SIREN of the transferor (*cédant*) and of the
beneficiary.

The active approach is "bourrin": one generic prompt, one LLM call doing classification *and*
extraction, for all eight types (`bourrin.py` and the `explorer*.py` scripts around it).

**`old/` is a frozen archive** of everything earlier that bourrin does not use: the legacy
`VE`-only S3 evaluation (`main.py`, `modele/evaluate.py`, `metrics.py`) and the decomposed
pipeline (LLM routing `src/routing`, per-type extraction `src/operation`, the four
`src/modele/*_benchmark.py` runners), plus their tests and full pre-trim copies of
`src/bodacc/api.py` and `src/utils.py`. Its `src.*` imports were not rewritten, so nothing in
`old/` runs as is, and `unittest discover -s tests` does not see `old/tests/`. Do not import from
or edit `old/` unless asked; read it for prior art (e.g. the fusion reconciliation logic).

## Commands

Dependencies are managed with `uv` (Python ≥ 3.13):

```bash
uv sync
```

Tests are plain `unittest` (no pytest), fully offline — LLM calls and BODACC fetches are
injected fakes:

```bash
uv run python -m unittest discover -s tests -t .        # whole suite (~50 tests, <1 s)
uv run python -m unittest tests.test_explorer_batch     # one module
uv run python -m unittest tests.test_explorer_grid.GridTest.test_name  # one test
```

No linter, formatter or type checker is configured (none in `pyproject.toml`, no pre-commit).

The "bourrin" approach — one generic prompt, one LLM call doing routing *and* extraction.
`--approche metier|juridique` (default `metier`) picks the prompt, on both `bourrin.py` and
`explorer_batch.py`:

```bash
uv run python bourrin.py                      # interactive session
uv run python bourrin.py A20230147853         # one-shot
uv run python bourrin.py A20230147853 --json  # normalized envelope only, for piping
uv run python bourrin.py A20230147853 --approche juridique  # legal-reading prompt
uv run python bourrin.py A20230147853 --analyse courte      # analyse capped at 2-3 sentences
uv run python bourrin.py A20230147853 --no-annotations  # skip loading annotations (offline, faster)
```

By default `bourrin.py` loads annotations from `s3://projet-citrus/data/operations_verifiees.parquet`
(override with `--annotations`); `--json` also skips that load. `--temperature` is forwarded to
the LLM, and `--no-reasoning` disables model reasoning (faster, direct answer); both flags also
exist on `explorer_batch.py`. `bourrin.py` has no dedicated unit tests; it is exercised only indirectly through
`tests/test_explorer_batch.py`.

Batch evaluation of the bourrin approach on annotated operations:

```bash
uv run python explorer_batch.py                             # 20 random operations
uv run python explorer_batch.py --types VE LG -n 50         # filter on annotated type_op
uv run python explorer_batch.py --types FUSION --per-type -n 5   # FUSION = FU AB SP AP ST
uv run python explorer_batch.py --types TP --all            # no sampling
uv run python explorer_batch.py --approche juridique -n 20  # legal-reading prompt
uv run python explorer_batch.py --analyse aucune            # no analyse field (fastest)
uv run python explorer_batch.py --load artifacts/bourrin_batch/<timestamp>  # reopen, no LLM call
```

It samples annotation *rows* (seeded via `--seed`, default 0, stable order; asks for confirmation
above 100 announcements unless `--yes`), runs `run_bourrin` over a thread pool
(`--workers`, default 4), appends each record to `artifacts/bourrin_batch/<timestamp>/results.jsonl`
(`--output-dir` to override) as it arrives (gitignored; Ctrl-C keeps partial results), then prints metrics and opens a
`batch>` browsing session (`--no-browse` to skip). Reference rows are stored already normalized
via `_reference_value` under their annotation column names, so bourrin's `_pair_operations` /
`format_comparison` work unchanged on reloaded batches. Both approaches are scored against the annotations (the only reference, produced by the
business rules); `meta.json` records `approche`, `prompt_version` and `analyse`. Field accuracy is
computed on predicted/annotated operation pairs only. An operation with status `non annotée`
is an extra *predicted* operation left unmatched — the announcement itself is always annotated.

`explorer_grid.py` runs the full grid (model × reasoning oui/non × analyse × approche, 36
configurations by default — models from `MODELS` in `explorer_grid.py`; `--modeles`, `--raisonnement`, `--analyses`, `--approches` to
restrict; `--dry-run` lists it without any call) on one shared sample, and logs each batch as a
nested MLflow run under a `grille-<timestamp>` parent (experiment `citrus-bourrin-grille` by
default). It does not reimplement the batch: it calls `explorer_batch.main(argv)` per
configuration and switches the model by setting `LLM_MODEL_NAME`, which `get_model_name` reads on
every call — so configurations run sequentially. Batches go to
`artifacts/bourrin_grid/<timestamp>/<configuration>/` (plus `metriques.txt`,
`durees_par_type.csv`); `--resume <dir>` skips finished ones and refuses a different sample.
MLflow credentials are mandatory (`MLFLOW_TRACKING_URI`, plus `MLFLOW_TRACKING_USERNAME`+`_PASSWORD` or `_TOKEN` for http(s)): without them, or if `set_experiment` fails, the script exits with code 2 before any LLM call (`--dry-run` only warns). Tests:
`tests/test_explorer_grid.py` (fake MLflow, patched `explorer_batch.main`).

`explorer.py` holds the `# %%` cells for driving those functions from a VS Code interactive
window. Importing `bourrin.py` runs nothing except a REPL convenience: it `chdir`s into
`citrus-ia-gen/` if launched from the parent directory.

## Configuration

`.env` at the repo root is loaded on `import src`. `LLM_LAB_API_KEY` is required for anything
touching an LLM; `LLM_LAB_ENDPOINT` (default `https://llm.lab.sspcloud.fr/api`) and
`LLM_MODEL_NAME` (default `qwen3-8-27b`, from `DEFAULT_MODEL` in `src/llm/client.py` — the
module docstring still says `gemma4-26b-moe`, which is stale) override the SSP Cloud lab defaults. Beware that
the endpoint path is case-sensitive (`/api`, not `/API`) and that a wrong one surfaces as an
opaque `405 Method Not Allowed`. Langfuse is optional: `get_client` only wraps with
`langfuse.openai.OpenAI` when both `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are set. S3/MinIO access uses the AWS profile
`service-account`. Importing `src` also creates `log/` and opens `log/<timestamp>_citrus.log`.

## Architecture

**Source layer — `src/`** (only what bourrin needs)
- `src/__init__.py`: loads `.env`, configures the `citrus` logger (file + console handler).
- `src/bodacc/api.py`: `bodacc_api.fetch_annonce_json` fetches one exact announcement and raises
  categorized `BodaccFetchError`s.
- `src/bodacc/normalization.py`: turns a raw payload into a frozen `NormalizedBodaccAnnouncement`
  exposing **source facts only** — dialect (`RCS-A`/`RCS-B`), parties, descriptions, dates,
  origin-of-funds. No role inference, date cascade or amount normalization; keep it that way.
- `src/llm/client.py`: OpenAI-compatible client (`ask`, `ask_json`, `parse_json_answer`);
  `reasoning=False` sends `reasoning_effort="none"`.
- `src/utils.py`: `is_luhn_valid` (SIREN check), `annuaire` (public record URL).

**The bourrin approach — `bourrin.py`**
`run_bourrin` sends the raw BODACC payload (string-encoded JSON fields expanded by
`_expand_payload`) to a single LLM call with one generic French prompt that asks for the
operation type *and* every business field at once.

There are two approaches, each with its own prompt and its own LLM call (split so that one call
does not have to produce both readings). Both prompts sit at the repo root, are read verbatim
into `SYSTEM_PROMPTS[approach]`, and are versioned in `PROMPT_VERSIONS`:

- `metier` (`bourrin_prompt_metier.md`): the **business rules** the operators actually apply —
  the ordered cascade LG → TP → fusion/scission ≤2 → >2 → VE → not retained, the textual anchors
  for dates and amounts, one operation per secondary SIREN. The annotations were produced by
  this process.
- `juridique` (`bourrin_prompt_juridique.md`): the **legal definitions** of the eight types and
  the confusions to avoid, with no business-rule cascade.

The section on announcement structure (§1) is duplicated in both files, so edit both.

The `.md` files are **templates**, rendered by `system_prompt(approach, analysis)`.
`--analyse complete|courte|aucune` (default `complete`, on both CLIs) controls the free-text
`analyse` field, which is most of the generated text and therefore of the latency. The mode
fills two placeholders: `{{ANALYSE_JSON}}` (the key in the JSON example, dropped for `aucune`)
and `{{ANALYSE_REGLES}}` (the output rule, taken from `ANALYSIS_RULES[mode][approach]` in
`bourrin.py`). The template bodies must never mention `analyse` themselves — anything the model
should explain goes into `ANALYSIS_RULES`. Otherwise `aucune` would still ask for it.
`system_prompt` raises if a `{{` placeholder is left unfilled.

Both approaches return the same flat shape — `{id, retenu, codeTypeOperation, operations[],
analyse}`, with `analyse` absent in mode `aucune` — and the
normalized envelope adds `approche`. `operations` is a **list**, matching the per-operation
granularity of the annotation file. `approach` is an explicit keyword on `run_bourrin` /
`run_batch` / `run_one` (as is `analysis`), kept out of `**ask_options`, which go straight
to the LLM client.

Determinism lives in Python, not in the model: `normalize_answer` coerces the free-form answer
(SIREN zero-padding + Luhn warning, several date formats → ISO, unknown type → `UNKNOWN`), and
the amount is asked in EUR then converted to the kEUR contract via `_eur_to_integer_keur`.
`format_comparison` pairs predicted operations with annotated rows on the (cédant, bénéficiaire)
couple before falling back to order. `fetch`/`ask_fn` are injectable, as elsewhere.

**Scripts layered on top:** `explorer.py` (VS Code cells) and `explorer_batch.py` import from
`bourrin.py`; `explorer_grid.py` drives `explorer_batch.main`. Annotation rows are expected to
carry `ref_annonce_complet` (join key), `type_op`, `siren_cedante`, `siren_beneficiaire`,
`date_effet_comptable_op`, `date_realisation_juridique_op`, `montant`.

## Conventions that matter here

- **Never invent a classification.** Unknown or unparseable types normalize to `UNKNOWN`; do not
  fall back to `VE`.
- **Reference labels stay out of the prediction.** `type_op` only selects the sample and is
  joined back for scoring.
- **Dependency injection over patching internals.** `fetch` / `ask_fn` are parameters — that is
  what keeps the test suite offline.
- **Prompt versions.** Bump `PROMPT_VERSIONS` in `bourrin.py` when editing a `bourrin_prompt_*.md`
  file; batch `meta.json` and MLflow runs record it.
