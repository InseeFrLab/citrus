# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`citrus` extracts structured information about **French company restructurings** from BODACC
legal announcements (opendatasoft `annonces-commerciales` dataset): *which* of eight canonical
operation types an announcement is (`VE`, `FU`, `AB`, `TP`, `SP`, `AP`, `ST`, `LG`), and the
business fields it carries — above all the SIREN of the transferor (*cédant*) and of the
beneficiary.

The two halves have different coverage, and are **not yet wired together**:

- **Classification** (`src/routing`) handles all eight types.
- **Extraction** (`src/operation`) has skills for `VE`, `LG` and `TP` only; nothing extracts
  fields for the fusion family (`FU`, `AB`, `SP`, `AP`, `ST`).
- No code path runs router → skill end to end. `oracle_benchmark` selects the skill from the
  *annotated* `type_op`, and the routing benchmarks run classification without extraction; tests
  assert this separation (e.g.
  `RoutingTest.test_router_has_no_operation_skill_dependency_or_dispatch` in `tests/test_routing.py`).

Note: `README.md` predates the routing/benchmark layer. It still calls classification
unimplemented and lists `TP`/`LG` among the missing parsers; both are outdated. `main.py` is the
legacy `VE`-only S3 path and touches neither `src/routing` nor `src/operation` — the current
entry points are the four `src/modele/*_benchmark.py` runners.

## Commands

Dependencies are managed with `uv` (Python ≥ 3.13):

```bash
uv sync
```

Tests are plain `unittest` (no pytest), fully offline — LLM calls and BODACC fetches are
injected fakes:

```bash
uv run python -m unittest discover -s tests -t .        # whole suite (~240 tests, ~2 s)
uv run python -m unittest tests.test_routing            # one module
uv run python -m unittest tests.test_routing.RoutingTest.test_name  # one test
```

No linter, formatter or type checker is configured (none in `pyproject.toml`, no pre-commit).

Benchmarks are the real-data entry points; each takes an annotations CSV/Parquet and an output
directory, and hits both the BODACC API and the LLM lab:

```bash
uv run python -m src.modele.routing_benchmark --annotations <file> --output-dir <dir> [--max-per-type 10|all]
uv run python -m src.modele.fusion_subtype_benchmark --annotations <file> --output-dir <dir> [--max-per-type 5|all]
uv run python -m src.modele.fusion_reconciliation_benchmark --annotations <file> --output-dir <dir> [--max-seeds 5 | --all]
uv run python -m src.modele.oracle_benchmark --annotations <file> --output-dir <dir> [--max-ve N] [--max-lg N] [--max-tp N] [--amount-tolerance 0.1]
```

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
configurations by default; `--modeles`, `--raisonnement`, `--analyses`, `--approches` to
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
`citrus-ia-gen/` if launched from the parent directory. `test.py` is a stale scratch script
(hardcoded `chdir` to a `citrus/` path that no longer exists) — not part of the test suite.

Legacy S3-backed vente evaluation (reads/writes `s3://projet-citrus/...`):

```bash
uv run main.py
```

Call graph regeneration: `bash docs/graphs.sh` (code2flow → `docs/call_graph_all_but_test.png`).
Despite the name, the script excludes `./src/test/*`, which does not exist, so `tests/` is
included in the graph.

## Configuration

`.env` at the repo root is loaded on `import src`. `LLM_LAB_API_KEY` is required for anything
touching an LLM; `LLM_LAB_ENDPOINT` (default `https://llm.lab.sspcloud.fr/api`) and
`LLM_MODEL_NAME` (default `qwen3-8-27b`) override the SSP Cloud lab defaults. Beware that
the endpoint path is case-sensitive (`/api`, not `/API`) and that a wrong one surfaces as an
opaque `405 Method Not Allowed`. Langfuse is optional: `get_client` only wraps with
`langfuse.openai.OpenAI` when both `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are set. S3/MinIO access uses the AWS profile
`service-account`. Importing `src` also creates `log/` and opens `log/<timestamp>_citrus.log`.

## Architecture

The pipeline is layered so that each boundary can be tested and benchmarked alone.

**1. Source normalization — `src/bodacc/`**
`api.py` has two generations side by side: legacy `bodacc_api.get_annonce_json` (returns the
first result, no error typing) and the evaluation-oriented `fetch_annonce_json`, which raises
categorized `BodaccFetchError`s. `normalization.py` turns a raw payload into a frozen
`NormalizedBodaccAnnouncement` exposing **source facts only** — dialect (`RCS-A`/`RCS-B`),
parties, descriptions, dates, origin-of-funds. It performs no role inference, no date cascade,
no amount normalization; that separation is deliberate and should be preserved.

**2. Semantic routing — `src/routing/`**
Two stages, both LLM-backed, both returning validated frozen dataclasses:

- `router.py` (`family_router`) picks a coarse `RoutingFamily`: `VE`, `LG`, `TP`,
  `FUSION_FAMILY` or `UNKNOWN`. `build_routing_context` projects the normalized announcement
  into a compact deterministic JSON context (dates deliberately excluded).
- The fusion family then has two alternative paths:
  - `fusion_subtype.py` — single-shot LLM router straight to `FU`/`AB`/`SP`/`ST`/`AP`, plus
    inspectable axes (transfer scope, transferor fate, beneficiary creation/count).
  - `fusion_semantics.py` + `fusion_reconciliation.py` — the two-phase design. The LLM parses
    one announcement into source-grounded participants and a `LegalFamily`; then
    `build_fusion_provisional` produces a provisional record using internal states `FZ`/`SZ`,
    and `reconcile_fusion_family` runs a **pure, offline, campaign-global** pass that resolves
    them (an `AB` anchor propagates to `FZ` rows sharing a beneficiary SIREN; `SP` anchors
    propagate through transferor SIRENs unless local facts conflict). Rows are grouped via
    `description_fingerprint` (NFKC + whitespace canonicalization + SHA-256).

Prompts live in dedicated `*_prompt.py` modules, are written in French, demand a single JSON
object, and each carries a version constant (`ROUTER_PROMPT_VERSION`,
`FUSION_SUBTYPE_PROMPT_VERSION`, `FUSION_SEMANTICS_PROMPT_VERSION`, …) alongside taxonomy
versions. Bump these when you change a prompt or a label set — benchmark summaries record them.

**3. Extraction skills — `src/operation/`**
"Skill" here is this repo's own vocabulary for a per-operation-type extraction unit — a plain
`Protocol` implementation, unrelated to agentic Agent Skills (`SKILL.md`), tool calling or any
agent loop, none of which exist in this codebase. Each skill implements the `OperationSkill`
protocol (`operation_type` + `extract(announcement)
-> OperationResult`, a fixed `TypedDict` of Citrus business fields). `vente.py` mixes structured
JSON reads with LLM extraction for free-text amount/date; `transmission_patrimoine.py` and
`location_gerance.py` are deterministic (regex/date parsing over normalized facts).

**Alternative: the "bourrin" approach — `bourrin.py`**
A deliberate counter-experiment to layers 2 and 3: `run_bourrin` sends the raw (dict-expanded)
BODACC payload to a single LLM call with one generic French prompt that asks for the operation
type *and* every business field at once.

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

Note `_expand_payload`: `bourrin.py` deliberately does **not** use `src.bodacc.api._clean_json`,
which tests a hardcoded key and passes `None` to `json.loads` — it raises on 7 of the 8
operation types, only vente payloads survive it. Do not reuse `_clean_json` outside the vente
path.

**4. Evaluation — `src/modele/`**
`benchmark.py` is the pure offline comparator: it normalizes annotations and predictions and
compares them field by field on `JOIN_KEY = "ref_annonce_complet"`. `bodacc_lookup.py` derives
the exact BODACC OpenData id from annotation references and refuses to synthesize one when
anything is ambiguous. The four runners wire fetch → route/extract → compare, write Parquet
predictions/errors plus a `*_summary.json` (model name, prompt/taxonomy versions, git HEAD),
and select deterministic sub-samples in stable id order. `evaluate.py` + `metrics.py` are the
older polars/S3 vente path used by `main.py`.

Annotation files are expected to carry `ref_annonce`, `numero_annonce`, `ref_annonce_complet`,
`type_op`, `siren_cedante`, `siren_beneficiaire`, `date_effet_comptable_op`,
`date_realisation_juridique_op`, `montant`.

## Conventions that matter here

- **Never invent a classification.** `UNKNOWN` is a first-class outcome at every routing stage;
  ambiguous announcements must not fall back to `VE`. `fusion_reconciliation` documents a
  historical `FU`/`ST` → `AP` fallback it explicitly does *not* implement
  (`HISTORICAL_ISOLATED_AP_FALLBACK_IMPLEMENTED = False`) rather than guess.
- **Reference labels stay out of the pipeline.** `type_op` is used only to build cohorts or as a
  declared oracle (`oracle_benchmark`), and is joined back after prediction.
- **Strict LLM output validation.** Every router parses JSON strictly (duplicate keys rejected,
  evidence/reason length-capped) and raises typed `*LLMError` / `*OutputError` subclasses instead
  of coercing a bad answer.
- **Dependency injection over patching internals.** Routers and benchmark runners accept `ask`
  functions, fetchers and skill maps as parameters — that is what keeps the whole suite offline.
- **Two code styles coexist.** Newer modules (`bodacc/normalization`, `routing/*`, `operation/*`,
  `modele/*benchmark*`) use `from __future__ import annotations`, frozen `slots=True` dataclasses,
  `str`-valued `Enum`s and exhaustive docstrings; legacy modules (`bodacc/api`, `operation/vente`
  helpers, `modele/evaluate`, `modele/metrics`, `main.py`, `test.py`) are looser. Match the style
  of the module you are editing rather than the repo average.
