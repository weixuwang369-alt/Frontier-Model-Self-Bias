# ARCHITECTURE.md

## Repository layout (target end-state)

```
selfbias/
├── README.md
├── pyproject.toml                  # uv-managed
├── .env.example                    # API key names only
├── config/
│   ├── experiment.example.yaml     # full parameter surface (committed)
│   ├── pricing.yaml                # per-model $/Mtok in+out (committed, user-updatable)
│   └── runs/                       # user-created experiment configs (gitignored)
├── docs/                           # research plan, metrics, decisions, roadmap
├── src/selfbias/
│   ├── schemas.py                  # pydantic models for every artifact below
│   ├── providers/                  # base.py, anthropic.py, google.py, openai.py, mock.py
│   ├── cache.py                    # content-addressed response cache
│   ├── costs.py                    # dry-run estimator + live accounting
│   ├── manifest.py                 # run manifests, resumability
│   ├── tasks/                      # domain task curation + verifiable-constraint checkers
│   ├── generation/                 # controlled-length generation + truncation series
│   ├── judging/                    # pwc.py, rubric.py, prompts/ (templates, disclosure arms)
│   ├── probes/                     # recognition probes, confidence elicitation
│   ├── fingerprint/                # Phase 3: templates, discovery, assignment, density
│   ├── metrics/                    # implements docs/METRICS.md 1:1
│   ├── analysis/                   # curves, onsets, regressions, bootstrap
│   └── cli.py                      # typer CLI: estimate / run / resume / status / report
├── dashboard/                      # Streamlit app (thin client over src/selfbias)
│   ├── app.py
│   └── pages/ (1_Configure, 2_Run, 3_Results)
├── data/                           # gitignored except tasks/seeds/
│   ├── tasks/  generations/  judgments/  probes/  features/  manifests/  cache/
└── tests/                          # pytest; mock provider only
```

## Pipeline stages (each idempotent, resumable, cache-first)

1. **curate** - build/validate task sets per domain; attach verifiable checkers
   (objective arm) or rubric sets with polarity + token-length metadata (subjective).
2. **generate** - for each (prompt × roster model × length bin): controlled-length
   generation with tolerance ±20% and bounded retries; record realized tokens; derive
   the truncation series from the longest generation per (prompt, model).
3. **judge** - PWC (order-swapped duplicate calls) and RB, per disclosure arm.
   Confidence elicited in the same structured output. Judges isolated.
4. **probe** - pairwise and single-text recognition probes over controlled-length and
   truncation texts.
5. **verify** - run programmatic checkers (objective arm) → rubric ground truth.
6. **fingerprint** (Phase 3) - discovery pool → structured templates → comparative
   analysis → feature proposal/dedup → corpus-wide assignment → XGBoost + SHAP →
   fingerprint sets and per-text density scores.
7. **metrics** - compute everything in METRICS.md from disk; no API calls.
8. **report** - tables/figures to `data/reports/`; consumed by the dashboard.

## Data schemas (pydantic; JSONL on disk; IDs deterministic)

- **Task**: `task_id` (hash of domain+prompt), `domain`, `prompt`, `constraints[]`
  (objective) | `rubrics[]` ({`rubric_id`, `text`, `polarity`, `weight`, `n_tokens`}),
  `reference_source` ∈ {programmatic, ensemble, human}.
- **Generation**: `gen_id` = hash(task_id, model, length_bin, seed), `model`, `family`,
  `target_tokens`, `realized_tokens`, `text`, `truncation_of` (nullable), `raw_response`,
  `usage`, `familiarity_scores` (nullable; reserved for milestone 2A).
- **Judgment**: `judg_id`, `judge_model`, `paradigm` ∈ {pwc, rubric, da},
  `disclosure` ∈ {anonymous, true_label, false_label}, `disclosed_as` (nullable),
  `subject_gen_ids[]` (1 for RB, 2 ordered for PWC), `verdict`, `per_rubric[]`,
  `confidence`, `position_index`, `raw_response`, `usage`, `seed`.
- **Probe**: `probe_id`, `judge_model`, `probe_type` ∈ {pairwise_recognition,
  single_recognition}, `subject_gen_ids[]`, `answer`, `confidence`, `correct`,
  `raw_response`, `usage`.
- **RunManifest**: `run_id`, config snapshot (frozen), planned call inventory, completed
  call ids, running cost by provider, budget cap, status, seeds.

All rows append-only; corrections via `supersedes: <id>`.

## Provider abstraction

`Provider.generate(request: LLMRequest) -> LLMResponse` where LLMRequest is
provider-agnostic (model string, messages, temperature, seed, max_tokens, structured
output schema). Each adapter handles auth from env, retries with exponential backoff,
rate-limit handling, and usage extraction. `mock.py` returns deterministic canned
responses for tests and for a keyless "demo mode" of the dashboard.

Cache: SHA-256(provider, model, canonical-JSON request) → stored raw response under
`data/cache/`. All pipeline calls check cache first; cache hits cost $0 and are counted
in the manifest's `cache_hit` tally.

## Cost control flow

`selfbias estimate <config>` → call inventory × token estimates × pricing.yaml →
printed table (per stage, per provider, total) → user confirms → `selfbias run`
executes with a hard halt (resumable) at `budget_usd`.

## Dashboard (Streamlit, thin client)

- **Configure**: load/edit/validate experiment YAML via forms (roster, domains, bins,
  arms, sample sizes, budget); writes to `config/runs/`.
- **Run**: shows estimate, requires explicit confirm, streams progress (calls done /
  planned, live spend vs cap, cache-hit rate, per-stage status), supports
  pause/resume via manifest.
- **Results**: bias heatmaps (Δ matrix, HSPP-R tables), length-sweep curves with CIs
  (bias / recognition / attributability overlaid - the H1-vs-H2 money plot), disclosure
  arm comparison, paradigm/domain/rubric-factor breakdowns, calibration diagrams,
  diagnostics (position-bias rate, length compliance, repeatability).
- Keyless behavior: Configure and Results fully functional; Run shows "keys missing".
