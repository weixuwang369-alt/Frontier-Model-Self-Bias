# SelfBias: When & Why Do LLMs Self-Bias?

An executable research program measuring **self-preference bias (SPB)** in LLM judges:
when a model judges outputs (its own and others'), does it favor its own - and if so,
why, and at what text length does the effect switch on?

It's built to be **cloned and run with your own keys**. Point it at any mix of models -
Claude, Gemini, GPT, or open models like **Qwen/Llama via Ollama, vLLM, or OpenRouter** -
bring the built-in prompts or **your own prompt library via Excel**, and get cost-guarded,
resumable runs plus a Streamlit dashboard.

- **RQ1 (mechanism):** *why* does a judge self-bias - recognition of self, or familiarity?
- **RQ2 (threshold):** at what token length does self-bias become measurable, and does it
  track when the model's "dialect" becomes attributable?

> **Status:** the pipeline runs end-to-end and is validated on a deterministic mock
> provider ($0, offline). Real provider adapters are wired for generation, structured
> judging, and recognition probes. The analysis layer computes length-sweep curves with
> prompt-level bootstrap CIs, the RQ2 onset length `L*`, TF-IDF attributability, and a v1
> mechanism regression (`selfbias analyze`). The mixed-effects regression upgrade and the
> Phase-3 fingerprint arm are next - see [`docs/ROADMAP.md`](docs/ROADMAP.md).

## Quickstart

```bash
# 0. one-time: install uv (https://docs.astral.sh/uv) if you don't have it, then:
uv sync                                            # create the venv, install deps

# 1. try it with zero keys (deterministic mock provider, $0, fully offline):
uv run selfbias run config/experiment.mock.yaml    # curate -> generate -> judge -> probe
uv run streamlit run dashboard/app.py              # Configure / Run / Results

# 2. run for real: add your keys, then verify and estimate before spending:
cp .env.example .env                               # fill in the keys you need
uv run selfbias check    config/experiment.example.yaml   # one tiny call per model
uv run selfbias estimate config/experiment.example.yaml   # dry-run cost, no API calls
uv run selfbias run      config/experiment.example.yaml   # confirm, then execute
```

Or use the one-command launcher: `./run.sh` (dashboard), `./run.sh demo` (mock run),
`./run.sh test`.

## Choose your models: any provider, any mix

The roster is the only place model names appear. Use **N ≥ 2** models from any providers:

```yaml
roster:
  - { slot: claude, provider: anthropic,        model: claude-opus-4-8, family: anthropic }
  - { slot: gpt,    provider: openai,           model: gpt-5,           family: openai }
  - { slot: qwen,   provider: openai_compatible, model: "qwen2.5:7b",   family: qwen,
      base_url: "http://localhost:11434/v1", api_key_env: OLLAMA_KEY }   # local Ollama
```

- `provider: openai_compatible` + a `base_url` reaches **any OpenAI-style endpoint** -
  Ollama, vLLM, LM Studio, OpenRouter, Together, Groq - so open models need no new code.
- `family` is a free-form label used to group siblings for the family-bias metric.
- `api_key_env` names the env var holding that model's key (defaults per provider); local
  endpoints usually need none.

Two ready-made configs: [`config/experiment.example.yaml`](config/experiment.example.yaml)
(Claude + Gemini + GPT, ~$283 pilot) and
[`config/experiment.ollama.yaml`](config/experiment.ollama.yaml) (two local open models,
$0).

## Choose your prompts: built-in, your own (Excel), or model-drafted

Set the `prompts.source` in your config:

- **`builtin`** (default): a curated prompt set across the domains.
- **`excel`**: your own prompt library - scaffold a sheet, fill it in, point at it.
- **`llm_generated`**: a roster model drafts a fresh prompt set at run time (one
  cost-guarded call per domain; set `generator_model` and `n_per_domain`).

```bash
uv run selfbias template prompts.xlsx         # writes a fillable Excel/CSV template
uv run selfbias import-check prompts.xlsx     # validate what it parses to
```
```yaml
prompts:
  source: excel
  excel_path: prompts.xlsx
```

The sheet needs only `prompt` + `domain`; optional `reference`, `rubrics`, and
`constraint` columns unlock the subjective and objective-verifiable arms. See
[`data/tasks/seeds/example_prompts.csv`](data/tasks/seeds/example_prompts.csv).

## Cost safety (non-negotiable)

- Every run starts with a **dry-run estimate** (`selfbias estimate`) - call counts ×
  token estimates × [`config/pricing.yaml`](config/pricing.yaml) - shown before any call.
- `budget_usd` in the config is a **hard cap**: the run halts (resumably) when spend would
  cross it. You set it; you own it.
- Runs are **cache-first** (an identical re-run costs ~$0) and **resumable** (rerun to
  continue a halted run - completed work is skipped).
- Nothing is launched implicitly; the dashboard Run action requires a typed confirmation.

## The CLI

| Command | Does |
|---|---|
| `selfbias check <config>` | one tiny real call per model - confirm keys/endpoints work |
| `selfbias estimate <config>` | dry-run cost, per stage/provider (no API calls) |
| `selfbias run <config>` | estimate → confirm → execute (cache-first, budget-capped) |
| `selfbias resume <config>` | continue a halted/partial run |
| `selfbias status <config>` | stages, spend, cache hits for a run |
| `selfbias analyze <config>` | compute metrics from a run: length curves + CIs, onset `L*`, mechanism regression |
| `selfbias template <path>` | write an Excel/CSV prompt-library template |
| `selfbias import-check <path>` | validate a prompt library |
| `selfbias export <config> <out>` | export a run's rows to Excel/CSV |

## Dashboard

`uv run streamlit run dashboard/app.py` - **Configure** (form-edit + validate a config,
live cost preview), **Run** (key status, "Test keys", estimate, confirmed launch with live
progress), **Results** (bias heatmaps, the length-sweep curves, breakdowns). Configure and
Results work with **no keys**; Results shows an illustrative synthetic demo.

## How it works

Pipeline stages, each idempotent, resumable, and cache-first:
**curate** → **generate** (controlled length + truncation series) → **judge** (pairwise +
rubric, order-swapped, judges isolated) → **probe** (self-recognition) → *metrics/analysis*.
All LLM calls go through a provider abstraction; every raw response is persisted so metrics
recompute from disk without re-spending. See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Documentation

| Read | For |
|---|---|
| [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md) | **Start here** - full walkthrough: setup, configuring, running, and reading results |
| [`docs/RESEARCH_PLAN.md`](docs/RESEARCH_PLAN.md) | Hypotheses, design matrix, analysis plan |
| [`docs/METRICS.md`](docs/METRICS.md) | Formal metric definitions (HSPP-R, EO-Bias, onset `L*`, …) |
| [`docs/DECISIONS.md`](docs/DECISIONS.md) | Locked decisions, resolved items, schema migrations |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Modules, schemas, pipeline stages, dashboard |
| [`docs/ROADMAP.md`](docs/ROADMAP.md) | Phases and what's in/out of scope per phase |

## Development

```bash
uv sync --extra dev
uv run pytest        # mock-only; never hits a real API
uv run ruff check .  # lint
```

Providers' SDKs are an optional extra (installed only when you run real models):
`uv sync --extra providers`.

## Grounding literature

Wataoka et al. 2025 (Equal-Opportunity metric; familiarity hypothesis) · Pombal et al.
2026 (HSPP-Ratio; rubric-based SPB; ensemble references) · Russell et al. 2026
(StoryScope; attributability fingerprints) · Li et al. 2025 (calibration; judge
isolation).

## License

[Apache License 2.0](LICENSE) - permissive use/modification/commercial use, with an
explicit patent grant. Copyright 2026 Alex Wang. See [`NOTICE`](NOTICE).
