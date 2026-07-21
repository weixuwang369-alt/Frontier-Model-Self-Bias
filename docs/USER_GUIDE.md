# SelfBias User Guide

This guide walks you through running SelfBias end to end: what it measures, how to set up
an experiment, how to run it safely, and how to read the results. It aims for the middle
ground between a README and a research paper. You do not need to be a machine-learning
researcher to use it, but the precise definitions are here when you want them.

---

## 1. What SelfBias measures

When a language model acts as a **judge** (grading text that models produce), does it tend
to prefer its **own** writing, or writing from its **family** (sibling models from the same
maker)? That tendency is called **self-preference bias**. SelfBias measures it end to end
and asks two questions:

- **Why does it happen?** Two competing explanations: the judge *recognizes* its own
  writing, or it just favors text that *feels familiar* to it, with authorship incidental.
- **When does it show up?** How long does a piece of text have to be before the bias is
  measurable, and does that onset line up with when the text becomes *attributable* (when a
  classifier, or the model itself, can tell who wrote it)?

The whole thing runs as a pipeline: models **write** responses at controlled lengths, other
models **judge** them, the judges take **recognition** quizzes, and then an analysis step
computes the numbers and charts.

You supply your own API keys (or point at local open models). Nothing runs, and nothing
costs money, unless you start it.

---

## 2. Requirements and setup

- **Python 3.11+** and **[uv](https://docs.astral.sh/uv/)** (the package manager this
  project uses).
- Clone the repository, then from its root:

```bash
uv sync            # creates the virtual environment and installs everything
```

That is the only setup. API keys are optional and only needed for real runs (see §5).

> If `uv` is not on your PATH in a fresh terminal, add it once:
> `export PATH="$HOME/Library/Python/3.14/bin:$PATH"` (adjust the path to where uv lives).

---

## 3. A 60-second tour (no API keys)

Everything works offline using a built-in **mock** model that returns deterministic,
made-up text. This is how you learn the tool without spending anything.

```bash
./run.sh demo       # run the full pipeline on the mock model ($0, offline)
./run.sh analyze    # compute the metrics and charts from that run
./run.sh            # open the dashboard in your browser
```

In the dashboard, the **Results** page opens with example data so every chart is populated.
Your own analyzed runs appear at the bottom of that page.

`./run.sh` is a convenience wrapper. The real commands are `uv run selfbias <command>` and
`uv run streamlit run dashboard/app.py`; see §9 for the full list.

---

## 4. Key ideas, in plain terms

You will see these words throughout the app and config. Here is what they mean.

| Term | Plain meaning |
|---|---|
| **Roster** | The list of models in the experiment. Each one both writes and judges. |
| **Generator / judge** | The same models play both roles: a model *generates* text, and models (including itself) *judge* it. |
| **Family** | A label grouping sibling models (e.g. two Claude models share the family "anthropic"). Used to measure family favoritism. |
| **Domain** | The kind of task: verifiable instructions, open Q&A/summarizing, or creative writing. |
| **Prompt / task** | One instruction a model responds to. |
| **Length bin** | A target length (in tokens) for the writing, so you can watch how bias changes with length. |
| **Paradigm** | How judging works: **pairwise** (compare two pieces head-to-head) or **rubric** (score one piece against a checklist). |
| **Reference (ground truth)** | What "correct" means. For verifiable tasks it is computed exactly; for subjective tasks it is the majority vote of all the judges. |
| **Recognition probe** | A separate quiz where a model is shown two pieces and asked which one it wrote. |
| **Attributability** | Whether a simple classifier can guess who wrote a piece from its style. |

And the headline **measurements**:

- **HSPP-Ratio (HSPP-R)** - the main bias number. It compares how often a judge
  over-credits its own writing versus how often it over-credits unrelated models. **1.0
  means fair; above 1 means it favors itself.** It is *leniency-normalized*, so a judge
  that is generous to everyone does not look biased.
- **Overestimation rate** - among cases where a writer objectively fell short, how often a
  judge passed them anyway. This is the raw ingredient of HSPP-R.
- **Onset length (L\*)** - the shortest text length at which a signal (recognition,
  attributability, or bias) becomes statistically real (its confidence interval clears the
  no-effect line). This is the answer to "when does it show up?"
- **Bootstrap confidence interval** - the shaded bands on the charts. They show how much
  the number would wobble if you resampled your prompts, so you know what is solid versus
  noise.
- **Mechanism regression** - a statistical model that estimates how much each factor
  (length, whose writing it is, the kind of task) pushes a judge toward favoritism.

---

## 5. Choosing your models

The **roster** is the only place model names appear. Use any two or more models, from any
mix of providers. Example:

```yaml
roster:
  - { slot: claude, provider: anthropic,        model: claude-opus-4-8, family: anthropic }
  - { slot: gpt,    provider: openai,           model: gpt-5,           family: openai }
  - { slot: qwen,   provider: openai_compatible, model: "qwen2.5:7b",   family: qwen,
      base_url: "http://localhost:11434/v1", api_key_env: OLLAMA_KEY }
```

- **`provider`** is the adapter type: `anthropic`, `google`, `openai`, `openai_compatible`,
  or `mock`.
- **`openai_compatible` + `base_url`** reaches any OpenAI-style endpoint, which covers
  **local models** (Ollama, vLLM, LM Studio) and **hosted open-model gateways** (OpenRouter,
  Together, Groq). This is how you run Qwen, Llama, Mistral, and similar with no new code.
- **`family`** is a free label. Give sibling models the same family so the family-favoritism
  metric works.
- **`api_key_env`** names the environment variable holding that model's key (defaults to the
  provider's usual name). Local endpoints often need no key at all.
- You need **at least two** models, so a judge always has something other than itself to
  compare against.

### API keys

Keys live in a file named `.env` in the project root (never committed). Copy the template:

```bash
cp .env.example .env
```

Then fill in only the providers you use:

```
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
# a model can point at any variable name via api_key_env, e.g.:
OPENROUTER_API_KEY=sk-or-...
```

### Pricing

Cost estimates read `config/pricing.yaml` (dollars per million input/output tokens, keyed
by model string). Update it when prices change or when you add a model. Local models are
priced at $0; any model string starting with `mock` is treated as free.

---

## 6. Choosing your prompts

Set `prompts.source` in the config to one of three options.

**Built-in (default).** A curated prompt set across the three domains. Reproducible and
free. Nothing to configure.

```yaml
prompts:
  source: builtin
```

**Your own, via Excel.** Bring a spreadsheet of prompts.

```bash
uv run selfbias template prompts.xlsx        # writes a fillable template
uv run selfbias import-check prompts.xlsx    # checks it and reports what it found
```
```yaml
prompts:
  source: excel
  excel_path: prompts.xlsx
```

The sheet needs only two columns, **`prompt`** and **`domain`**. Optional columns unlock
more: `reference` (`ensemble` or `programmatic`), `rubrics` (`;`-separated criteria; add
`| negative` after one to flip its polarity), and `constraint` (`;`-separated
`checker:value` for the objective arm, e.g. `must_include:firmware; max_words:120`). See
`data/tasks/seeds/example_prompts.csv` for a worked example.

**Model-drafted.** A model on your roster writes a fresh prompt set at run time.

```yaml
prompts:
  source: llm_generated
  generator_model: claude-opus-4-8   # optional; defaults to the first roster model
  n_per_domain: 20
```

This makes one small, cost-guarded model call per domain. It is deterministic (same seed
gives the same prompts), so re-running is free from cache.

---

## 7. The workflow

You can drive everything from the dashboard or the command line. The steps are the same.

1. **Configure.** Pick models, prompts, lengths, and a spending cap. In the dashboard, the
   **Configure** page validates as you go and previews the cost. Or copy
   `config/experiment.example.yaml` to `config/runs/mine.yaml` and edit it.
2. **Check your keys.** `selfbias check config/runs/mine.yaml` makes one tiny real call per
   model to confirm the keys and endpoints work before you spend anything.
3. **Estimate the cost.** `selfbias estimate config/runs/mine.yaml` prints the number of
   calls and the projected dollar cost per stage. No API calls are made.
4. **Run.** `selfbias run config/runs/mine.yaml` shows the estimate, asks you to confirm,
   then executes: writing, judging, and recognition probes. It is **cache-first** (an
   identical re-run costs about $0) and **budget-capped** (it pauses itself before crossing
   your cap).
5. **Resume if needed.** If a run paused (budget) or was interrupted, `selfbias resume
   config/runs/mine.yaml` continues where it left off. Completed work is skipped.
6. **Analyze.** `selfbias analyze config/runs/mine.yaml` computes the curves, the onset
   lengths, and the regression, and saves a report under `data/reports/`.
7. **Explore.** Open the dashboard; your analyzed run appears at the bottom of **Results**.

---

## 8. Cost and safety

SelfBias is built so you never spend by surprise.

- **Every run starts with an estimate** you must confirm. `estimate` alone makes no calls.
- **`budget_usd` in the config is a hard cap.** The run halts (resumably) before it would
  cross it. You set it; you own it.
- **Runs are cache-first.** Identical requests are served from disk, so re-running an
  experiment, or resuming one, is nearly free.
- **Nothing launches implicitly.** The dashboard's Run action requires you to type the run
  name to confirm.
- **Local and mock runs cost $0.** Use them freely to learn the tool and shake out a config.

---

## 9. Command reference

Run these as `uv run selfbias <command>` (or via `./run.sh <shortcut>` for the common ones).

| Command | What it does |
|---|---|
| `check <config>` | One tiny real call per model to confirm keys and endpoints work. |
| `estimate <config>` | Dry-run cost by stage and provider. No API calls. |
| `run <config>` | Estimate, confirm, then execute the pipeline. Cache-first, budget-capped. |
| `resume <config>` | Continue a paused or interrupted run. |
| `status <config>` | Show a run's progress, spend, and cache hits. |
| `analyze <config>` | Compute curves, onset L\*, and the regression; save a report. |
| `template <path>` | Write a starter Excel/CSV prompt library. |
| `import-check <path>` | Validate a prompt library and summarize what it parses to. |
| `export <config> <out>` | Export a run's rows to Excel (or CSV). |

`./run.sh` shortcuts: `dashboard` (default), `demo`, `estimate`, `check`, `status`,
`analyze`, `test`.

---

## 10. The dashboard

Launch it with `uv run streamlit run dashboard/app.py` (or `./run.sh`). It follows your
system light/dark theme. Throughout, plain-language descriptions are shown, and the precise
technical detail sits behind the small **info icon** next to each heading; hover it to read.

- **Configure** - pick a starting config, adjust the basics (name, seed, spending cap,
  lengths, models), preview the cost, and save your own config to `config/runs/`.
- **Run** - see which models are ready, test your keys, review the estimate, and start the
  run with a typed confirmation. Progress and final spend show live.
- **Results** - example charts up top so the page is never empty, and your analyzed runs at
  the bottom. Every chart supports hover.

---

## 11. Reading the results

Here is how to interpret each part of the **Results** page.

- **Favors own writing (the top tiles).** The average HSPP-Ratio across judges. `1.00x`
  means fair on average; `1.5x` means judges over-credit their own writing about 50% more
  often than they over-credit strangers.
- **Length sweep (the main chart).** Three lines versus text length: how well models
  recognize their own writing, how guessable the author is, and how strong the bias is. If
  the bias line switches on at roughly the same length as the recognition line, that
  supports the "it recognizes itself" explanation; if bias appears earlier, that supports
  the "it just favors familiar text" explanation.
- **How much each judge favors itself (table).** HSPP-R per model, for self and family.
- **Who over-credits whom (heatmap).** Rows are judges, columns are writers; hotter cells
  mean more over-crediting. A bright diagonal is the self-preference signature.
- **Who gets bumped up or down (heatmap).** Warm cells are writers a judge scores above the
  crowd; cool cells are below. Self cells tend warm.
- **Can each model spot its own writing? (chart).** Recognition accuracy versus length, per
  model. The dashed line is a 50/50 guess.
- **By how the grading works / by kind of task (bars).** Bias is usually higher for
  head-to-head comparisons than checklists, and higher on taste-based tasks than
  verifiable ones.
- **Sanity checks (diagnostics).** Health checks on the experiment itself (does swapping the
  order of two pieces flip the winner, did models hit the target lengths, do judges agree).
- **Your analyzed runs (bottom).** The same charts from your real data, plus the **onset
  lengths** and the **mechanism regression**. In the regression table, an odds ratio above
  1 means that factor pushes toward favoritism; the confidence interval and p-value tell you
  how sure to be.

> On the offline mock model there is no real bias, so the onsets read "not yet" and the
> regression is not significant. That is correct. Signal appears with real models.

---

## 12. Where things are saved

Everything a run produces lives under `data/` (which is not committed, except the example
prompt seeds):

- `data/generations/` - the text each model wrote, with the full raw API responses.
- `data/judgments/` - every judge verdict.
- `data/probes/` - the recognition-quiz answers.
- `data/manifests/` - one file per run tracking progress and spend (this is what makes runs
  resumable).
- `data/cache/` - cached API responses, keyed by request, so re-runs are free.
- `data/reports/` - the analysis output the dashboard reads.

Because the raw responses are saved, all metrics can be recomputed from disk without
spending API budget again.

---

## 13. Provider cookbook

**Anthropic / OpenAI / Google (cloud).** Put the key in `.env` and use the native provider:

```yaml
- { slot: claude, provider: anthropic, model: claude-opus-4-8, family: anthropic }
- { slot: gpt,    provider: openai,    model: gpt-5,           family: openai }
- { slot: gemini, provider: google,    model: gemini-2.5-pro,  family: google }
```

**OpenRouter / Together / Groq (hosted open models).** Use `openai_compatible` with the
gateway's base URL and your key:

```yaml
- { slot: llama, provider: openai_compatible, model: "meta-llama/llama-3.1-70b-instruct",
    family: llama, base_url: "https://openrouter.ai/api/v1", api_key_env: OPENROUTER_API_KEY }
```

**Ollama / vLLM / LM Studio (local, no key).** Point at localhost; see
`config/experiment.ollama.yaml` for a complete, ready-to-run example:

```yaml
- { slot: qwen, provider: openai_compatible, model: "qwen2.5:7b", family: qwen,
    base_url: "http://localhost:11434/v1", api_key_env: OLLAMA_KEY }
```

Remember to add a `config/pricing.yaml` entry for any new cloud model so the cost estimate
is right (local models are already treated as free).

---

## 14. Troubleshooting

| Symptom | Fix |
|---|---|
| `command not found: uv` | Add uv to your PATH (§2), or call it by full path. |
| `command not found: selfbias` | Prefix with `uv run`, e.g. `uv run selfbias run ...`. |
| `check` says a model failed | The error text is shown verbatim. Usually a wrong or missing key, or an unreachable `base_url`. |
| A real run errors on the first call | Install the provider SDKs once: `uv sync --extra providers`. |
| "roster needs at least 2 models" | Add a second model; a judge needs something other than itself to compare against. |
| Onsets read "not yet" / regression not significant | Expected on the mock model, or on a run with too few prompts. Use real models and more prompts. |
| Cost estimate warns about unpriced models | Add the model to `config/pricing.yaml`. |
| Want a clean slate | Delete the generated run data. Keep `data/tasks/seeds/`, which holds the example prompt library. |

---

## 15. A note on scope

This version computes the length-sweep curves, onset lengths, attributability, and a first
version of the mechanism regression (logistic with prompt-clustered errors). The full
mixed-effects regression and the fingerprint-based attribution arm are planned next; see
`docs/ROADMAP.md` and `docs/DECISIONS.md`. For the formal metric definitions, see
`docs/METRICS.md`.
