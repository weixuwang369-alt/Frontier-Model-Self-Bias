# DECISIONS.md: Design Decision Log

Status values: **LOCKED** (implement as stated) | **FUTURE** (design for, do not build
until scheduled) | **OPEN** (needs a decision before the affected code is written).

## D1: Ground truth strategy - LOCKED: two-arm (1C), human anchor FUTURE (1D)
Objective arm (programmatically verifiable instruction-following) anchors metrics with
exact ground truth; subjective arms use 6-judge ensemble majority as pseudo-reference.
The objective arm doubles as a measurement of how much ensemble references
underestimate bias. **FUTURE 1D:** human-labeled anchor set (~100–200 items) as an
external gold reference; schemas must already support `reference_source: human`.

## D2: Mechanism instrumentation - LOCKED: recognition probes (2B) + fingerprint
pipeline (2C) + disclosure manipulation (2D); perplexity arm FUTURE (2A)
2B and 2D are cheap and causal; 2C is the differentiating contribution (Phase 3).
**FUTURE 2A:** perplexity/familiarity via open-weights proxy models - deferred because
it requires local/hosted inference infra and its proxy validity is debatable. Schemas
must already carry an optional `familiarity_scores` field per text.

## D3: Model roster - LOCKED: 6 models (3B)
Frontier + small sibling per family (Anthropic, Google, OpenAI). Unlocks HSPP-R_fam and
stabilizes ensemble references and HSPP denominators. Model strings are config-only.

## D4: Task domains - LOCKED: 3 domains (4B)
Verifiable instruction-following / open QA & summarization / creative writing.
Objectivity is a manipulated factor; creative writing is the primary RQ2 domain.

## D5: Evaluation paradigms - LOCKED: PWC + rubric-based (5B); DA FUTURE
Direct assessment deferred (noisiest, least marginal information). Paradigm enum in
schemas includes `da` so adding it later is config + prompt template only.

## D6: Length design - LOCKED: controlled-length generation + truncation series (6B+C)
Controlled bins for bias measurement; truncation of the longest generation for
attributability/recognition only. Truncated texts are never quality-judged.

## D7: Scale - LOCKED: architect for Tier 3, execute phased (see ROADMAP.md)
Phase gates with dry-run cost estimates and hard budget caps.

## D8: UI - LOCKED: Streamlit
Python-native, shares the pipeline codebase, form-based config editing, adequate for an
internal research dashboard. No second frontend stack.

## D9: Storage - LOCKED: JSONL (append-only, raw responses persisted) + DuckDB for
analysis queries. No external database.

## Cross-cutting integrity requirements: LOCKED
Order-swapped PWC; judges isolated from each other's outputs; length as covariate in
all bias models; leniency-normalized headline metrics; anonymized generator identity
except in disclosure arms; temperature-0/fixed-seed judging; cache-first API layer;
recognition probes as separate calls.

## Future milestones (roadmap tail)
- 1D human anchor set; 2A perplexity arm; DA paradigm; style-neutralization arm
  (LAMP-style third-model rewrite, tests whether bias survives style removal - currently
  scheduled inside Phase 3, may be split out); multi-judge deliberation / confidence
  contagion study (Li et al. follow-on); additional families (open-weights generators).

## Open items (deviations from METRICS to revisit)
- **Mechanism regression v1 (2026-07-17):** METRICS §8 specifies a mixed-effects logistic
  regression with random intercepts for prompt and judge. The shipped v1
  (`analysis/regression.py`) is a **fixed-effects logistic with prompt-clustered robust
  SEs** - it accounts for within-prompt correlation and yields interpretable odds ratios
  without a GLMM fit, which is fragile on sparse data. Upgrade to the full random-intercepts
  model (statsmodels `BinomialBayesMixedGLM` or an equivalent) in Phase 2, and add the
  fingerprint-density regressor (H3) in Phase 3. The regression currently uses rubric-level
  overestimation (RB) only; adding PWC-derived per-comparison outcomes (with the `paradigm`
  factor) is a Phase 2 item.

## Resolved items
- **Model strings for the 6 roster slots - RESOLVED (2026-07-16, Phase 0 kickoff).**
  Config-only; code never assumes these. Change in `config/experiment.example.yaml`
  or a `config/runs/*.yaml` without touching code.

  | Slot | Provider | Model string |
  |---|---|---|
  | anthropic_frontier | anthropic | `claude-opus-4-8` |
  | anthropic_small    | anthropic | `claude-haiku-4-5-20251001` |
  | google_frontier    | google    | `gemini-2.5-pro` |
  | google_small       | google    | `gemini-2.5-flash` |
  | openai_frontier    | openai    | `gpt-5` |
  | openai_small       | openai    | `gpt-5-mini` |

- **Rubric authoring source - RESOLVED (2026-07-16): LLM-drafted, human-reviewed,
  polarity-labeled** (the documented default). Config value
  `source: llm_drafted_human_reviewed` stands for the subjective domains.

## Open items
- Creative-writing prompt source: original prompt set (avoids copyright and
  memorization) vs adapting a public prompt corpus. Default: original, generated at
  Phase 1, reviewed for diversity.

## Schema migrations
- **2026-07-17 (Phase 0.5, W2):** `StageCost` gained `parse_failures` (default 0) -
  counts structured judge/probe calls whose output couldn't be parsed/validated (the
  call was made and billed; the row is skipped, never coerced to a default verdict).
  Older manifests without the field read back as 0.
- **2026-07-17 (Phase 0.5):** model **`family` is now a free-form string**, not the
  fixed {anthropic, google, openai} enum - the roster is open-ended (any provider, any
  model incl. Qwen/Llama). `RosterModel` gained optional `base_url` + `api_key_env`;
  `Provider` gained `openai_compatible`. New `prompts` config block
  (`source: builtin | excel | llm_generated`). Rosters now require ≥2 models. No
  persisted-data migration (Phase 0 rows already used the same family string values).
- **2026-07-17 (Phase 0):** `Probe` gained a required `series ∈ {controlled, truncation}`
  field. Rationale: RQ2 recognition (METRICS §5.2) is reported per length series, so the
  series must be explicit on the row rather than re-derived from the seed. Truncation
  probes are now *skipped* when a bin has no truncated text (bin ≥ source length) instead
  of falling back to controlled text, which had produced ambiguous rows. No pre-existing
  data to migrate (Phase 0). Append-only rows written before this note lack the field and
  must be treated as `controlled` if ever re-read.
