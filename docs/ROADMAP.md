# ROADMAP.md: Phased Build & Execution Plan

Each phase ends with a gate: working demo + cost report + go/no-go on the next phase.

## Phase 0: Scaffold (no API spend)
- Repo skeleton per ARCHITECTURE.md; pydantic schemas; provider abstraction with mock
  provider; cache; manifests; cost estimator with pricing.yaml; CLI (`estimate`, `run`,
  `resume`, `status`); Streamlit shell with Configure + keyless demo Results (synthetic
  data); pytest suite green on mocks.
- Exit criteria: `selfbias estimate` produces a correct dry-run for the example config;
  full pipeline runs end-to-end on the mock provider; dashboard renders synthetic
  results.

## Phase 1: Pilot (~$150–400)
- Real keys in `.env`. ~60 prompts (20/domain), 3 length bins (100/500/1500), PWC only,
  anonymous only, pairwise recognition probe only, all 6 models.
- Purpose: validate schemas, length-compliance handling, cache behavior, cost model
  accuracy (estimate vs actual within ±25%), judge output parsing rates, repeatability
  sample.
- Deliverable: pilot report page in dashboard; first (noisy) HSPP-R table.

## Phase 2: Core experiment (~$800–2,500)
- ~300 prompts/domain × 3 domains; full length bins [25, 50, 100, 250, 500, 1000, 2500];
  PWC + rubric-based; all three disclosure arms; both recognition probes; truncation
  series; confidence elicitation throughout; TF-IDF attributability baseline per bin.
- Answers RQ1 (paradigm/domain/disclosure/family factors) and RQ2 (onset curves) with
  bootstrap CIs. Mechanism regression v1 (without fingerprint density).
- Deliverable: full Results dashboard; draft findings memo.

## Phase 3: Fingerprint arm + robustness (~$1,500–5,000 incremental)
- StoryScope-adapted fingerprint pipeline (discovery pool → features → corpus
  assignment → SHAP fingerprints → density scores); mechanism regression v2 with
  density (H3 test); style-neutralization arm (third-model LAMP-style rewrite of a
  stratified sample; re-judge + re-probe; does bias/recognition survive?); full
  bootstrap + multiple-comparison reporting.
- Deliverable: complete analysis, writeup-ready figures.

## Future milestones (designed-for, not scheduled)
- **1D Human anchor set:** 100–200 human-labeled items; `reference_source: human`
  already in schema.
- **2A Perplexity/familiarity arm:** open-weights proxy scoring; `familiarity_scores`
  field already reserved.
- **Direct assessment paradigm:** enum already includes `da`.
- **Deliberation/contagion study:** multi-judge panels with visible confidence
  (Li et al. follow-on); requires relaxing judge isolation as an explicit manipulation.
- **Roster expansion:** open-weights generator-only additions.
