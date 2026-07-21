# Research Plan: When and Why Do LLMs Self-Bias?

## 1. Research questions

- **RQ1 (mechanism):** What dictates when an LLM-as-judge exhibits self-preference bias?
- **RQ2 (threshold):** What is the minimum text length at which self-bias is determinable -
  when does an attributable "model dialect" emerge, and does bias track it?

## 2. Grounding literature (the four anchor papers)

1. **Wataoka, Takahashi & Ri (2025), "Self-Preference Bias in LLM-as-a-Judge."**
   Contributes the Equal-Opportunity bias metric (deviation from human/ground-truth
   preference conditioned on authorship) and the **familiarity hypothesis**: judges favor
   low-perplexity text regardless of authorship; self-outputs are simply low-perplexity
   to their author.
2. **Pombal, Rei & Martins (2026), "Self-Preference Bias in Rubric-Based Evaluation."**
   Contributes the **HSPP-Ratio** (overestimation-rate ratio, self vs unrelated models;
   plus a family variant), the finding that SPB persists even with programmatically
   verifiable rubrics, the paradigm comparison (pairwise > rubric/direct in bias), the
   ensemble-reference method for subjective tasks, and the factor analysis template
   (rubric polarity, rubric length, topic, inter-judge agreement).
3. **Russell et al. (2026), "StoryScope."** Contributes the attributability methodology:
   interpretable feature spaces + simple classifiers (XGBoost + SHAP) yield per-model
   narrative **fingerprints**; attribution survives style-removal; models converge in a
   shared region while humans are rarer/more dispersed. We repurpose fingerprints as an
   *explanatory variable* for bias, not just a detector.
4. **Li et al. (2025), "As Confidence Aligns" (CHI).** Contributes the calibration lens
   (confidence conditioned on authorship) and a design constraint: expressed confidence
   is contagious across agents, so judges must be isolated.

## 3. Hypotheses

- **H1 (recognition):** SPB magnitude increases with the judge's self-recognition
  accuracy on the same texts. Prediction: bias onset (over length) coincides with
  recognition onset; false authorship labels causally move bias.
- **H2 (familiarity):** SPB exists even where recognition is at chance (short texts,
  style-neutralized texts). Prediction: bias below the attributability floor; false
  labels move bias little relative to text-intrinsic factors.
- **H3 (fingerprint density):** SPB magnitude is better predicted by the density of the
  generator's fingerprint features present in the text than by raw token length.
- **H4 (threshold):** Attributability (classifier F1 and judge self-recognition) is at
  chance below some length L* and rises with length; L* differs by model and domain
  (creative < factual, hypothesized).
- **H5 (paradigm/factors, replication at frontier):** Pairwise > rubric-based in SPB;
  negative rubrics > positive; subjective domains > objective; family bias
  (judge favors sibling models) > unrelated-model baseline.

## 4. Design

### 4.1 Model roster (Decision 3B): 6 models, 3 families, frontier + small sibling

| Family | Frontier | Sibling |
|---|---|---|
| Anthropic | (config: `anthropic_frontier`) | (config: `anthropic_small`) |
| Google | (config: `google_frontier`) | (config: `google_small`) |
| OpenAI | (config: `openai_frontier`) | (config: `openai_small`) |

Concrete model strings live in config only (models change; code must not assume names).
All 6 generate; all 6 judge → 36 judge×generator cells (6 self, 6 within-family
non-self, 24 cross-family).

### 4.2 Domains (Decision 4B): objectivity as a manipulated axis

1. **Verifiable instruction-following** (objective arm): IFEval-style prompts with
   programmatically checkable constraints. Ground truth is computed, free, exact.
2. **Open QA / summarization** (semi-subjective): reference = ensemble majority.
3. **Creative writing** (fully subjective; maximal dialect signal): reference =
   ensemble majority; primary domain for RQ2 length analysis.

### 4.3 Ground truth (Decision 1C)

- Objective arm: programmatic verification per constraint (rubric-level ground truth).
- Subjective arms: majority vote over all 6 judges' verdicts as pseudo-reference
  (Pombal method). Known limitation: every judge is inside the reference, so bias is
  underestimated; the objective arm quantifies that underestimation and calibrates
  interpretation. **Future milestone (1D):** human-labeled anchor set (~100–200 items).

### 4.4 Length sweep (Decision 6B+C)

- **Controlled-length generation (primary, for bias):** target bins, e.g.
  `[25, 50, 100, 250, 500, 1000, 2500]` tokens (config), with a ±20% tolerance band and
  bounded retry; record realized token counts; realized length is the analysis variable.
- **Truncation series (supplementary, for attributability only):** the longest
  generation per (prompt, model) truncated to each bin. Used for recognition probes and
  classifier attributability, **never** for quality judging (truncation depresses
  quality non-uniformly).

### 4.5 Evaluation paradigms (Decision 5B)

- **Pairwise comparison (PWC):** both orderings, resolution rule in METRICS.md.
  Pairs are cross-generator on the same prompt at the same length bin.
- **Rubric-based (RB):** binary criteria per task; objective arm rubrics are the
  verifiable constraints themselves; subjective arms use authored rubric sets with
  logged polarity (positive/negative) and rubric length in tokens (both are factor
  variables, per Pombal).
- Direct assessment (DA) is out of scope (deferred; see DECISIONS.md).

### 4.6 Disclosure arms (Decision 2D): causal test of recognition

Per judging call: `anonymous` (default) | `true_label` (real generator named) |
`false_label` (a different roster model named, balanced assignment). Identical texts
across arms; only the prompt header differs.

### 4.7 Mechanism instrumentation (Decision 2B + 2C)

- **Recognition probes (2B):** separate calls; two formats:
  (a) pairwise "one of these two texts was written by you - which?" and
  (b) single-text "did you write this? yes/no + confidence".
  Run across length bins and truncation series → per-judge recognition curves.
- **Fingerprint pipeline (2C, Phase 3):** StoryScope-adapted:
  discovery pool (held out) → structured template extraction → cross-source comparative
  analysis → feature proposal (closed-form questions, 5 response types) → dedup →
  corpus-wide feature assignment → XGBoost + SHAP → per-model fingerprint features →
  **fingerprint-density score** per text = weighted presence of its generator's
  fingerprint features. Density is a regressor in the bias model.
- **Confidence & calibration (from Li et al.):** every judging and probe call elicits a
  numeric confidence; compute calibration conditioned on self/family/other.
- **Future milestone (2A):** perplexity via open-weights proxies (infra-heavy; deferred).

## 5. Analysis plan

1. **Bias matrices:** 6×6 judge×generator overestimation rates per domain × paradigm ×
   length bin; HSPP-R_self and HSPP-R_fam per judge; centered score-delta matrices
   (Pombal Fig. 5 style) for practical skew.
2. **Objective-arm anchor:** EO-Bias and HSPP-R against programmatic ground truth;
   compare against ensemble-reference values on the same arm to estimate
   reference-contamination shrinkage.
3. **Length curves (RQ2):** per judge: HSPP-R vs realized length; recognition accuracy
   vs length; classifier attribution F1 vs length (train per-bin on held-out split).
   Estimate onset points (first bin where CI excludes chance/1.0). **Key comparison:**
   bias onset vs attributability onset (H1 vs H2).
4. **Causal recognition test:** bias(anonymous) vs bias(true_label) vs bias(false_label).
   If false labels transfer bias to the falsely named model → recognition is causal.
5. **Mechanism regression (H3):** mixed-effects logistic regression of per-comparison
   overestimation on: fingerprint density, realized length (log), self/family/other,
   disclosure arm, paradigm, domain, rubric polarity & length (RB only), with random
   intercepts for prompt and judge.
6. **Calibration:** reliability diagrams + ECE conditioned on authorship relation.
7. **Statistics:** prompt-level bootstrap CIs (B ≥ 1000) on all headline numbers;
   Holm–Bonferroni within hypothesis families; all seeds logged.

## 6. Known limitations (state in any writeup)

- Ensemble reference underestimates bias (mitigated, not solved, by the objective arm).
- Three families only; "unrelated models" pools are small (24 cross-family cells).
- No perplexity access on closed models until milestone 2A.
- Controlled-length instructions may themselves shift style at extreme short bins.
