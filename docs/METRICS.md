# METRICS.md: Formal Definitions (implement exactly; propose changes via DECISIONS.md)

Notation: judge J, generator G, prompt/instance x ∈ D, rubric k. Roster partition for a
judge J: {J} ∪ F_J (same-family, excl. J) ∪ S_J (unrelated families).

## 1. Judge outcomes

### 1.1 Pairwise (PWC)
Each unordered pair (G, G′, x, length-bin) is judged twice (order-swapped).
Resolution: if one run yields a winner and the other a tie → the winner stands;
if runs disagree on the winner, or both tie → **tie**. Judge outcome
w_J(G, G′, x) ∈ {+1, 0, −1} (G wins / tie / G loses).

### 1.2 Rubric-based (RB)
Per-rubric binary verdict b_J(G, x, k) ∈ {−1, +1} (unsatisfied / satisfied).
Instance score s_J(G, x) = fraction of satisfied rubrics (weighted if config defines
weights; judge never sees weights). Derived pairwise outcome:
w_J(G, G′, x) = sgn(s_J(G, x) − s_J(G′, x)).

### 1.3 Reference outcomes
Objective arm: b*(G, x, k) from programmatic verification; s*, w* derived identically.
Subjective arms: b*(G, x, k) = majority vote of all 6 judges on that rubric
(ties broken toward "unsatisfied" - conservative); w* derived from s*.
PWC subjective reference: majority over judges' resolved pairwise outcomes.

## 2. Overestimation rates (Pombal)

### 2.1 Instance-level
O_inst(J, G) = P over (G′, x) with w*(G, G′, x) = −1 that w_J(G, G′, x) > w*(G, G′, x).
("Among comparisons G should lose, how often does J rule more favorably than warranted?")

### 2.2 Rubric-level
O_rub(J, G) = P over (x, k) with b*(G, x, k) = −1 that b_J(G, x, k) = +1.
("Among rubrics G objectively fails, how often does J mark them passed?")

## 3. Headline bias metrics

### 3.1 HSPP-Ratio (self) : primary headline metric
HSPP-R_self(J) = O(J, J) / mean_{G ∈ S_J} O(J, G)
Computed at both instance and rubric level. 1.0 = no self-preference; >1 = self-bias.

### 3.2 HSPP-Ratio (family)
HSPP-R_fam(J) = mean_{G ∈ F_J} O(J, G) / mean_{G ∈ S_J} O(J, G)

### 3.3 Equal-Opportunity Bias (Wataoka): objective arm and anchor sets only
EO-Bias(J) = P(J prefers own | reference prefers own) − P(J prefers own | reference
prefers other). Range [−1, 1]; 0 = unbiased; negative = self-deprecation.

### 3.4 Centered score-delta matrix (practical skew)
Δ(J, G) = [score_J(G) − score*(G)] − mean_G [score_J(G) − score*(G)], where score is the
system-level mean instance score. Reported as the 6×6 heatmap.

## 4. Judge accuracy (context metrics)

- **MIPA** (Mean Instance Pairwise Accuracy): fraction of (G, G′, x) with
  w_J = w* (ties count as agreement only if both are ties).
- **MRA** (Mean Rubric Accuracy): fraction of (G, x, k) with b_J = b*.
- Objective arm only, unless an anchor set exists (else circular).

## 5. RQ2 metrics

### 5.1 Attributability (classifier)
Per length bin: 6-way attribution macro-F1 and binary per-model one-vs-rest AUPRC from
an XGBoost classifier on fingerprint features (Phase 3) and on a TF-IDF baseline
(Phase 2, so length curves exist before the fingerprint pipeline). Prompt-level
train/test grouping; report bootstrap CIs. Chance = 1/6 (16.7%) macro-F1.

### 5.2 Self-recognition accuracy
Pairwise probe: accuracy vs 50% chance, per judge per length bin, on both
controlled-length and truncation series. Single-text probe: AUROC of "yes" confidence
against true authorship.

### 5.3 Onset estimation
Onset length L* for a curve = smallest bin whose bootstrap 95% CI excludes the null
(chance for recognition/attribution; 1.0 for HSPP-R), with all larger bins also
excluding it (monotone-onset rule). Report per judge and pooled.

### 5.4 Fingerprint density (Phase 3)
For text t by generator G: density(t) = Σ_{f ∈ FP_G} |SHAP_f| · 1{feature f active in t
in G's characteristic direction} / Σ_{f ∈ FP_G} |SHAP_f|, where FP_G is G's fingerprint
feature set. Range [0, 1].

## 6. Calibration

Confidence c ∈ [0, 1] elicited per verdict. Per authorship relation
r ∈ {self, family, other}: reliability diagram, Expected Calibration Error (15 bins),
and over/under-confidence index (mean confidence − accuracy). Bias signature of
interest: ECE_self vs ECE_other and sign of the confidence gap on incorrect verdicts.

## 7. Agreement / diagnostics

- Inter-judge pairwise agreement per rubric (Cohen's κ pooled) - used to replicate
  Pombal's finding that filtering low-agreement rubrics lowers HSPP-R.
- Judge repeatability: repeated identical calls (n=5 on a 2% sample), Krippendorff's α.
- Position-bias rate: fraction of PWC pairs where order swap flips the winner.
- Length-instruction compliance: |realized − target| / target distribution per model.

## 8. Statistical reporting rules

- Every headline number: prompt-level bootstrap, B ≥ 1000, 95% CI.
- Regressions (mechanism model in RESEARCH_PLAN §5.5): mixed-effects logistic,
  random intercepts for prompt and judge; report odds ratios with CIs.
- Multiple comparisons: Holm–Bonferroni within each hypothesis family (H1–H5).
- All randomness seeded from config; seeds recorded in run manifests.
