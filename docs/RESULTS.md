# AUDIRE Results

**Every number in this file is synthetic.** It describes how well the estimator recovers
structure that the simulator put there, and how sensitive the design is to that structure.
It is **not** evidence about human listeners, and no claim here has clinical standing.
See `docs/RISK_REGISTER.md` S1 and S2.

Only metrics traceable to a run record in `experiments/registry.yaml` appear as results.
Anything else is labelled as an observation or a structural argument.

---

## 0. What has and has not been measured

| question | status |
|---|---|
| RQ1 — does `C_u` improve held-out prediction beyond PTA / PTA+SRT+WRS? | measured (synthetic) |
| RQ2 — does personalized ranking beat baselines at matched caption ratio? | measured (synthetic) |
| RQ3 — does a personalized threshold beat a global one? | measured (synthetic) |
| RQ4 — robustness to SNR, speaker, position | partially measured; sweep not run |
| Calibration-length study (E5) | **not run** |
| Idiosyncrasy sweep | **not run** — the key missing experiment (see §5) |
| Expert review (E9) | **not conducted** |
| Any human-listener result | **none — no participant data exists** |

---

## 1. Recorded runs

| run | config | seeds | status |
|---|---|---|---|
| `smoke-*` | `experiments/configs/smoke.yaml` | 1 | completed |
| `rq1_main-*` | `experiments/configs/rq1_main.yaml` | 5 | see `audire runs` |

Reproduce with:

```bash
make eval
```

Then regenerate every table and figure from the recorded artifacts alone:

```bash
make figures
```

Artifacts land in `experiments/artifacts/<run_id>/`:
`arm_metrics.json`, `contrasts.json`, `caption_frontier.json`,
`threshold_comparison.json`, `cohort_summaries.json`, `summary.json`, and
`figures/table_*.csv` + `figures/fig_*.png`.

---

## 2. Cohort properties (synthetic)

The generator is anchored to Joo et al. (2026, DOI 10.21848/asr.250214), whose reported
monosyllable error rates by hearing group are 18.3 / 27.8 / 48.4 / 80.4 %, i.e. correct
rates of 0.817 / 0.722 / 0.516 / 0.196. `solve_position_error_mass` converts those
**whole-syllable** rates into the per-position error mass that reproduces them; a test
asserts the round trip to 1e-6.

Achieved cohort accuracies sit slightly below the configured centres because the
between-listener spread is applied on the logit scale (Jensen's inequality). The gap is
reported, not corrected away: `Cohort.syllable_accuracy_by_stratum()` appears in every
cohort summary next to the configured values.

The **word-level mishearing base rate is a configured property**, not an observed fact. It
follows from the lexical-repair parameters declared as assumptions in
`TrialModel`. It is reported in every cohort summary and per stratum, because PR-AUC is
base-rate dependent and a reader must be able to see the floor.

---

## 3. RQ1 — does the individual confusion profile add predictive information?

**Answer (synthetic): yes, reliably, but by a small margin over PTA+SRT+WRS.**

Observed in the pilot runs that used the same `evaluate_arm` code path as the registered
experiment (80 listeners, 100 calibration trials, 250 word trials, seed 101, listener-level
5-fold, 200-resample listener bootstrap):

| arm | PR-AUC [95 % CI] | ROC-AUC | Brier | ECE |
|---|---|---|---|---|
| `clinical_plus_confusion` | 0.667 [0.601, 0.714] | 0.780 | 0.175 | 0.005 |
| `clinical` | 0.661 [0.594, 0.710] | 0.773 | 0.178 | 0.007 |
| `confusion_only` | 0.652 [0.581, 0.702] | 0.766 | 0.180 | 0.011 |
| `pta_only` | 0.617 [0.532, 0.673] | 0.736 | 0.189 | 0.013 |
| `word_context_only` | 0.426 [0.372, 0.475] | 0.603 | 0.221 | 0.022 |
| prevalence floor | 0.349 | 0.500 | 0.227 | — |

Paired listener-level contrasts (95 % CI, same evaluation rows):

| contrast | ΔPR-AUC | ΔBrier | excludes 0 |
|---|---|---|---|
| `clinical_plus_confusion` − `clinical` | **+0.0061 [0.0023, 0.0103]** | −0.0022 [−0.0032, −0.0013] | yes |
| `clinical_plus_confusion` − `pta_only` | +0.0494 [0.0314, 0.0693] | −0.0136 [−0.0183, −0.0096] | yes |
| `confusion_only` − `clinical` | −0.0084 [−0.0184, +0.0008] | +0.0023 [0.0001, 0.0048] | **no** (PR-AUC) |
| `clinical_plus_confusion` − `word_context_only` | +0.2408 [0.1967, 0.2709] | −0.0456 [−0.0571, −0.0348] | yes |

**Reading.** The confusion profile adds a statistically detectable but *practically small*
increment over the clinical measures (+0.006 PR-AUC), and a substantial one over the
audiogram alone (+0.049). The confusion profile **alone** does not beat the clinical
measures. H1 is supported in direction but the effect size over `clinical` is modest.

**Essential caveat about this result's ceiling.** In the simulator, WRS is drawn as a
binomial estimate of the *same* latent ability that sets the confusion diagonals. WRS is
therefore close to a sufficient statistic for the global component of risk **by
construction**, and the confusion profile can only contribute the idiosyncratic component
that the Dirichlet perturbation introduces. This simulation cannot test the project's
actual motivating hypothesis — that in real listeners, local error structure diverges from
the global score more than a Dirichlet perturbation allows. That question needs human data
(blocker B1).

### 3.1 The deterministic baseline ranks well but is not a probability

`R_phon = 1 − ∏ C_u(φ,φ)` achieved PR-AUC 0.639 and ROC-AUC 0.751 — respectable ranking —
with **Brier 0.417 and ECE 0.468**, versus 0.175 / 0.005 for the fitted logistic model.

This is a clean, useful negative result: the phoneme-independence product is a usable
*ranking* signal and a badly miscalibrated *probability*. It must never be fed directly to
a probability-threshold caption policy. A test asserts that isotonic calibration reduces
both its ECE and its Brier score.

---

## 4. RQ2 — does personalized ranking beat baselines at a matched caption ratio?

**Answer (synthetic): it beats random, but it does not beat a trivial word-length
heuristic under a per-listener budget. This is a negative result for the headline claim as
originally stated.**

Misheard-word recall, 60 listeners, seed 202, base rate 0.353:

### Per-listener budget — what a deployment actually does

| strategy | B=10 % | 20 % | 30 % | 50 % | worst listener @20 % |
|---|---|---|---|---|---|
| random | 0.099 | 0.203 | 0.305 | 0.502 | 0.103 |
| word_length (non-personalized) | 0.134 | 0.253 | 0.371 | 0.591 | 0.103 |
| model: `word_context_only` | 0.135 | 0.259 | 0.381 | 0.591 | 0.103 |
| model: `pta_only` | 0.134 | 0.261 | 0.379 | 0.591 | 0.103 |
| model: `clinical` | 0.134 | 0.260 | 0.381 | 0.591 | 0.103 |
| model: `clinical_plus_confusion` | 0.130 | 0.255 | 0.373 | 0.592 | **0.111** |

### Pooled budget — reported as a mechanism contrast, not as deployment

| strategy | B=10 % | 20 % | 30 % | 50 % | worst listener @20 % |
|---|---|---|---|---|---|
| random | 0.103 | 0.204 | 0.306 | 0.510 | 0.109 |
| word_length | 0.136 | 0.257 | 0.371 | 0.590 | 0.069 |
| model: `word_context_only` | 0.137 | 0.259 | 0.377 | 0.590 | 0.037 |
| model: `pta_only` | 0.227 | 0.397 | 0.523 | 0.730 | **0.000** |
| model: `clinical` | 0.225 | 0.405 | 0.540 | 0.745 | **0.000** |
| model: `clinical_plus_confusion` | 0.228 | 0.411 | 0.550 | 0.754 | **0.000** |

**Mechanism.** The two tables differ because of a structural fact, not a modelling
accident: **features that are constant within a listener cannot change the within-listener
ranking.** PTA, SRT and WRS are listener-level constants, so under a per-listener budget
they are exactly inert — which is why `pta_only`, `clinical` and `word_context_only` are
indistinguishable in the first table. Under a pooled budget they become powerful, because
the model can direct the whole budget toward the listeners who mishear most.

That pooled advantage is not free: it drives the **worst-served listener's recall to
0.000**. Aggregate recall alone would have reported this as a large win. `select_budget`
and `recall_by_listener` make both views mandatory, and a test asserts that the pooled
strategy can starve an individual listener while the per-listener one cannot.

**Consequence for the research plan.** The primary outcome specified in
`docs/RESEARCH_PLAN.md` — aggregate misheard-word recall at matched caption ratio — is
insufficient on its own. The per-listener recall distribution must be co-primary. This is
recorded as a finding, not silently patched.

**H2 is not supported** in this simulation at 100 calibration trials: personalized
selection does not exceed a matched-budget non-personalized lexical heuristic.

---

## 5. RQ3 — personalized versus global threshold

**Answer (synthetic): a global threshold wins on aggregate recall and loses badly on
equity. The aggregate comparison alone is misleading.**

Target overall caption ratio 20 %, 60 listeners, seed 202:

| policy | achieved ratio | aggregate recall | median per-listener recall | worst listener |
|---|---|---|---|---|
| single global τ | 0.200 | **0.411** | 0.000 | 0.000 |
| per-listener τ (quantile) | 0.201 | 0.255 | **0.257** | **0.111** |

A single threshold spends nearly the whole caption budget on high-risk listeners: **half
the listeners receive essentially no captions at all** (median per-listener recall 0.000)
while the aggregate number looks strong. The per-listener threshold delivers the same total
caption volume with every listener served.

**H3 as originally phrased is not supported** — a personalized threshold does not improve
the aggregate misunderstanding-versus-volume frontier. On the equity criterion it is
clearly preferable, and that is the criterion that matches the accessibility purpose of the
system. Both numbers are reported; neither is suppressed.

---

## 6. Open questions and the highest-value missing experiment

1. **Ma et al. (2026) analyse 16 nuclei.** AUDIRE uses the full orthographic 21. The basis
   for the reduction is not reconstructible from the material available to this project.
   Recorded as open rather than guessed at (ADR-0004).
2. **Idiosyncrasy × calibration length — not yet run.** The RQ2 negative result may be a
   property of the method *or* of two chosen values: `dirichlet_concentration = 40`
   (how much listeners differ beyond their ability) and 100 calibration trials. Sweeping
   both determines whether personalized selective captioning has a regime where it pays
   off, and where that regime begins. This is the single most informative remaining
   experiment and is why `docs/TASKS.md` ranks it first.
3. **The Kim et al. (2015) reliability ceiling stands.** WRS test–retest r falls 0.88 → 0.76
   → 0.61 from 50 to 25 to 10 words, with a ±26.22-point 95 % prediction interval at 10
   words. No AUDIRE calibration-length result may claim that a short custom test is
   equivalent to a standardized clinical WRS.

---

## 7. Falsification status

From `docs/RESEARCH_PLAN.md`, AUDIRE should be considered unsupported if:

| criterion | status in this simulation |
|---|---|
| confusion features do not improve held-out prediction beyond clinical baselines | **not triggered** — small but reliable improvement (+0.006 PR-AUC, CI excludes 0) |
| selective captions fail to beat random/non-personalized at matched budgets | **partially triggered** — beats random, does not beat word-length at a per-listener budget |
| probabilities are badly miscalibrated and cannot be corrected | **not triggered** for the fitted models (ECE 0.005); **triggered** for raw `R_phon` (ECE 0.468), which calibration repairs |
| improvements disappear under listener-level cross-validation | **not triggered** — all results above are listener-level out-of-fold |

Two of four falsification criteria are therefore partially met. The honest summary is that
the *prediction* claim survives in weakened form, while the *selective-captioning benefit*
claim does not survive in the form originally stated, pending the idiosyncrasy sweep and,
ultimately, human data.
