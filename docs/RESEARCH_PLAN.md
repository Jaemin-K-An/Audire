# AUDIRE Research Plan

## Working title
**Personalized Prediction of Korean Word Misrecognition for Selective Captioning**

Korean: **개인 어음인지 프로파일 기반 한국어 오청 예측 및 선택 자막**

## Core claim to test
A user's speech-perception profile — especially an **individual phoneme confusion matrix** — should predict word-level Korean mishearing risk better than an audiogram-only or clinical-score-only model, and this risk can drive captions that preserve high-risk information while displaying substantially less text than full captions.

## Research questions
- **RQ1**: Does adding individual phoneme confusion information to PTA/SRT/WRS improve word-level mishearing prediction over PTA-only and PTA+SRT+WRS baselines?
- **RQ2**: At a fixed caption budget, does personalized risk ranking capture more true misheard words than non-personalized strategies?
- **RQ3**: Does a WRS-informed or validation-optimized personal threshold improve the accuracy–caption-volume tradeoff over a global threshold?
- **RQ4 (secondary)**: How robust is the model under changes in SNR, speaker and word/phoneme position?

## Hypotheses
- H1: `clinical + confusion` yields lower Brier score / higher PR-AUC than `audiogram only` and `clinical only`.
- H2: Personalized selective captions achieve higher misheard-word recall at matched caption ratios.
- H3: Calibration quality degrades as the phoneme test is shortened; a principled subset-selection method retains more predictive information than random shortening.

## Profile definition
For listener `u`:

`U = {audiogram_250..8000, PTA, SRT, WRS, optional_PBmax, C_u}`

`C_u(i,j) = P(perceived phoneme=j | target phoneme=i, listener=u)`

WRS is a **global speech-recognition factor**; the confusion matrix is a **local error-structure factor**. Do not collapse them into one number.

## Primary outcome
`P(mishear word w | listener u, acoustic context e)`

The initial interpretable model family must include:
1. deterministic phoneme-risk baseline
2. logistic regression / generalized linear probabilistic model
3. at least one nonlinear baseline if data volume supports it
4. probability calibration comparison (Platt/isotonic or equivalent when appropriate)

Deep learning is not a requirement and must not be used merely to claim AI usage.

## Word-risk baseline
For phonemes `phi_k` in word `w`:

`R_phon(w,u) = 1 - product_k C_u(phi_k, phi_k)`

This independence baseline is deliberately simple and must be tested against learned alternatives. Position (onset/nucleus/coda), repetition and phonological class should be explicit features.

## Selective-caption objective
Two equivalent evaluation forms:

1. **Budget form**: maximize true misheard-word recall subject to `caption_ratio <= B`.
2. **Utility form**: minimize `alpha * misunderstanding_loss + beta * caption_ratio`.

Report a Pareto curve rather than one cherry-picked threshold.

## Baselines
- B0: full captions
- B1: random words at matched caption ratio
- B2: lexical/content heuristic (non-personalized)
- B3: audiogram/PTA-only risk
- B4: PTA + SRT + WRS
- **AUDIRE**: B4 + individualized confusion features

## Evaluation metrics
Prediction:
- PR-AUC (primary for imbalanced errors)
- ROC-AUC
- Brier score
- log loss
- Expected Calibration Error
- sensitivity/recall, specificity, precision, F1

Caption utility:
- misheard-word recall at 10/20/30/40/50% caption budgets
- Caption Reduction Ratio
- weighted information recall if semantic importance is later added
- end-to-end ASR + risk pipeline failure rate

Subgroup/sensitivity reporting:
- hearing-level group
- onset/nucleus/coda
- SNR condition
- speaker
- WRS bands

## Simulation plan
Simulation is for engineering/research power analysis and must be labeled synthetic.

### Synthetic listener generator
- choose hearing severity stratum
- sample plausible audiogram shape
- sample SRT/WRS conditional on severity using transparent priors or an auxiliary open audiology dataset
- generate a confusion matrix by perturbing a literature-informed/group prior with Dirichlet noise
- preserve Korean phonological constraints (e.g., stronger within-class confusions) only when supported by cited sources

### Synthetic trial generator
1. choose target Korean word/sentence
2. decompose Hangul into onset/nucleus/coda jamo
3. draw perceived phonemes from `C_u`
4. create observed mishearing label at word level
5. optionally inject SNR/speaker effects through predeclared parameters

### Simulation sweeps
- N listeners: 20, 40, 80, 160
- calibration lengths: 10, 25, 50, 100, 200+ stimuli
- caption budget: 10–50%
- WRS strata
- SNR levels
- missing-clinical-variable scenarios

Run >= 100 deterministic Monte Carlo seeds per configuration when computationally feasible. Publish confidence intervals, not only means.

## Human/expert validation
A 10-year hearing-aid professional with audiologist/hearing-specialist credentials may provide structured expert review.

Expert review validates:
- clinical variable selection
- calibration protocol feasibility
- plausibility of confusion/risk explanations
- UI safety and interpretation

One expert is **not** a consensus panel. Treat the result as structured expert validation/qualitative evidence, not population-level clinical validation.

If human listeners are recruited, obtain the required school/institutional ethics/consent approvals before data collection. Keep all participant-level data private.

## Key falsification criteria
AUDIRE should be considered unsupported if:
- confusion features do not improve held-out prediction beyond clinical baselines;
- selective captions fail to outperform random/non-personalized selection at matched caption budgets;
- probabilities are badly miscalibrated and cannot be corrected;
- improvements disappear under listener-level cross-validation.

Negative results must remain in the final report.
