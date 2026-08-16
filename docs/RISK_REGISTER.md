# AUDIRE Risk Register

Scientific, legal, engineering and ethical risks. Each entry names the risk, its impact, the
mitigation actually implemented (with a pointer to code or a test), and the residual risk that
survives mitigation and must therefore appear in the limitations section of any report.

Severity: **H** = would invalidate a headline claim or breach a legal/ethical obligation,
**M** = would materially weaken a claim, **L** = quality/usability.

---

## S1 — No participant-level Korean confusion data is available to this project · **H**
The 72-participant response matrices behind Joo et al. (2026) and Ma et al. (2026) are not
public. Every quantitative model in this repository is therefore fitted to **simulated**
listeners.

*Mitigation.* Two separate paths: (a) a fully synthetic research path used for power,
sensitivity and model-recovery analysis; (b) a real calibration workflow that collects an
individual's responses locally and builds their profile without any public participant data.
`is_synthetic` is a required field (ADR-0009) and is propagated into every result artifact.

*Residual.* **No result in this repository is clinical evidence.** RQ1–RQ3 are answered here
*for the simulator's data-generating process*, which is an engineering and design-sensitivity
result, not a finding about human listeners. This sentence must appear in the report.

---

## S2 — Simulator assumptions could manufacture the hypothesis they are meant to test · **H**
If synthetic word-mishearing labels are generated from the same phoneme-independence rule the
model uses, "confusion features help" is true by construction.

*Mitigation.* The generative process and the scoring model are deliberately different: the
simulator samples perceived phonemes and derives a word label from the reconstructed form,
optionally with position-dependent and context effects, while the learned models see only
summarised features. A phoneme-independence *comparator* is included precisely so the gap can
be measured. Every generative parameter lives in a config file, and a sweep over those
parameters is part of E3/E5.

*Residual.* A positive RQ1 result on synthetic data demonstrates that the estimator can
recover structure the generator put there. It does not establish that human listeners carry
that structure.

---

## S3 — Listener-level leakage would inflate every metric · **H**
Trial-level random splits put the same listener in train and test; a model could then memorise
the listener rather than learn a transferable relationship.

*Mitigation.* Grouped cross-validation by listener id is the only supported split for the
headline results; a dedicated leakage test asserts that no listener id appears on both sides
of any fold and that a deliberately leaky split is detected.

*Residual.* Within-listener repeated *words* can still inflate variance estimates; bootstrap
CIs are computed by resampling listeners, not trials.

---

## S4 — Probability calibration may be poor even when ranking is good · **M**
The threshold caption policy consumes probabilities directly, so miscalibration changes how
much text is shown.

*Mitigation.* Brier score, log loss, ECE and reliability curves are reported alongside PR-AUC
and ROC-AUC for every arm; a calibration wrapper (Platt / isotonic) is evaluated as an explicit
arm rather than applied silently. The budget policy is provided as a calibration-free
alternative.

*Residual.* ECE is bin-count sensitive; the bin count is preregistered in the config.

---

## S5 — A short calibration cannot support phoneme-specific claims · **H**
A 10-item calibration touches at most 10 of 19 onsets. Estimates for unobserved phonemes are
the prior, not evidence.

*Mitigation.* `coverage()` and `n_observations()` are first-class and surfaced in the UI;
unobserved rows are reported by name; E5 measures the degradation curve directly. The external
constraint from Kim et al. (2015) — WRS test-retest r falls from 0.88 (50 words) to 0.76 (25)
to 0.61 (10), with a 95% prediction interval of ±26.22 points at 10 words — is cited as a
ceiling on any "short test is enough" claim.

*Residual.* AUDIRE's calibration is not KS-MWL-A and its reliability has not been measured on
human listeners. It must never be described as a substitute for a clinical WRS.

---

## L1 — CC BY-NC-ND 4.0 on the primary corpus · **H**
Redistribution of modified audio, derived audio corpora, and commercial use are all prohibited;
the card additionally asks users to notify the creator of intended use and scope.

*Mitigation.* `data/raw/` is git-ignored; the fetcher refuses to download until
`AUDIRE_PRIMARY_DATA_USE_NOTIFIED=1` is set by a human and states in its error message that it
will not send the notification itself; `Source.assert_permits()` is a tripwire against adding a
prohibited use later; `redistribution_allowed` is `False` for this source, and noise-mixing
experiments (E8) are restricted to the CC BY-licensed corpus.

*Residual.* The environment variable records only that a human asserts the step was done.

---

## L2 — Auxiliary audiology database is not Korean phoneme ground truth · **M**
Zenodo 17091997 is a non-Korean audiological measures database.

*Mitigation.* Its registry entry prohibits "treating its speech labels as Korean
phoneme-confusion ground truth" and the tripwire test asserts that the prohibition fires.

*Residual.* Any sensitivity analysis using it must be labelled non-Korean in the report.

---

## L3 — Transcribed literature values could be wrong · **M**
Aggregate values taken from published tables (group error rates, similarity/distance means,
test-retest coefficients) were read from journal web pages, not from the typeset PDFs.

*Mitigation.* Each value is stored in `data/literature/` with article, DOI, volume/issue/pages,
URL, access date, and a `verification` field. Values are `second_verification: pending` until a
human confirms them against the published table.

*Residual.* Until second verification, these values may be used only as simulator priors and
plausibility checks — never quoted as findings.

---

## E1 — ASR errors could be mistaken for listener risk · **H**
A word the ASR got wrong is not a word the listener would mishear.

*Mitigation.* ADR-0010: separate fields, ranking on listener risk only, WER reported separately.

*Residual.* A missing word cannot be captioned at all; deletion errors bound achievable recall
and that bound is reported.

---

## E2 — Word timestamp quality · **M**
Selective captioning needs word-level timing; word timestamps are estimated, not exact.

*Mitigation.* The adapter records per-word timing and confidence; exports are validated for
monotonic non-overlapping cues; a timing-jitter sensitivity check is part of E7.

---

## E3 — Model/backend substitution breaking reproducibility · **M**
*Mitigation.* The ASR adapter is an interface with a recorded backend name, model id and
revision written into every result artifact; the deterministic CI path uses a pinned fixture
and does not require downloading weights.

---

## P1 — Private hearing data · **H**
Audiograms, SRT/WRS values and trial responses are sensitive.

*Mitigation.* Real profiles are written only under a git-ignored `private/` directory; the repo
contains schemas and synthetic examples only; `.gitignore` blocks `private/`, `*.participant.*`
and `profiles_local/`; explicit export and delete operations are part of the API surface.

*Residual.* A user can still choose to place a profile elsewhere; the UI states where data is
stored.

---

## P2 — Participant recruitment and ethics · **H** (blocking, unresolved)
If human listeners are recruited, institutional/school ethics approval and informed consent are
required first.

*Status.* **Not obtained. No human participant data has been collected in this repository.**
This is a genuine external blocker: it cannot be resolved by the software. Everything that does
not depend on it has been built.

---

## P3 — Misinterpretation as a diagnostic tool · **H**
Displaying PTA/SRT/WRS next to a risk score invites a diagnostic reading.

*Mitigation.* The package docstring, the UI and every export state that AUDIRE is research and
accessibility software and not a medical device; no output is phrased as a diagnosis, degree of
loss, or treatment recommendation.

---

## X1 — One expert is not a panel · **M**
*Mitigation.* `docs/EXPERT_PROTOCOL.md` forbids describing the review as consensus, Delphi or
clinical validation, and requires disagreements to be preserved.

*Status.* The expert instruments are prepared; **no expert review has been conducted.**
Reporting any expert result before it happens would be fabrication.

---

## X2 — Cherry-picking seeds or operating points · **M**
*Mitigation.* Configs declare seed *lists*; the runner executes all of them and reports
distributions with CIs; caption results are reported as a full Pareto frontier across
10/20/30/40/50 % budgets rather than a single chosen threshold.
