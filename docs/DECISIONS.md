# Architecture and Method Decision Records

ADR-style. Each record states the decision, why, what was rejected, and how the decision
can be falsified or revisited. Decisions that affect a scientific claim also name the
test or experiment that would expose them as wrong.

---

## ADR-0001 — Python 3.12 with a source layout, pinned versions and a lockfile
**Status:** accepted (2026-08-16)

**Context.** The host machine's default interpreter is 3.9.13, but the current releases of
numpy (2.5.2), scipy (1.18.0) and librosa require ≥3.12, and pandas/scikit-learn/matplotlib
require ≥3.11. Python 3.12.9 is available via Homebrew.

**Decision.** Target `>=3.12,<3.14`. Use a `src/` layout so tests import the installed
package rather than the working tree. Pin exact versions in `pyproject.toml` and freeze a
`requirements.lock`.

**Rejected.** Supporting 3.9 — it would force old numpy/scipy and defeat the reproducibility
goal.

**Revisit if** a required dependency drops 3.12 support.

---

## ADR-0002 — Dependency versions verified against PyPI before pinning
**Status:** accepted (2026-08-16)

**Context.** `AGENTS.md` §"Verify current official documentation … before pinning".

**Decision.** Every pinned version in `pyproject.toml` was read from the live PyPI JSON API
on 2026-08-16, not from memory. `faster-whisper` 1.2.1 was confirmed to exist and to support
Python ≥3.9 before being made the default ASR backend candidate.

**Consequence.** Re-verification is required before any version bump; `make audit` runs a
vulnerability check.

---

## ADR-0003 — External data provenance is verified from live source APIs, not from prose
**Status:** accepted (2026-08-16)

**Context.** `docs/DATA_REGISTRY.md` asserts licences and row counts. Prose can drift from
reality.

**Decision.** Before writing `data/sources.yaml`, each source's live metadata API was queried
and the registry records what was actually observed, including two discrepancies:

1. **The primary corpus's metadata file is `metadata.csv`, not `updated_metadata.csv`.**
   The dataset card's prose names the latter; the repository's real file listing (729 entries:
   `.gitattributes`, `README.md`, `metadata.csv`, and 726 `audio_files/*.wav`) contains only
   the former. The registry pins the real name and the code reads it from the registry.
2. **The Zeroth-Korean mirror has no machine-readable `license` front-matter field.**
   `cardData.license` is absent. CC BY 4.0 is stated in the card *body* and by upstream
   OpenSLR SLR40. The registry records the licence together with the fact that the
   machine-readable tag is missing, so the claim is auditable.

**Consequence.** `data/sources.yaml` carries a `verified_at` date per source and
`_fetch_zeroth`/`_fetch_zenodo` re-check the live licence at download time; a Zenodo licence
change is a hard error rather than a silent inheritance.

---

## ADR-0004 — Full orthographic jamo inventories, not the literature's reduced inventories
**Status:** accepted (2026-08-16)

**Context.** Ma et al. (2026, DOI 10.21848/asr.250216) analyse 18 onsets, 16 nuclei and 8
codas. AUDIRE must score arbitrary Korean caption text, not only their stimulus set.

**Decision.** Confusion matrices use the complete modern orthographic inventory: 19 onsets
(including the null onset `ㅇ`), 21 nuclei, and 27 codas plus an explicit `NO_CODA`
category. `audire.confusion.grouping` provides the phonological mapping down to the
7-way neutralised coda set when a comparison to published tables is required.

**Consistency check.** 19 onsets − the null onset = **18**, matching Ma et al.'s onset count;
7 neutralised surface codas + `NO_CODA` = **8**, matching their coda count. Their nucleus
count of 16 is not reconstructible from the material available to this project and is
recorded as an open question in `docs/RESULTS.md` rather than guessed at.

**Falsifiable by** any evidence that a reduced inventory predicts held-out mishearing better;
`with_smoothing`/`grouping` make that ablation cheap to run.

---

## ADR-0005 — Independent error taxonomy, explicitly not a reimplementation
**Status:** accepted (2026-08-16)

**Context.** Joo et al. (2026) code errors into 10 categories. The coding manual is not
available to this project.

**Decision.** AUDIRE defines its own taxonomy (correct / substitution / omission / addition /
compound / no-response) directly from the orthographic decomposition, and documents that its
counts are **not** interchangeable with that paper's. `ㅇ` in onset position is the
phonologically null onset, so `ㄱ→ㅇ` is an omission and `ㅇ→ㄱ` an addition; coda omission and
addition are ordinary `NO_CODA` transitions.

**Consequence.** No AUDIRE output may be presented as reproducing a published error-rate table.

---

## ADR-0006 — Dirichlet posterior-mean smoothing with an explicit, serialisable prior
**Status:** accepted (2026-08-16)

**Context.** A 25-item calibration cannot observe most of a 19×20 onset matrix. Unsmoothed
rows are degenerate (0 or 1) and unobserved rows are undefined.

**Decision.** Estimate row *i* as `(n_ij + α·π_ij) / (n_i + α)` with `α` a total pseudo-count
and `π_i` an explicit row-stochastic prior. Default `α = 1.0` with a uniform prior. When a
group profile exists, `SmoothingSpec(kind="explicit", prior=group)` performs hierarchical
shrinkage toward the group.

**Rejected.** Add-one-per-cell Laplace — with 20–28 columns it injects 20–28 pseudo-counts and
would swamp a 25-trial calibration.

**Guardrails.** `α = 0` with an unobserved row raises rather than returning `NaN`; counts are
never discarded, so `p_correct` and `n_observations` are always both available; unobserved rows
are reported by `unobserved_targets` and `coverage`.

---

## ADR-0007 — The system must work without the licence-gated primary corpus
**Status:** accepted (2026-08-16)

**Context.** The primary corpus is CC BY-NC-ND 4.0 and its card asks users to notify the
creator before use. That is a human step AUDIRE must not automate, so a fresh evaluator may
not have the audio.

**Decision.** Ship a second stimulus source: a deterministic phoneme-balanced monosyllable
design generated from the Hangul inventory itself (19 × 21 × 8 = 3,192 combinations, visited
in a coprime cyclic order so that the first *k* items are the balanced *k*-item list). It
carries no third-party licence. The web client speaks it with the browser's Korean speech
synthesiser.

**Stated limitation, carried into every report.** Browser TTS is **not level-calibrated**.
Calibration performed with it is a research/accessibility measurement, never a clinical one.
Recorded corpus audio remains the preferred source when the human step has been completed.

---

## ADR-0008 — Confusion profile kept structurally separate from the clinical profile
**Status:** accepted (2026-08-16)

**Context.** RQ1 asks whether `C_u` adds information beyond PTA/SRT/WRS. If the two were
merged into one object or one score, the question could not be asked.

**Decision.** `HearingProfile` (audiometric + speech scores) and `ConfusionProfile` (three
matrices) are separate types with separate persistence, and the feature builders that consume
them are separately switchable so that the four required ablation arms are literal
configurations rather than code edits.

**Falsification.** If the combined arm never beats the clinical-only arm on held-out
listener-level data, RQ1 is answered negatively and that must be reported.

---

## ADR-0009 — Synthetic provenance is a required field, enforced at every boundary
**Status:** accepted (2026-08-16)

**Decision.** `is_synthetic` is a required (non-defaulted) constructor argument on
`ConfusionProfile` and on the listener/trial records, so it cannot be forgotten.
`pool_profiles` refuses to merge synthetic and non-synthetic listeners, which would otherwise
launder simulated evidence into a real listener's prior.

---

## ADR-0010 — ASR uncertainty is a separate channel from listener mishearing risk
**Status:** accepted (2026-08-16)

**Context.** `AGENTS.md` §9: a bad ASR hypothesis must never become evidence that the listener
would mishear the word.

**Decision.** `WordRisk` carries `asr_confidence` and `listener_risk` as distinct fields, and
the caption policies rank on `listener_risk` only. ASR confidence is surfaced in the
explanation and in the JSON export, and end-to-end evaluation reports ASR WER separately from
mishearing-prediction error.
