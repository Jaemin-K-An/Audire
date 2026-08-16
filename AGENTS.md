# AUDIRE Agent Harness

This repository is an autonomous research-engineering project. The target is a **finished, reproducible, working system**, not an MVP, mockup, notebook-only demo, or slideware prototype.

## 0. Mission
Build and validate AUDIRE: a Korean selective-caption system that predicts word-level mishearing risk from a personalized speech-perception profile containing audiometric and speech-recognition measures plus an individual phoneme confusion matrix.

The research question and system must remain coupled: every UI feature must map to a tested pipeline component; every claimed research result must be reproducible from code and versioned configuration.

## 1. Non-negotiable rules
1. Never fabricate clinical data, participant responses, p-values, effect sizes, confusion matrices, or citations.
2. Synthetic data must always carry explicit synthetic provenance.
3. Never commit raw participant-level hearing data or identifiable information.
4. Never commit third-party raw audio/model weights unless license and repository policy explicitly permit it. Prefer scripted acquisition.
5. The primary 726-stimulus Korean dataset is CC BY-NC-ND 4.0; do not redistribute modified audio. Respect the dataset-card request to notify the creator before use.
6. Do not treat the auxiliary non-Korean audiology dataset as Korean phoneme ground truth.
7. Do not turn PTA/SRT/WRS into diagnosis. AUDIRE is research/accessibility software, not a medical device.
8. No `TODO`, fake API, placeholder metrics, mocked inference, or manually hard-coded “successful” result may remain at final acceptance.
9. Any external fact that affects design must have a source in `docs/` with URL/DOI and access date.
10. If a proposed claim is not supported, weaken the claim; never weaken the evidence standard.

## 2. Harness control loop
For each work unit, execute this loop:

**OBSERVE -> SPECIFY -> TEST -> IMPLEMENT -> VERIFY -> RECORD -> NEXT**

### OBSERVE
Read repository state, failing tests, experiment registry, current decisions and data provenance before editing.

### SPECIFY
Write the smallest explicit contract for the work unit: inputs, outputs, failure modes, acceptance test.

### TEST
Create or update tests before implementation when feasible. For research code, create an executable validation check or expected invariant.

### IMPLEMENT
Make the smallest coherent production-quality change. Avoid parallel alternate implementations unless an experiment explicitly compares them.

### VERIFY
Run targeted tests, then the relevant integration suite. For model changes, rerun the pinned evaluation configuration.

### RECORD
Update `docs/DECISIONS.md`, experiment results, provenance and/or task ledger. Record negative outcomes.

### NEXT
Choose the next highest-risk blocker, not the easiest cosmetic task.

## 3. Single source of truth
Create and maintain:
- `docs/SYSTEM_SPEC.md` — architecture/contracts
- `docs/DECISIONS.md` — ADR-style decisions
- `docs/TASKS.md` — granular work ledger with evidence links
- `docs/RESULTS.md` — only reproduced metrics
- `experiments/registry.yaml` — config, seed, git SHA, data manifest, metrics artifact
- `data/manifests/` — source/version/license/checksum manifests

Do not duplicate contradictory requirements across files. `AGENTS.md` is the top-level execution contract.

## 4. Required system architecture
Implement as typed Python packages plus a deployable local web application/API. A suggested structure:

- `src/audire/hangul/` — jamo decomposition/recomposition
- `src/audire/profile/` — audiogram/SRT/WRS/PBmax schema and validation
- `src/audire/confusion/` — response parser, confusion matrices, smoothing
- `src/audire/risk/` — baselines, learned probabilistic models, calibration
- `src/audire/asr/` — ASR adapter with word timestamps
- `src/audire/caption/` — word ranking, threshold/budget policy, SRT/ASS/JSON
- `src/audire/sim/` — clearly synthetic listener/trial generator
- `src/audire/eval/` — metrics, bootstrap CIs, ablations, plots
- `apps/api/` — stable API
- `apps/web/` — usable calibration/profile/audio/caption interface
- `tests/` — unit, property, integration, E2E

The exact frontend stack may be changed by ADR if a simpler robust option improves reproducibility.

## 5. Core data contracts
### HearingProfile
Required fields must support missingness explicitly:
- frequencies and left/right dB HL thresholds
- PTA calculation method/version
- SRT
- WRS and presentation level
- optional PBmax / PI function
- optional MCL/UCL
- hearing-aid state
- provenance/source (`manual`, `clinical_export`, `synthetic`)

### ConfusionProfile
Store counts and probabilities separately for onset/nucleus/coda. Include deletion/no-coda/addition states where needed. Never infer confidence from a probability without retaining sample count.

### WordRisk
Must contain:
- text/token
- start/end timestamp
- predicted probability
- model version
- contributing feature/risk explanation
- caption decision + threshold/budget policy
- ASR confidence separately from listener risk when available

## 6. Model development contract
Start with interpretable baselines. Compare models; do not assume complexity wins.

Required minimum comparisons:
- audiogram/PTA only
- PTA + SRT + WRS
- confusion-only
- PTA + SRT + WRS + confusion
- deterministic phoneme independence baseline

Primary split is **listener-level**, never random trial-level when that leaks a listener into both train and test.

Probability quality matters. Report Brier score, log loss and calibration along with discrimination metrics.

## 7. Research simulation contract
Synthetic simulation must be a test instrument, not evidence of real clinical efficacy.

All random generators use explicit seeds. Every experiment is config-driven. A run is invalid unless it records:
- git SHA
- data manifest IDs
- environment/dependency lock hash
- random seed(s)
- exact config
- stdout/log artifact
- metrics artifact

## 8. Data acquisition contract
Implement `scripts/fetch_data.py` plus manifest/checksum validation. Downloads must be idempotent and restartable.

For the CC BY-NC-ND primary dataset, require an explicit local acknowledgment (environment variable or CLI flag) that the research-use notification requirement has been handled. Never auto-send email.

## 9. ASR contract
Use a maintained Korean-capable ASR backend with word timestamps. Default may be `faster-whisper`, but verify current library/model support before pinning. The adapter must allow model substitution.

Separate:
- ASR transcription error
- listener mishearing risk

A bad ASR hypothesis must never be silently treated as a true listener-specific risk signal.

## 10. Caption policy contract
Support both:
- probability threshold policy
- fixed caption-budget policy

Outputs:
- HTML/web rendering
- SRT
- ASS (for visual emphasis)
- JSON with full risk metadata

Full captions remain available as a comparison/accessibility mode.

## 11. Testing contract
Required before final acceptance:
- exhaustive Hangul syllable round-trip property tests
- matrix normalization/count invariants
- missing-data validation tests
- synthetic parameter-recovery tests
- model leakage tests
- deterministic experiment tests
- data manifest/checksum tests
- API contract tests
- browser E2E happy path and failure paths
- SRT/ASS export snapshot tests
- CPU-only end-to-end smoke test

Target: critical core modules >=90% branch coverage; do not game coverage with trivial tests.

## 12. CI / quality gates
CI must run formatting/linting, type checking, tests, security/dependency checks and a small deterministic E2E fixture. Pin dependencies with lockfiles.

No merge/release if:
- test suite fails
- data provenance is missing
- an experiment result cannot be reproduced
- a clinical claim lacks evidence
- a license requirement is unresolved

## 13. Milestone gates
### G0 — evidence and legal/data gate
Research plan, source registry, licenses, dataset fetchers, claims matrix.

### G1 — perceptual core
Hangul parser + response parser + confusion matrix engine + tests.

### G2 — profile and simulation
Validated HearingProfile + synthetic listener/trial engine + parameter-recovery tests.

### G3 — risk modeling
Baselines + learned model + calibration + listener-level evaluation + ablations.

### G4 — caption engine
Risk-to-caption policy + exports + evaluation by caption budget.

### G5 — end-to-end ASR
Korean audio -> transcript/timestamps -> risk -> selective caption, with ASR error tracked separately.

### G6 — complete user system
Profile import/editor, calibration workflow, upload/process/result UI, explanation and exports.

### G7 — validation
Noise/speaker sensitivity, expert consultation, failure analysis, ethics/limitations.

### G8 — reproducible final release
Fresh-machine install, one-command data preparation, one-command tests, one-command research reproduction, packaged release, final technical/research report.

Do not label any earlier gate as “finished product”.

## 14. Definition of done
The project is done only when a new evaluator can clone the repository and, using documented commands:
1. install pinned dependencies;
2. download/verify permitted data;
3. run all tests;
4. generate a synthetic profile and reproduce benchmark results;
5. enter/import a real profile without committing it;
6. perform a Korean monosyllable calibration;
7. process a Korean audio/video file;
8. view selective captions and per-word risk explanations;
9. export SRT/ASS/JSON;
10. regenerate the final tables/figures/report from experiment configs;
11. inspect provenance showing exactly which data/model/version produced each result.

If any step requires hand-editing source code, hidden local files, fabricated outputs, or an undocumented manual patch, the system is not done.
