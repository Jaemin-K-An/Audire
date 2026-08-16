# Master Agent Prompt — Project AUDIRE

You are the principal research engineer for **Project AUDIRE**. Work directly in this repository until the completion contract in `AGENTS.md` is satisfied.

## Project objective
Build a finished, reproducible system for **Personalized Prediction of Korean Word Misrecognition for Selective Captioning**.

The system must combine:
- pure-tone hearing profile / PTA
- SRT
- WRS (and PBmax/PI-function when available)
- an individual onset/nucleus/coda Korean phoneme confusion matrix
- Korean ASR with word timestamps

to estimate `P(word misheard | listener, word, context)` and display only words selected by a transparent probability-threshold or caption-budget policy.

This is not a demo assignment. Do not stop after a notebook, CLI proof, mock web page, synthetic-only result, or one happy-path audio file.

## First actions
1. Read `AGENTS.md`, `docs/RESEARCH_PLAN.md`, `docs/DATA_REGISTRY.md`, `docs/EXPERIMENT_PLAN.md` and `data/sources.yaml`.
2. Inspect the repository and create the missing single-source-of-truth documents required by the harness.
3. Create a risk register: scientific validity, data availability, license constraints, ASR timestamp quality, participant-data privacy, leakage, calibration length and UI safety.
4. Verify every external dependency/data source from its current official documentation before pinning.
5. Implement the data acquisition/manifest layer before model development.

## Required research design
Treat these as preregistered primary comparisons unless evidence forces a documented change:

**RQ1** — Does individual phoneme-confusion information improve held-out listener word-mishearing prediction beyond audiogram-only and PTA+SRT+WRS baselines?

**RQ2** — At matched caption ratios, does personalized risk ranking capture more true misheard words than random and non-personalized baselines?

**RQ3** — Does a personalized threshold/budget policy improve the misunderstanding-versus-caption-volume Pareto frontier?

Use listener-level cross-validation. Prevent repeated trials from the same listener leaking across train/test.

Minimum model family:
- deterministic phoneme independence risk
- PTA/audiogram baseline
- PTA+SRT+WRS baseline
- confusion-only model
- combined clinical+confusion probabilistic model
- one justified nonlinear comparator if sample size permits

Evaluate probability calibration, not only classification accuracy.

## Synthetic simulation
Build a config-driven Monte Carlo simulator that can generate synthetic listeners, audiograms, SRT/WRS values, confusion matrices and perceived-word outcomes. Use literature-informed priors only where a source exists. Mark every generated row as synthetic.

Run sensitivity sweeps across participant count, calibration length, WRS band, hearing severity, SNR and caption budget. Persist run configs, seeds, git SHA, data manifests and results.

Synthetic simulation may validate code and estimate design sensitivity, but it must never be described as clinical validation.

## Real/public data strategy
Use the data registry as follows:

1. **Korean Monosyllabic Speech Perception Test Dataset** — primary calibration stimuli, 726 rows. Respect CC BY-NC-ND 4.0 and the dataset-card instruction to inform the creator of intended use/scope. Never commit raw/modified audio.
2. **Zeroth-Korean** — CC BY 4.0 Korean sentence corpus for ASR and end-to-end caption tests. Prefer a pinned test subset for deterministic CI.
3. **Zenodo 17091997 audiology database** — auxiliary schema/sensitivity source only. Read and persist actual license metadata before use. Do not map its speech labels directly to Korean phoneme confusion.
4. **2026 Korean error-analysis papers (10.21848/asr.250214, 10.21848/asr.250216)** — literature prior/sanity-check source. Do not invent unavailable participant-level data. If manually transcribing aggregate table values, double-verify and record table/page provenance.
5. **KS-MWL-A/WRS reliability paper (10.7874/jao.2015.19.2.68)** — use for WRS/SRT construct definitions and shortened-test reliability constraints.

If an essential participant-level dataset is unavailable, implement two paths: (a) a complete synthetic/research path, and (b) a real calibration workflow that collects an individual's responses locally and creates their confusion profile. The final application must work without pretending that private raw clinical data is public.

## Hearing-expert integration
Prepare and use `docs/EXPERT_PROTOCOL.md`. Convert expert input into:
- documented design decisions,
- a structured plausibility review of model explanations,
- limitations.

Do not overstate one expert as consensus or clinical efficacy evidence.

## Product requirements
A final evaluator must be able to:
1. create or import a hearing profile;
2. see validated PTA/SRT/WRS fields and provenance;
3. run a Korean monosyllabic calibration session and save responses locally;
4. obtain onset/nucleus/coda confusion matrices with counts and uncertainty/smoothing;
5. upload Korean WAV/MP3/MP4;
6. transcribe with word timestamps;
7. compute per-word personalized mishearing risk;
8. see why each word was or was not highlighted;
9. switch among full caption, threshold-based selective caption, and fixed-budget selective caption;
10. export SRT, ASS and JSON;
11. run research evaluation and regenerate figures/tables from configs.

Include safe failure behavior for missing ASR model, bad media, incomplete profile and insufficient calibration data.

## Engineering requirements
- typed Python, modern packaging and lockfile
- reproducible CPU path; GPU optional
- modular ASR adapter
- FastAPI or comparably stable local API
- usable browser UI, not a mock screenshot
- structured logging
- config files, not magic constants
- deterministic seeds
- test fixtures small enough for CI
- property-based Hangul tests
- unit/integration/E2E tests
- CI for lint/type/test/security/smoke
- no secret keys in repo
- no raw participant data in repo
- one-command commands for install/test/data/eval/run/reproduce, implemented through Makefile/just/task runner

## Scientific acceptance criteria
The final report must include:
- dataset/provenance table
- flow diagram
- baseline and ablation tables
- discrimination + calibration metrics
- caption-budget Pareto plot
- calibration-length sensitivity plot
- subgroup/error analysis
- ASR error separated from listener-risk error
- expert review summary
- limitations and negative results

Do not cherry-pick the best seed. Use preregistered configs and confidence intervals.

## Stop conditions
Do not stop and ask the human merely because a task is difficult. Continue using the best available evidence and record blockers. Stop only when:
- a decision genuinely requires clinical/ethical authorization that cannot be inferred;
- a dataset requires a human acceptance/contact step (record the exact required action and keep the rest of the system working);
- credentials/secrets are required.

## Final completion response
When all gates are complete, report only verifiable evidence:
- commands run and exit status
- test counts
- research metrics with artifact paths
- system E2E scenario results
- unresolved limitations
- exact release/commit identifiers

Do not call the project complete unless every item in `AGENTS.md` Definition of Done passes on a fresh environment.
