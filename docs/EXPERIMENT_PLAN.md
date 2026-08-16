# Experiment and Simulation Plan

## E0 — data integrity
Acceptance:
- all datasets downloaded through registered sources
- source/version/license/checksum manifest generated
- expected primary row count = 726
- Zeroth test rows = 457 when using the mirrored v2 layout
- no raw data tracked by Git

## E1 — Hangul decomposition correctness
Build exhaustive tests for modern Hangul syllable decomposition/recomposition. Validate onset/nucleus/coda indexing and no-coda handling. Property test: recomposition(decomposition(x)) == x for all supported Hangul syllables.

## E2 — confusion matrix construction
Input: target/perceived response pairs.
Output:
- onset matrix
- nucleus matrix
- coda matrix
- smoothed row-normalized probabilities
- sample-count matrix

Tests:
- rows sum to one after smoothing
- absent categories handled explicitly
- additions/omissions represented without silently dropping data

## E3 — synthetic listener recovery
Generate listeners with known parameters, fit the pipeline, and check whether estimated risks recover simulated truth within predeclared tolerance.

## E4 — ablation study
Listener-level split only. Compare:
- PTA only
- PTA+SRT+WRS
- confusion-only
- PTA+SRT+WRS+confusion

Never split repeated trials from the same listener across train and test in a way that leaks identity.

## E5 — calibration-length study
Compare 10/25/50/100/full calibration stimuli. Selection strategies:
- random
- phoneme-balanced
- information-gain / uncertainty-based

Outcome: prediction degradation and confidence intervals.

## E6 — selective-caption budget study
For each model, rank words by predicted risk and evaluate 10/20/30/40/50% caption budgets. Produce Pareto plot and paired bootstrap confidence intervals.

## E7 — ASR end-to-end regression
Run pinned Zeroth-Korean test subset through ASR -> token/word alignment -> Hangul decomposition -> risk scoring -> caption renderer. Record ASR WER separately from hearing-risk errors.

## E8 — noise robustness
Create reproducible noise mixtures only from sources whose licenses permit transformations. Test several SNR values. Never create or redistribute derivatives from the CC BY-NC-ND primary monosyllable dataset.

## E9 — expert review
Run the structured protocol in `EXPERT_PROTOCOL.md`; store de-identified ratings and qualitative notes outside the public repo, with only aggregate/report text committed.

## E10 — final system acceptance
A fresh machine must be able to:
1. install dependencies from lockfiles;
2. fetch permitted datasets using scripts;
3. create/import a hearing profile;
4. run calibration from local stimuli;
5. upload Korean audio/video;
6. transcribe and timestamp it;
7. compute word-level risk;
8. render selective captions;
9. export SRT/ASS/JSON;
10. reproduce the research report from fixed configs and seeds.
