# AUDIRE Work Ledger

Live status. A gate is **done** only when its code, its tests and its recorded evidence
all exist. Nothing below G8 may be described as a finished product.

Legend: ✅ done · 🟡 partial · ⬜ not started · 🚫 externally blocked

---

## G0 — evidence, licences, provenance, data layer ✅

| item | status | evidence |
|---|---|---|
| Research plan / experiment plan / data registry / expert protocol in `docs/` | ✅ | `docs/` |
| Source registry with per-source permitted/prohibited uses | ✅ | `data/sources.yaml` |
| Licences verified against **live** source APIs | ✅ | ADR-0003; two discrepancies recorded |
| Literature refs verified against publishers | ✅ | 4 entries with DOI + access date |
| `scripts/fetch_data.py` — idempotent, restartable, manifest + SHA-256 | ✅ | `src/audire/data/fetch.py` |
| Human-acknowledgement gate for the CC BY-NC-ND corpus | ✅ | 3 tests incl. "never offers to send" |
| Prohibited-use tripwire | ✅ | `Source.assert_permits` + 2 tests |
| Repo hygiene gate (no participant data, corpora, weights, secrets) | ✅ | `scripts/check_repo_hygiene.py`, in CI |
| ADRs + risk register | ✅ | `docs/DECISIONS.md`, `docs/RISK_REGISTER.md` |
| Makefile one-command entrypoints | ✅ | `Makefile` |
| CI: format, lint, strict types, tests, provenance, privacy, smoke | ✅ | `.github/workflows/ci.yml` |

## G1 — perceptual core ✅

| item | status | evidence |
|---|---|---|
| Hangul decomposition/recomposition | ✅ | exhaustive over all 11,172 syllables |
| `recompose(decompose(x)) == x` property | ✅ | exhaustive + Hypothesis on mixed text |
| "No coda" as an explicit category, never `""` | ✅ | regression test |
| Response parser + error taxonomy | ✅ | ADR-0005; 3 observations per trial always |
| Confusion matrices: counts and probabilities separate | ✅ | `n_observations` always available |
| Omission / addition / no-response as ordinary cells | ✅ | 5 tests |
| Dirichlet smoothing with explicit serialisable prior | ✅ | ADR-0006 |
| Rows sum to 1 including unobserved rows | ✅ | parametrised over all positions |
| Korean phonology tables (7-coda neutralisation, 3-way laryngeal) | ✅ | import-time self-check |

## G2 — profile and simulation ✅

| item | status | evidence |
|---|---|---|
| `HearingProfile` with explicit missingness | ✅ | `missing()` / `available()` / `completeness()` |
| PTA names its method; partial averages refused | ✅ | 5 named methods + `pta_detail` |
| "No response at max output" ≠ "not tested" ≠ threshold | ✅ | dedicated test |
| WRS requires its presentation level | ✅ | validator + test |
| PBmax / PI function / rollover (descriptive only) | ✅ | `PIFunction` |
| MCL/UCL with ordering validation | ✅ | `LoudnessLevels` |
| Severity strata from a named, cited scheme | ✅ | WHO grades, verified 2026-08-16 |
| Private profile store (git-ignored) + export + delete | ✅ | `ProfileStore` |
| Simulator: every constant carries an evidence label | ✅ | `Evidenced` base class |
| Listener accuracy anchored to Joo et al. 2026 | ✅ | `solve_position_error_mass` |
| Generator structurally differs from the scoring model | ✅ | lexical repair (RISK S2) |
| Parameter recovery as calibration lengthens | ✅ | `@pytest.mark.slow` test |
| `is_synthetic` propagated everywhere | ✅ | one test over the whole cohort |
| `model_inputs()` cannot expose the true structure | ✅ | dedicated test |

## G3 — risk modelling ✅

| item | status | evidence |
|---|---|---|
| Deterministic `R_phon` baseline | ✅ | `PhonemeIndependenceRisk` |
| PTA-only / PTA+SRT+WRS / confusion-only / combined arms | ✅ | `ABLATION_ARMS`, nested blocks |
| Non-personalized floor arm | ✅ | `word_context_only` |
| Nonlinear comparator | ✅ | `GradientBoostedRiskModel` |
| Fold-safe imputation with missing indicators | ✅ | inside the sklearn Pipeline |
| Listener-level splits only; leakage guard on every fold | ✅ | `LeakySplitter` proves it fires |
| Brier / log loss / ECE / MCE / reliability + PR-AUC / ROC-AUC | ✅ | `eval/metrics.py`, 100% covered |
| Prevalence floor reported per arm | ✅ | `prevalence_baseline_metrics` |
| Listener-level bootstrap; paired contrasts | ✅ | `eval/bootstrap.py` |
| Platt / isotonic calibration on held-out **listeners** | ✅ | `CalibratedRiskModel` |

## G4 — caption engine ✅

| item | status | evidence |
|---|---|---|
| `WordRisk` keeps ASR confidence separate from listener risk | ✅ | ADR-0010 + tests |
| Threshold policy | ✅ | `ThresholdPolicy` |
| Fixed-budget policy | ✅ | `BudgetPolicy`, deterministic under ties |
| Full-caption mode | ✅ | `FullCaptionPolicy` |
| Per-word explanation with evidence counts | ✅ | `WordRisk.explanation` |
| SRT / ASS / JSON export with snapshot tests | ✅ | validated non-overlapping cues |
| Caption-budget Pareto study (RQ2) | ✅ | per-listener **and** pooled |
| Personalized vs global threshold (RQ3) | ✅ | with the equity view |

## G5 — end-to-end ASR ✅ (real weights not yet exercised here)

| item | status | evidence |
|---|---|---|
| Replaceable ASR adapter interface | ✅ | `ASRBackend`, `Token`, `Transcript` |
| `faster-whisper` backend with word timestamps | ✅ | written against the verified current API |
| Per-token confidence kept separate from risk | ✅ | `None` when absent, never fabricated |
| Word-timestamp defects surfaced, not swallowed | ✅ | `Transcript.timing_problems()` |
| Missing weights fail loudly, never degrade silently | ✅ | `ASRUnavailable` with an actionable message |
| Replay backend for deterministic CI | ✅ | preserves the original recogniser's identity |
| Non-Hangul tokens kept and flagged, not dropped | ✅ | `meta.reason_not_scored` |
| Incomplete profile / thin calibration refused | ✅ | `check_ready`, ≥10 trials |
| CPU-only E2E smoke test (no model download) | ✅ | 21 integration tests |
| **Real `faster-whisper` weights run on audio** | ⬜ | `@pytest.mark.asr` tests exist but have **not** been executed against downloaded weights on this machine |
| WER reported separately from mishearing error | ⬜ | contract in place; E7 regression not run |
| Media ingest verified for mp3/mp4 | ⬜ | ffmpeg 7.1.1 present; not exercised |

## G6 — complete user system ⬜ NOT STARTED

| item | status | note |
|---|---|---|
| FastAPI application | ⬜ | fastapi 0.141.1 installed, no app written |
| Profile create/import UI | ⬜ | schema and store are ready behind it |
| Calibration session UI | ⬜ | built-in balanced stimulus list is ready |
| Upload → transcribe → risk → caption UI | ⬜ | depends on G5 |
| Export SRT / ASS / JSON from the UI | ⬜ | exporters are ready behind it |
| API contract tests + browser E2E | ⬜ | |

## G7 — validation and sensitivity 🟡 PARTIAL

| item | status | evidence |
|---|---|---|
| Preregistered config with all seeds executed | ✅ | `experiments/configs/rq1_main.yaml` |
| Ablation + caption + threshold studies recorded | ✅ | `experiments/registry.yaml` |
| Calibration-length study (10/25/50/100/full) | ⬜ | selection strategies exist, sweep not run |
| SNR sweep | 🟡 | simulator supports it; one condition run |
| Speaker sensitivity | 🟡 | simulator supports it; not analysed |
| Subgroup analysis by severity / WRS band | ⬜ | cohort records the strata |
| Idiosyncrasy sweep (Dirichlet concentration) | ⬜ | **the key missing sweep** given the RQ2 finding |
| Expert review | 🚫 | instruments prepared; **no review conducted** |

## G8 — reproducible release 🟡 PARTIAL

| item | status | evidence |
|---|---|---|
| Experiment registry with git SHA / lock hash / seeds / manifests | ✅ | `src/audire/experiments/registry.py` |
| `make simulate` / `eval` / `caption-eval` / `figures` | ✅ | via the `audire` CLI |
| Figures and tables regenerate from recorded artifacts alone | ✅ | `experiments/figures.py` |
| `make reproduce` full chain | 🟡 | the `sensitivity` target is not implemented |
| Fresh-environment install verification | ⬜ | |
| Final technical/research report | 🟡 | `docs/RESULTS.md` holds the recorded findings |

---

## External blockers (cannot be resolved by software)

| id | blocker | exact action required |
|---|---|---|
| B1 | Human participant data | Institutional/school ethics approval and informed consent **before any collection**. None obtained; none collected. |
| B2 | Primary corpus notification | The dataset card asks users to inform Woojae Han (woojaehan@hallym.ac.kr) of intended use and scope. A human must do this, then set `AUDIRE_PRIMARY_DATA_USE_NOTIFIED=1`. AUDIRE will not send it. |
| B3 | Expert review | Requires the hearing professional's time and their consent to report credentials. Protocol is ready in `docs/EXPERT_PROTOCOL.md`. |

---

## Next highest-risk work, in order

1. **Idiosyncrasy × calibration-length sweep.** The RQ2 result says the confusion profile
   does not beat a word-length heuristic at a per-listener budget under the current
   generative settings. The sweep determines whether that is a property of the method or of
   the chosen `dirichlet_concentration` and 100-trial calibration. This is the single most
   informative remaining experiment.
2. **G5 ASR adapter**, because G6 depends on it and because the ASR-versus-listener-risk
   separation is currently only a contract, never exercised on real audio.
3. **G6 API and web application.**
4. Fresh-environment install verification and the final report.
