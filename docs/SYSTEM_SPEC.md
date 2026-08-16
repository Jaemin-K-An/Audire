# AUDIRE System Specification

Architecture and data contracts. Where an interface is defined but not yet implemented,
this document says so explicitly — see `docs/TASKS.md` for gate status.

---

## 1. Pipeline

```
                      ┌─────────────────────────────────────────┐
   audio / video ───► │ ASR adapter        (G5, NOT IMPLEMENTED)│
                      │  → tokens + word timestamps + conf      │
                      └───────────────┬─────────────────────────┘
                                      │ text tokens
                                      ▼
  ┌────────────────┐          ┌───────────────────┐
  │ HearingProfile │────┐     │ audire.hangul     │
  │ PTA/SRT/WRS/…  │    │     │ onset/nucleus/coda│
  └────────────────┘    │     └─────────┬─────────┘
                        ▼               │
  ┌────────────────┐  ┌─────────────────▼─────────┐   ┌──────────────────┐
  │ calibration    │─►│ audire.risk.features      │──►│ RiskModel        │
  │ responses      │  │  word · context · pta ·   │   │ logistic / R_phon│
  └───────┬────────┘  │  clinical · confusion     │   │ / boosting       │
          ▼           └───────────────────────────┘   └────────┬─────────┘
  ┌────────────────┐                                            │ P(misheard)
  │ ConfusionProfile│                                           ▼
  │ C_u per position│                              ┌────────────────────────┐
  └────────────────┘                               │ WordRisk               │
                                                   │  listener_risk         │
                                                   │  asr_confidence (sep.) │
                                                   └───────────┬────────────┘
                                                               ▼
                                            ┌──────────────────────────────┐
                                            │ CaptionPolicy                │
                                            │  full / threshold / budget   │
                                            └──────────┬───────────────────┘
                                                       ▼
                                              SRT · ASS · JSON
```

The two personalization inputs are deliberately **separate objects**: `HearingProfile`
carries the global speech-recognition picture, `ConfusionProfile` carries the local error
structure. RQ1 is the question of whether the second adds information beyond the first, so
merging them would make the question unaskable (ADR-0008).

---

## 2. Module contracts

| module | responsibility | key invariant |
|---|---|---|
| `audire.hangul` | jamo decomposition/recomposition, inventories | total and exact over U+AC00–U+D7A3; "no coda" is an explicit category |
| `audire.confusion` | response parsing, error taxonomy, `C_u` | counts and probabilities always both available; no observation dropped |
| `audire.profile` | clinical schema, derived measures, private storage | missingness explicit; every derived value names its method |
| `audire.risk` | features, models, calibration | arms differ only in listener representation; imputation inside the pipeline |
| `audire.caption` | policies, `WordRisk`, exports | ASR confidence never enters `listener_risk` |
| `audire.sim` | synthetic listeners and trials | `is_synthetic=True` everywhere; generator ≠ scoring model |
| `audire.eval` | metrics, splits, bootstrap, studies | listener-level splits only; leakage asserted per fold |
| `audire.data` | source registry, fetch, manifests, stimuli | nothing fetched that is not registered; SHA-256 per file |
| `audire.experiments` | configs, provenance registry, figures | every reported number traces to a run record |

---

## 3. Core data contracts

### 3.1 `HearingProfile`

Required: `listener_id` (opaque, never a name), `source` ∈ {manual, clinical_export,
synthetic}, `is_synthetic`, at least one ear.

Per ear: air-conduction thresholds keyed by frequency, each an `AudiogramPoint` with
`db_hl | None`, `no_response`, `masked`; `SpeechScores` (SRT, WRS + its presentation level
and word list); optional `PIFunction` and `LoudnessLevels`.

Invariants enforced by validators:

* A WRS without its presentation level is rejected — it is uninterpretable.
* `no_response=True` requires the level at which no response was obtained, and that point
  is excluded from PTA rather than averaged in.
* PTA returns `None`, never a partial average, when a required frequency is missing.
* `source=synthetic` and `is_synthetic` must agree.
* UCL may not fall below MCL.

### 3.2 `ConfusionProfile`

Three `ConfusionMatrix` objects (onset / nucleus / coda). Each is rectangular: rows are
target categories, columns are targets plus `NO_RESPONSE`.

* `counts` (integer) and `probabilities()` (smoothed) are separate; `n_observations` is
  always available beside any probability.
* Smoothing is a Dirichlet posterior mean with an explicit prior (`SmoothingSpec`).
* Unobserved rows equal the prior exactly and are listed by `unobserved_targets`.
* `coverage` reports the fraction of each alphabet with any evidence.
* `pool_profiles` refuses to mix synthetic and non-synthetic listeners.

### 3.3 `WordRisk`

`text`, `start_s`, `end_s`, `listener_risk`, `asr_confidence` (may be `None`),
`model_version`, `model_arm`, `decision`, `policy`, `contributions`, `meta`.

`decision` distinguishes `shown_high_risk` from `shown_low_asr_confidence`, so a word
surfaced because the recogniser was unsure can never be scored as a personalization hit.

---

## 4. Privacy contract

* Real profiles and raw calibration responses are written only under `private/`, which is
  git-ignored and enforced by `scripts/check_repo_hygiene.py` in CI.
* Listener ids are restricted to `[A-Za-z0-9._-]` — the validator rejects names.
* `ProfileStore.export()` returns everything stored for a listener; `delete()` removes it
  irreversibly and errors if there was nothing to delete, so a failed erasure is visible.
* The repository contains schemas and synthetic examples only.

---

## 5. Reproducibility contract

Every run appends to `experiments/registry.yaml`: `run_id`, git SHA, **git dirty flag**,
dependency lock hash, Python version, platform, seed list, the full config, data-manifest
content digests, artifact paths, metrics and status. Failed runs are recorded too.

`audire figures` regenerates every table and figure from `summary.json` alone, never from a
live model, so a figure cannot drift from the run that produced it.

---

## 6. Not yet implemented

These interfaces are specified and depended upon, but no implementation exists:

* **`audire.asr`** — the adapter is an empty package. The contract is: given a media path,
  return tokens with `start_s`, `end_s` and an optional per-token confidence, plus the
  backend name and model revision for provenance. Confidence must be returned as a
  separate field and never folded into risk.
* **`apps/api`, `apps/web`** — the FastAPI application and browser client.

Until these exist, AUDIRE can compute personalized word risk and render selective captions
from *supplied* word lists and timestamps, but cannot ingest audio end to end.
