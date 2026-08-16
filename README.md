# AUDIRE

**Personalized Prediction of Korean Word Misrecognition for Selective Captioning**
개인 어음인지 프로파일 기반 한국어 오청 예측 및 선택 자막

AUDIRE estimates, for a specific listener, the probability that a specific Korean word will be
misheard —

```
P(word misheard | listener, word, acoustic context)
```

— from a personalized speech-perception profile (audiogram/PTA, SRT, WRS, optionally
PBmax/MCL/UCL, and an individual onset/nucleus/coda phoneme confusion matrix), and uses that
probability to caption only the words that need captioning.

The central hypothesis is that **WRS and the confusion matrix carry different information**:
WRS says how much speech a listener misrecognizes globally, the confusion matrix says what
they confuse locally. AUDIRE keeps them structurally separate so that the question can be
tested rather than assumed.

> **AUDIRE is research and accessibility software. It is not a medical device.** It does not
> diagnose hearing loss and does not recommend treatment.

---

## Status

| Gate | Scope | Status |
|---|---|---|
| G0 | Evidence, licences, provenance, data layer | complete |
| G1 | Hangul engine, response parser, confusion matrices | complete |
| G2 | HearingProfile + synthetic simulation | in progress |
| G3 | Risk models, calibration, listener-level evaluation | pending |
| G4 | Selective caption engine + exports | pending |
| G5 | ASR end-to-end | pending |
| G6 | Complete user system | pending |
| G7 | Validation and sensitivity | pending |
| G8 | Reproducible release | pending |

`docs/TASKS.md` is the live ledger. No gate below G8 may be described as a finished product.

---

## Quick start

```bash
make bootstrap
```

```bash
make test
```

```bash
make data
```

`make data` fetches every source that does not require an outstanding human step. The primary
Korean monosyllable corpus is **CC BY-NC-ND 4.0** and its dataset card asks users to inform the
creator of their intended use and scope before using it. AUDIRE will not send that message for
you; after you have handled it yourself:

```bash
AUDIRE_PRIMARY_DATA_USE_NOTIFIED=1 make data-primary
```

AUDIRE works without that corpus — see *Stimulus sources* below.

---

## Repository map

```
src/audire/
  hangul/       modern-Hangul decomposition/recomposition, jamo inventories
  confusion/    response parsing, error taxonomy, per-position confusion matrices
  profile/      HearingProfile schema (PTA/SRT/WRS/PBmax/MCL/UCL) with explicit missingness
  risk/         baselines, learned probabilistic models, probability calibration
  asr/          replaceable Korean ASR adapter with word timestamps
  caption/      risk ranking, threshold/budget policies, SRT/ASS/JSON export
  sim/          synthetic listener and trial generators (every row is_synthetic=true)
  eval/         metrics, bootstrap CIs, ablations, figures
  data/         source registry, fetchers, manifests, stimulus catalogues
apps/api/       FastAPI application
apps/web/       browser client (calls the real pipeline, no mocks)
experiments/    preregistered configs; artifacts are regenerated, never committed
data/manifests/ provenance for every acquired byte
docs/           research plan, decisions, risks, claims, results
```

---

## Stimulus sources

Calibration needs Korean monosyllables. Two sources are supported and the system is fully
functional with either:

**`builtin`** — a deterministic phoneme-balanced design generated from the Hangul inventory
itself: 19 onsets × 21 nuclei × 8 coda categories = 3,192 combinations, enumerated in a coprime
cyclic order so the first *k* items are the balanced *k*-item calibration list. No third-party
licence. The browser speaks the syllables with its Korean speech synthesiser, which is **not
level-calibrated** — usable for research and accessibility, never for clinical measurement.

**`kmsp`** — the Korean Monosyllabic Speech Perception Test Dataset: 726 recorded utterances,
2 speakers, 16 kHz, with duration/pitch/formant metadata. Preferred when available. Neither its
audio nor its metadata is ever committed or redistributed (ND clause).

---

## Data, privacy and licences

* `data/raw/`, `data/processed/` and `private/` are git-ignored. **No participant-level
  audiogram, SRT, WRS or trial response may ever be committed.**
* Every acquired dataset gets a manifest in `data/manifests/` with a SHA-256 per file, the
  pinned revision, the licence and the retrieval time. `make data-verify` re-checks them.
* Every source in `data/sources.yaml` declares its permitted and prohibited uses, and
  `Source.assert_permits()` is a tripwire against adding a prohibited use later.
* Licences were read from the live source APIs on 2026-08-16, not from prose — including two
  documented discrepancies (see `docs/DECISIONS.md` ADR-0003).

---

## Reproducibility

```bash
make reproduce
```

runs simulate → train → eval → caption-eval → sensitivity → figures. Every experiment records
its git SHA, config, seed list, dependency lock hash, data manifest ids, metrics and logs into
`experiments/registry.yaml`, so every reported number can be traced to the bytes and code that
produced it.

---

## Reading order

1. `AGENTS.md` — the execution contract
2. `docs/RESEARCH_PLAN.md` — questions, hypotheses, falsification criteria
3. `docs/DECISIONS.md` — why the system is built the way it is
4. `docs/RISK_REGISTER.md` — what could invalidate the claims
5. `docs/RESULTS.md` — only reproduced numbers, including negative results

## Licence

Apache-2.0 for the code. Third-party data retains its own licence; see `data/sources.yaml`.
