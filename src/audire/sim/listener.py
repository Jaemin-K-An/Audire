"""Synthetic listener generation.

A synthetic listener consists of a :class:`~audire.profile.schema.HearingProfile` and a
*true* confusion structure. The true structure is what the trial generator samples from;
it is **not** handed to the estimator. What the estimator sees is a
:class:`~audire.confusion.profile.ConfusionProfile` reconstructed from a finite number of
observed calibration trials, exactly as for a real listener. That separation is what makes
parameter-recovery and calibration-length experiments meaningful.

Every object produced here carries ``is_synthetic=True``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from audire.confusion.profile import POSITIONS
from audire.hangul.inventory import NO_RESPONSE, Position, categories_for
from audire.profile.schema import (
    Audiogram,
    AudiogramPoint,
    Ear,
    EarProfile,
    HearingAidState,
    HearingProfile,
    LoudnessLevels,
    PIFunction,
    PIPoint,
    ProfileSource,
    SpeechScores,
)
from audire.sim.config import SimulationConfig
from audire.sim.similarity import similarity_matrix

FloatArray = npt.NDArray[np.float64]

#: Frequencies every synthetic audiogram carries.
SIM_FREQUENCIES: tuple[int, ...] = (250, 500, 1000, 2000, 3000, 4000, 6000, 8000)

#: Ordered severity strata, used to interpolate severity-dependent parameters.
STRATUM_ORDER: tuple[str, ...] = (
    "normal",
    "mild",
    "moderate",
    "moderately_severe",
    "severe",
    "profound",
)


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return float(np.log(p / (1 - p)))


def _expit(x: float | FloatArray) -> Any:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


@dataclass(slots=True)
class TrueConfusion:
    """The generative confusion structure of a synthetic listener (ground truth).

    Held separately from any estimated :class:`~audire.confusion.profile.ConfusionProfile`
    so that recovery error can be measured.
    """

    matrices: dict[Position, FloatArray]

    def row(self, position: Position, target: str) -> FloatArray:
        idx = categories_for(position, axis="target").index(target)
        row: FloatArray = self.matrices[position][idx]
        return row

    def p_correct(self, position: Position, target: str) -> float:
        targets = categories_for(position, axis="target")
        i = targets.index(target)
        return float(self.matrices[position][i, i])

    def diagonal(self, position: Position) -> FloatArray:
        m = self.matrices[position]
        n = m.shape[0]
        return m[np.arange(n), np.arange(n)]

    def sample(self, position: Position, target: str, rng: np.random.Generator) -> str:
        """Draw a perceived category for ``target``."""
        perceived = categories_for(position, axis="perceived")
        probs = self.row(position, target)
        return perceived[int(rng.choice(len(perceived), p=probs))]


@dataclass(slots=True)
class SyntheticListener:
    """A generated listener: clinical profile, latent ability and true confusion structure."""

    listener_id: str
    stratum: str
    #: Latent monosyllable accuracy on the logit scale; drives both WRS and the confusion
    #: diagonals, which is why those two are correlated but not identical.
    ability_logit: float
    hearing: HearingProfile
    true_confusion: TrueConfusion
    is_synthetic: bool = True
    generation: dict[str, Any] = field(default_factory=dict)

    @property
    def true_accuracy(self) -> float:
        return float(_expit(self.ability_logit))


# --------------------------------------------------------------------------- audiogram


def _draw_audiogram(
    rng: np.random.Generator, cfg: SimulationConfig, stratum: str, ear: Ear, asymmetry_db: float
) -> Audiogram:
    lo, hi = cfg.audiogram.pta_window_db[stratum]
    target_pta = float(rng.uniform(lo, hi)) + asymmetry_db
    slope = float(
        rng.normal(cfg.audiogram.slope_db_per_octave_mean, cfg.audiogram.slope_db_per_octave_sd)
    )

    # Build a shape anchored so that the PTA4 frequencies average to `target_pta`.
    octaves = {f: float(np.log2(f / 1000.0)) for f in SIM_FREQUENCIES}
    shape = {f: slope * octaves[f] for f in SIM_FREQUENCIES}
    pta_freqs = (500, 1000, 2000, 4000)
    offset = target_pta - float(np.mean([shape[f] for f in pta_freqs]))

    thresholds: dict[int, AudiogramPoint] = {}
    for f in SIM_FREQUENCIES:
        raw = shape[f] + offset + float(rng.normal(0.0, cfg.audiogram.threshold_noise_db_sd))
        # Audiometers step in 5 dB and cannot report below -10 dB HL.
        value = float(np.clip(round(raw / 5.0) * 5.0, -10.0, 120.0))
        no_response = value >= 120.0
        thresholds[f] = AudiogramPoint(db_hl=value, no_response=no_response)
    return Audiogram(ear=ear, thresholds=thresholds)


# --------------------------------------------------------------------------- confusion


def solve_position_error_mass(
    syllable_accuracy: float, multipliers: tuple[float, ...], *, tol: float = 1e-10
) -> float:
    """Return the per-position error mass ``e`` reproducing a whole-syllable accuracy.

    The reference literature reports *monosyllable* correct rates, but a confusion matrix
    is parameterised per position. Given multipliers ``m_p`` applied to the error mass at
    each position, the probability that a whole syllable is correct is

    .. math::  \\prod_p \\bigl(1 - e\\,m_p\\bigr)

    which is monotonically decreasing in ``e``, so a bisection recovers the unique ``e``
    that matches ``syllable_accuracy``. Without this step a syllable-level accuracy of
    0.817 would be applied *per position*, giving a whole-syllable accuracy near 0.55 and
    silently making every simulated listener far worse than the literature describes.
    """
    a = float(np.clip(syllable_accuracy, 1e-6, 1 - 1e-9))
    m = np.asarray(multipliers, dtype=np.float64)
    if np.any(m <= 0):
        raise ValueError("position difficulty multipliers must be positive")
    hi = float(min(1.0 / m.max(), 1.0)) * (1 - 1e-9)

    def syllable_accuracy_at(e: float) -> float:
        return float(np.prod(1.0 - e * m))

    lo = 0.0
    if syllable_accuracy_at(hi) > a:  # unreachable target: return the largest error mass
        return hi
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if syllable_accuracy_at(mid) > a:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def _stratum_index(stratum: str) -> float:
    """Position of ``stratum`` in [0, 1] from normal to profound."""
    try:
        i = STRATUM_ORDER.index(stratum)
    except ValueError:
        return 0.5
    return i / (len(STRATUM_ORDER) - 1)


def _build_true_confusion(
    rng: np.random.Generator, cfg: SimulationConfig, stratum: str, ability_logit: float
) -> TrueConfusion:
    """Construct one listener's true confusion matrices.

    For each target *i*:

    1. the diagonal mass is ``expit(ability_logit)`` adjusted by a position difficulty
       multiplier applied to the *error* mass;
    2. the remaining mass is spread over the other categories proportionally to
       ``exp(beta * similarity)``, so confusions concentrate within phonetic classes;
    3. a small share of the error mass goes to ``NO_RESPONSE``;
    4. the resulting row is perturbed by a Dirichlet draw so listeners with equal ability
       still differ in *which* confusions they make.
    """
    c = cfg.confusion
    s = _stratum_index(stratum)
    beta = c.similarity_beta_normal + s * (c.similarity_beta_severe - c.similarity_beta_normal)
    # `ability_logit` expresses whole-MONOSYLLABLE accuracy, because that is what the
    # reference literature reports. Convert it to the per-position error mass that
    # reproduces it once the position-difficulty multipliers are applied.
    base_error = solve_position_error_mass(
        float(_expit(ability_logit)),
        tuple(c.position_difficulty[p.value] for p in POSITIONS),
    )

    matrices: dict[Position, FloatArray] = {}
    for position in POSITIONS:
        targets = categories_for(position, axis="target")
        perceived = categories_for(position, axis="perceived")
        sim = similarity_matrix(position)
        n_t, n_p = len(targets), len(perceived)
        no_resp_col = perceived.index(NO_RESPONSE)

        rows = np.zeros((n_t, n_p), dtype=np.float64)
        difficulty = c.position_difficulty[position.value]
        error_mass = float(np.clip(base_error * difficulty, 1e-4, 0.98))
        diag = 1.0 - error_mass

        for i in range(n_t):
            # Similarity-weighted spread over the substitution candidates. The diagonal
            # and the no-response column are excluded here and set explicitly below.
            weights = np.exp(beta * sim[i])
            weights[i] = 0.0
            weights[no_resp_col] = 0.0
            total = float(weights.sum())

            row = np.zeros(n_p, dtype=np.float64)
            row[i] = diag
            row[no_resp_col] = error_mass * c.no_response_share
            if total > 0:
                spread = error_mass * (1.0 - c.no_response_share)
                row += (weights / total) * spread
            else:  # pragma: no cover - every alphabet has >=2 substitution candidates
                row[i] += error_mass * (1.0 - c.no_response_share)

            row /= row.sum()
            rows[i] = rng.dirichlet(np.maximum(row * c.dirichlet_concentration, 1e-3))
        matrices[position] = rows
    return TrueConfusion(matrices=matrices)


# --------------------------------------------------------------------------- speech


def _draw_speech(
    rng: np.random.Generator,
    cfg: SimulationConfig,
    ear: Ear,
    pta: float | None,
    ability_logit: float,
) -> tuple[SpeechScores, PIFunction | None, LoudnessLevels | None]:
    sp = cfg.speech
    accuracy = float(_expit(ability_logit))

    srt = None
    if pta is not None:
        srt = float(
            np.clip(
                round((pta + rng.normal(sp.srt_minus_pta_mean_db, sp.srt_minus_pta_sd_db)) / 5.0)
                * 5.0,
                -10.0,
                120.0,
            )
        )

    # A single-level WRS is a binomial estimate over a finite word list, not the latent
    # ability itself. This is what makes WRS a *noisy* global summary.
    n_words = sp.wrs_n_words
    correct = int(rng.binomial(n_words, accuracy))
    wrs = 100.0 * correct / n_words
    presentation = float(np.clip((srt if srt is not None else 40.0) + 30.0, 0.0, 110.0))

    speech = SpeechScores(
        ear=ear,
        srt_db_hl=srt,
        wrs_percent=wrs,
        wrs_presentation_level_db_hl=presentation,
        wrs_word_list=f"synthetic-{n_words}-item",
        wrs_n_words=n_words,
    )

    pi: PIFunction | None = None
    if rng.random() < sp.p_has_pi_function:
        peak = float(
            np.clip(wrs + rng.normal(sp.pbmax_increment_mean, sp.pbmax_increment_sd), 0.0, 100.0)
        )
        levels = [presentation - 20.0, presentation - 10.0, presentation, presentation + 10.0]
        scores = [
            float(np.clip(peak - 25.0, 0.0, 100.0)),
            float(np.clip(peak - 8.0, 0.0, 100.0)),
            peak,
            float(np.clip(peak - rng.uniform(0.0, 12.0), 0.0, 100.0)),
        ]
        pi = PIFunction(
            ear=ear,
            points=[
                PIPoint(level_db_hl=float(np.clip(lv, -10.0, 130.0)), score_percent=sc)
                for lv, sc in zip(levels, scores, strict=True)
            ],
            word_list=f"synthetic-{n_words}-item",
        )

    loud: LoudnessLevels | None = None
    if rng.random() < sp.p_has_loudness and pta is not None:
        mcl = float(np.clip(pta + rng.normal(sp.mcl_above_pta_db, sp.mcl_sd_db), 0.0, 110.0))
        dr = float(max(5.0, rng.normal(sp.dynamic_range_db_mean, sp.dynamic_range_db_sd)))
        loud = LoudnessLevels(
            ear=ear, mcl_db_hl=mcl, ucl_db_hl=float(np.clip(mcl + dr, mcl, 130.0))
        )

    return speech, pi, loud


# --------------------------------------------------------------------------- entry point


def generate_listener(
    rng: np.random.Generator, cfg: SimulationConfig, index: int
) -> SyntheticListener:
    """Generate one synthetic listener."""
    strata = cfg.severity.normalised()
    names = sorted(strata)
    stratum = str(rng.choice(names, p=[strata[n] for n in names]))

    mean_accuracy = cfg.speech.accuracy_by_stratum[stratum]
    ability_logit = float(rng.normal(_logit(mean_accuracy), cfg.speech.ability_sd_logit))

    asym = float(rng.normal(0.0, cfg.audiogram.interaural_asymmetry_db_sd))
    ears: dict[Ear, EarProfile] = {}
    for ear, offset in ((Ear.RIGHT, -abs(asym) / 2), (Ear.LEFT, abs(asym) / 2)):
        audiogram = _draw_audiogram(rng, cfg, stratum, ear, offset)
        pta = audiogram.pta(cfg.audiogram.pta_method)
        speech, pi, loud = _draw_speech(rng, cfg, ear, pta, ability_logit)
        ears[ear] = EarProfile(
            ear=ear, audiogram=audiogram, speech=speech, pi_function=pi, loudness=loud
        )

    listener_id = f"SYN{index:04d}"
    hearing = HearingProfile(
        listener_id=listener_id,
        source=ProfileSource.SYNTHETIC,
        is_synthetic=True,
        left=ears[Ear.LEFT],
        right=ears[Ear.RIGHT],
        hearing_aid_state=HearingAidState.UNAIDED,
        pta_method=cfg.audiogram.pta_method,
        severity_scheme=cfg.severity.scheme,
        provenance={
            "generator": "audire.sim.listener.generate_listener",
            "config_name": cfg.name,
            "stratum": stratum,
            "is_synthetic": True,
        },
    )

    return SyntheticListener(
        listener_id=listener_id,
        stratum=stratum,
        ability_logit=ability_logit,
        hearing=hearing,
        true_confusion=_build_true_confusion(rng, cfg, stratum, ability_logit),
        generation={
            "stratum": stratum,
            "true_accuracy": float(_expit(ability_logit)),
            "config_name": cfg.name,
        },
    )


def generate_cohort(cfg: SimulationConfig, seed: int) -> list[SyntheticListener]:
    """Generate ``cfg.n_listeners`` synthetic listeners from one seed.

    Deterministic: the same ``(cfg, seed)`` always produces the same cohort.
    """
    rng = np.random.default_rng(seed)
    return [generate_listener(rng, cfg, i) for i in range(cfg.n_listeners)]
