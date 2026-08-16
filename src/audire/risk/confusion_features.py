"""Phase D17 — richer, still-controlled representation of the listener confusion profile.

Why the existing block is not enough
------------------------------------
``audire.risk.features.confusion_features`` compresses ``C_u`` into 16 aggregates
(``R_phon``, mean/min ``p_correct``, per-position means, entropy, evidence counts). Those
summarise *how bad this listener is overall*, which is largely redundant with WRS. What a
per-listener caption budget actually needs is *which particular words are unusually risky
for this particular listener* — and that signal lives in the interaction

    "this listener is weak on class C"  ×  "this word contains class C".

An aggregate mean over the word cannot express it: two words with the same mean
``p_correct`` differ entirely if one of them happens to hit the listener's weak class.

Controlled dimensionality
-------------------------
The mission explicitly warns against flattening the whole matrix into hundreds of
uncontrolled dimensions. This module therefore does **not** emit per-phoneme columns.
It emits, per phonological *class*, one interaction term:

    weakness(listener, class) x share(word, class)

with the class inventory fixed and small (6 onset manners, 6 onset phonations,
4 nucleus shapes, 8 coda surfaces = 24 interaction terms), plus a bounded set of
distributional and uncertainty summaries — 59 columns in total, against ~2,000 word trials
per cohort. The block is additive to the existing one and is exposed as separate ablation
arms, so the old arm remains available for comparison.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

import numpy as np

from audire.confusion.grouping import (
    NUCLEUS_SHAPE,
    ONSET_MANNER,
    ONSET_PHONATION,
    neutralise_coda,
)
from audire.confusion.profile import POSITIONS, ConfusionProfile
from audire.hangul.inventory import NO_CODA, Position, categories_for
from audire.risk.features import _EPS, PhonemeRisk, phoneme_risks, word_syllables


#: Class inventories, derived from the phonological grouping tables rather than restated
#: here. Hand-listing them is how a class goes missing: an earlier draft of this module
#: omitted the sonorant phonations, which silently excluded every ``ㄴ/ㄹ/ㅁ/ㅇ`` onset from
#: the feature vector. Deriving them keeps the inventory small and fixed — it is exactly
#: the enum, whose size is a deliberate design decision — while making it impossible for
#: the feature code and the grouping tables to disagree.
def _values(table: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(sorted({member.value for member in table.values()}))


_ONSET_MANNERS: Final[tuple[str, ...]] = _values(ONSET_MANNER)
_ONSET_PHONATIONS: Final[tuple[str, ...]] = _values(ONSET_PHONATION)
_NUCLEUS_SHAPES: Final[tuple[str, ...]] = _values(NUCLEUS_SHAPE)
#: The seven neutralised coda surfaces plus "no coda" (음절의 끝소리 규칙).
_CODA_SURFACES: Final[tuple[str, ...]] = (NO_CODA, "ㄱ", "ㄴ", "ㄷ", "ㄹ", "ㅁ", "ㅂ", "ㅇ")


def _onset_class(jamo: str, kind: str) -> str:
    return ONSET_MANNER[jamo].value if kind == "manner" else ONSET_PHONATION[jamo].value


def listener_global_error(profile: ConfusionProfile) -> float:
    """The listener's overall observed error rate across all positions.

    Used as the no-evidence fallback below. ``0.5`` when the listener has no observations
    at all — maximally uninformative rather than maximally alarming.
    """
    total = correct = 0.0
    for position in POSITIONS:
        matrix = profile.matrix(position)
        counts = matrix.counts
        total += float(counts.sum())
        for i, target in enumerate(matrix.target_labels):
            if target in matrix.perceived_labels:
                correct += float(counts[i, matrix.perceived_labels.index(target)])
    return 1.0 - correct / total if total > 0 else 0.5


def listener_class_weakness(profile: ConfusionProfile) -> dict[str, float]:
    """Per-class mean error probability for this listener, independent of any word.

    This is the "this listener has this weakness" half of the interaction, computed once
    per listener over the whole confusion profile.

    Absence of evidence is not evidence of weakness
    -----------------------------------------------
    The estimate is weighted by **observations only**, and a class with no observations
    falls back to the listener's global error rate. An earlier version averaged the
    smoothed ``p_correct`` over every class member with a ``+1`` floor, which looks
    conservative but is not: under a uniform prior an unobserved target has
    ``p_correct = 1/n_labels ≈ 0.05``, so a class nobody was ever tested on scored as a
    *worse* weakness (0.95) than one they demonstrably failed (0.84). With a 25-trial
    calibration most classes are unobserved, so that artifact would have driven the
    interaction terms more strongly than the real data, and would have ranked a listener
    with no stop evidence above one measurably bad at stops.

    Shrinking toward the population (:mod:`audire.risk.hierarchical`) is the principled
    treatment of the same problem and composes with this: once a profile is shrunk, the
    unobserved rows carry population information and the fallback is rarely reached.
    """
    out: dict[str, float] = {}
    fallback = listener_global_error(profile)

    def _weighted(position: Position, members: list[str]) -> float:
        matrix = profile.matrix(position)
        weights, errors = [], []
        for target in members:
            n = matrix.n_observations(target)
            if n == 0:
                continue
            weights.append(float(n))
            errors.append(1.0 - matrix.p_correct(target))
        if not weights:
            return fallback
        w = np.asarray(weights, dtype=np.float64)
        e = np.asarray(errors, dtype=np.float64)
        return float((w * e).sum() / w.sum())

    onset_labels = list(categories_for(Position.ONSET, axis="target"))
    for manner in _ONSET_MANNERS:
        members = [j for j in onset_labels if _onset_class(j, "manner") == manner]
        out[f"wk_onset_manner_{manner}"] = _weighted(Position.ONSET, members)
    for phonation in _ONSET_PHONATIONS:
        members = [j for j in onset_labels if _onset_class(j, "phonation") == phonation]
        out[f"wk_onset_phon_{phonation}"] = _weighted(Position.ONSET, members)

    nucleus_labels = list(categories_for(Position.NUCLEUS, axis="target"))
    for shape in _NUCLEUS_SHAPES:
        members = [j for j in nucleus_labels if NUCLEUS_SHAPE[j].value == shape]
        out[f"wk_nucleus_{shape}"] = _weighted(Position.NUCLEUS, members)

    coda_labels = list(categories_for(Position.CODA, axis="target"))
    for surface in _CODA_SURFACES:
        members = [
            j
            for j in coda_labels
            if (j == NO_CODA and surface == NO_CODA)
            or (j != NO_CODA and neutralise_coda(j) == surface)
        ]
        key = "none" if surface == NO_CODA else surface
        out[f"wk_coda_{key}"] = _weighted(Position.CODA, members)

    return out


def word_class_share(word: str) -> dict[str, float]:
    """Fraction of the word's segments belonging to each class.

    The "this word contains class C" half of the interaction.
    """
    syls = word_syllables(word)
    shares = dict.fromkeys(
        [f"sh_onset_manner_{m}" for m in _ONSET_MANNERS]
        + [f"sh_onset_phon_{p}" for p in _ONSET_PHONATIONS]
        + [f"sh_nucleus_{s}" for s in _NUCLEUS_SHAPES]
        + [f"sh_coda_{'none' if c == NO_CODA else c}" for c in _CODA_SURFACES],
        0.0,
    )
    if not syls:
        return shares
    n = float(len(syls))
    for syl in syls:
        shares[f"sh_onset_manner_{_onset_class(syl.onset, 'manner')}"] += 1.0 / n
        shares[f"sh_onset_phon_{_onset_class(syl.onset, 'phonation')}"] += 1.0 / n
        shares[f"sh_nucleus_{NUCLEUS_SHAPE[syl.nucleus].value}"] += 1.0 / n
        surface = "none" if syl.coda == NO_CODA else neutralise_coda(syl.coda)
        shares[f"sh_coda_{surface}"] += 1.0 / n
    return shares


def _off_diagonal_structure(
    risks: list[PhonemeRisk], profile: ConfusionProfile
) -> dict[str, float]:
    """Where a listener's error mass goes, not merely how much of it there is.

    Two listeners with identical ``p_correct`` differ in usefulness: one may spread error
    mass thinly over many categories (hard to predict) while the other concentrates it on
    one confusable neighbour (highly predictable).
    """
    top1, within, across = [], [], []
    for r in risks:
        matrix = profile.matrix(r.position)
        row = matrix.probabilities()[matrix.target_labels.index(r.target)]
        labels = matrix.perceived_labels
        error_mass = max(1.0 - r.p_correct, _EPS)

        off = [(labels[j], float(row[j])) for j in range(len(labels)) if labels[j] != r.target]
        off.sort(key=lambda kv: -kv[1])
        top1.append(off[0][1] / error_mass if off else 0.0)

        same = 0.0
        for label, prob in off:
            if _same_class(r.position, r.target, label):
                same += prob
        within.append(same / error_mass)
        across.append((error_mass - same) / error_mass)

    if not risks:
        return {"x2_top1_share": 0.0, "x2_within_class_share": 0.0, "x2_across_class_share": 0.0}
    return {
        # Concentration of the error mass on its single most likely wrong answer.
        "x2_top1_share": float(np.mean(top1)),
        "x2_within_class_share": float(np.mean(within)),
        "x2_across_class_share": float(np.mean(across)),
    }


def _same_class(position: Position, target: str, other: str) -> bool:
    if other in ("?",):
        return False
    if position is Position.ONSET:
        return ONSET_MANNER.get(other) is ONSET_MANNER.get(target)
    if position is Position.NUCLEUS:
        return NUCLEUS_SHAPE.get(other) is NUCLEUS_SHAPE.get(target)
    if target == NO_CODA or other == NO_CODA:
        return target == other
    return neutralise_coda(other) == neutralise_coda(target)


def _posterior_uncertainty(risks: list[PhonemeRisk], profile: ConfusionProfile) -> dict[str, float]:
    """Beta posterior spread of each diagonal, given counts and the smoothing prior.

    Uncertainty is never concealed: a diagonal of 0.9 from 2 trials and one from 200 are
    different evidence, and the model is given the means to treat them differently.
    """
    if not risks:
        return {"x2_post_sd_mean": 0.0, "x2_post_sd_max": 0.0, "x2_frac_thin_evidence": 1.0}
    sds = []
    for r in risks:
        alpha = profile.matrix(r.position).smoothing.alpha
        # Effective sample size behind this row's diagonal.
        n_eff = r.n_observations + alpha
        p = min(max(r.p_correct, _EPS), 1 - _EPS)
        sds.append(float(np.sqrt(p * (1 - p) / (n_eff + 1.0))))
    return {
        "x2_post_sd_mean": float(np.mean(sds)),
        "x2_post_sd_max": float(np.max(sds)),
        "x2_frac_thin_evidence": float(np.mean([r.n_observations < 3 for r in risks])),
    }


def confusion_rich_features(word: str, profile: ConfusionProfile) -> dict[str, float]:
    """The Phase-D confusion block: interactions plus error-structure and uncertainty.

    Emitted **in addition to** the original block, never instead of it, so that the
    original arm stays available as a comparison.
    """
    risks = phoneme_risks(word, profile, top_k=0)
    weakness = listener_class_weakness(profile)
    share = word_class_share(word)

    out: dict[str, float] = {}
    # The central interaction. One term per class: does this word actually load on a
    # class this listener is weak at?
    for wk_key, wk_value in weakness.items():
        class_key = wk_key.removeprefix("wk_")
        out[f"ix_{class_key}"] = wk_value * share.get(f"sh_{class_key}", 0.0)

    # The word's own class profile, so the model can separate "word loads on class C"
    # from "listener is weak at C" from their product.
    out.update({f"w2_{k.removeprefix('sh_')}": v for k, v in share.items()})

    out.update(_off_diagonal_structure(risks, profile))
    out.update(_posterior_uncertainty(risks, profile))

    if risks:
        p = np.array([r.p_correct for r in risks], dtype=np.float64)
        # Worst-segment emphasis: a word is misheard if *any* segment fails, so the tail
        # matters more than the mean.
        out["x2_worst_two_mean"] = float(np.mean(np.sort(p)[:2]))
        out["x2_p_spread"] = float(p.max() - p.min())
        for position in POSITIONS:
            vals = [r.p_correct for r in risks if r.position is position]
            out[f"x2_min_p_{position.value}"] = float(np.min(vals)) if vals else 1.0
    else:
        out["x2_worst_two_mean"] = 1.0
        out["x2_p_spread"] = 0.0
        for position in POSITIONS:
            out[f"x2_min_p_{position.value}"] = 1.0
    return out


def n_rich_features() -> int:
    """Dimensionality of the block, for documenting the regularisation burden."""
    from audire.confusion import CalibrationTrial

    probe = ConfusionProfile.from_trials(
        "probe",
        [CalibrationTrial(stimulus_id="s0", target="각", response="각")],
        is_synthetic=True,
    )
    return len(confusion_rich_features("가족", probe))
