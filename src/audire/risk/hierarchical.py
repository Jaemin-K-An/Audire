"""Phase D18 — hierarchical shrinkage of sparse listener confusion profiles.

A 25-trial calibration cannot observe most of a 19x20 onset matrix. With a uniform prior,
every unobserved row is the same flat distribution, which throws away everything the
population already tells us about which confusions are common. Shrinking each listener
toward a **group prior** estimated from other listeners uses that information while still
letting a well-observed phoneme move toward its individual estimate:

    posterior_u  =  (counts_u + alpha * prior_group) / (n_u + alpha)

``alpha`` is the total pseudo-count, so a listener with many observations for a phoneme is
barely moved while an unobserved one sits at the prior. Raw counts, the prior, the
posterior and the evidence amount all remain separately inspectable — nothing conceals
uncertainty.

Leakage
-------
This is the dangerous part. A group prior fitted over **all** listeners, including the
held-out ones, leaks their labels into the training representation and inflates every
held-out metric. The prior must be estimated from training listeners only, refitted for
every fold. :func:`fit_group_prior` therefore takes an explicit list of profiles and the
evaluation harness refits it per fold; :func:`apply_group_prior` never looks at anything
beyond the profiles it is handed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from audire.confusion.matrix import ConfusionMatrix, SmoothingSpec
from audire.confusion.profile import POSITIONS, ConfusionProfile
from audire.hangul.inventory import Position

FloatArray = npt.NDArray[np.float64]

#: Default total pseudo-count for shrinkage toward the group. Larger than the uniform
#: default because a group prior carries real information and deserves more weight than a
#: flat one, but small enough that ~50 observations dominate it.
DEFAULT_GROUP_ALPHA: float = 5.0


@dataclass(frozen=True, slots=True)
class GroupPrior:
    """A population confusion prior, plus the provenance needed to audit it."""

    matrices: dict[Position, FloatArray]
    #: Listener ids that contributed. Recorded so a leak is visible in the artifact.
    fitted_from: tuple[str, ...]
    n_listeners: int
    n_trials: int
    is_synthetic: bool

    def smoothing_for(self, position: Position, alpha: float) -> SmoothingSpec:
        return SmoothingSpec(alpha=alpha, kind="explicit", prior=self.matrices[position])

    def describe(self) -> dict[str, Any]:
        return {
            "n_listeners": self.n_listeners,
            "n_trials": self.n_trials,
            "is_synthetic": self.is_synthetic,
            "fitted_from": list(self.fitted_from),
        }


def fit_group_prior(profiles: Sequence[ConfusionProfile]) -> GroupPrior:
    """Estimate a population prior by pooling the given listeners' raw counts.

    Only the profiles passed in contribute. The caller — not this function — is
    responsible for passing training listeners only; the harness enforces that per fold
    and :attr:`GroupPrior.fitted_from` records who was included so a leak is auditable
    after the fact rather than invisible.

    Raises
    ------
    ValueError
        If no profiles are given, or if synthetic and observed listeners are mixed, which
        would launder simulated evidence into a real listener's prior.
    """
    if not profiles:
        raise ValueError("집단 사전분포를 빈 목록에서 추정할 수 없습니다")
    synth = {p.is_synthetic for p in profiles}
    if len(synth) > 1:
        raise ValueError("합성(synthetic)과 실측 청취자를 섞어 집단 사전분포를 만들 수 없습니다")

    matrices: dict[Position, FloatArray] = {}
    for position in POSITIONS:
        pooled = ConfusionMatrix.empty(position)
        for profile in profiles:
            pooled.counts += profile.matrix(position).counts
        # A uniform-smoothed pooled matrix: even the population has unobserved rows when
        # the cohort is small, and those must stay proper distributions.
        matrices[position] = pooled.probabilities()

    return GroupPrior(
        matrices=matrices,
        fitted_from=tuple(sorted(p.listener_id for p in profiles)),
        n_listeners=len(profiles),
        n_trials=sum(p.n_trials for p in profiles),
        is_synthetic=synth.pop(),
    )


def apply_group_prior(
    profile: ConfusionProfile, prior: GroupPrior, *, alpha: float = DEFAULT_GROUP_ALPHA
) -> ConfusionProfile:
    """Return a copy of ``profile`` shrunk toward ``prior``.

    The listener's raw counts are untouched; only the smoothing specification changes, so
    ``n_observations`` still reports the real evidence and the posterior can always be
    compared against both the counts and the prior it came from.
    """
    shrunk = ConfusionProfile(
        listener_id=profile.listener_id,
        matrices={
            position: ConfusionMatrix(
                position=position,
                counts=profile.matrix(position).counts.copy(),
                smoothing=prior.smoothing_for(position, alpha),
            )
            for position in POSITIONS
        },
        is_synthetic=profile.is_synthetic,
        n_trials=profile.n_trials,
        n_unusable_responses=profile.n_unusable_responses,
        created_at=profile.created_at,
        provenance={
            **profile.provenance,
            "shrinkage": {
                "alpha": alpha,
                "prior": prior.describe(),
            },
        },
    )
    return shrunk


def shrinkage_report(
    original: ConfusionProfile, shrunk: ConfusionProfile, position: Position
) -> dict[str, Any]:
    """How far each row moved, and where the movement went.

    Used by the tests and by the results write-up to check the qualitative requirement:
    unobserved phonemes should rely strongly on the prior, well-observed ones should not.
    """
    matrix_o, matrix_s = original.matrix(position), shrunk.matrix(position)
    rows: list[dict[str, Any]] = []
    for target in matrix_o.target_labels:
        n = matrix_o.n_observations(target)
        rows.append(
            {
                "target": target,
                "n_observations": n,
                "p_correct_uniform_prior": matrix_o.p_correct(target),
                "p_correct_group_prior": matrix_s.p_correct(target),
                "moved": abs(matrix_s.p_correct(target) - matrix_o.p_correct(target)),
            }
        )
    unobserved = [r["moved"] for r in rows if r["n_observations"] == 0]
    observed = [r["moved"] for r in rows if r["n_observations"] >= 10]
    return {
        "rows": rows,
        "mean_move_unobserved": float(np.mean(unobserved)) if unobserved else float("nan"),
        "mean_move_well_observed": float(np.mean(observed)) if observed else float("nan"),
    }
