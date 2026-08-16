"""RQ2 and RQ3 — the selective-caption budget study.

RQ2 asks: at a **matched caption ratio**, does personalized risk ranking capture more
actually-misheard words than the non-personalized alternatives? The primary outcome is
therefore *misheard-word recall at a fixed budget*, computed identically for every
strategy so that the only difference is the ranking.

Strategies compared (docs/RESEARCH_PLAN.md B0-B4):

=========================  ===============================================
strategy                   ranking
=========================  ===============================================
``random``                 uniform random, seeded (B1)
``word_length``            longer words first — a non-personalized lexical
                           heuristic that needs no listener data (B2)
``model:<arm>``            predicted listener risk from an ablation arm
=========================  ===============================================

RQ3 asks whether a **per-listener** threshold beats one global threshold at the same
overall caption volume. Because a budget policy is per-listener by construction, the
comparison is run as: personalized quantile threshold vs a single pooled threshold, both
targeting the same overall ratio.

The whole Pareto frontier is reported. A single operating point is never presented alone.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from audire.caption.policy import STANDARD_BUDGETS, global_threshold, personalized_threshold
from audire.config.logging import get_logger
from audire.eval.ablation import ArmResult
from audire.eval.bootstrap import Interval, bootstrap_metric

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
StrArray = npt.NDArray[np.str_]

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class BudgetPoint:
    """One point on the caption Pareto frontier."""

    strategy: str
    budget: float
    #: Fraction of words actually shown. May differ slightly from ``budget`` because the
    #: per-listener count is rounded.
    achieved_ratio: float
    caption_reduction_ratio: float
    #: The primary outcome: fraction of truly-misheard words that were captioned.
    misheard_recall: float
    #: Of the words shown, what fraction were truly misheard.
    precision: float
    #: Misheard words that were NOT captioned, as a fraction of all words. The quantity a
    #: listener actually experiences as a failure.
    missed_rate: float
    n_words: int
    n_misheard: int
    #: Whether the budget was applied within each listener (deployment behaviour) or
    #: pooled across all listeners.
    per_listener: bool = True
    recall_ci: Interval | None = None
    #: Spread of per-listener recall. Aggregate recall hides who is served badly.
    recall_min: float = float("nan")
    recall_median: float = float("nan")
    recall_max: float = float("nan")
    #: Absolute caption counts. A ratio alone hides whether "matched volume" is真.
    achieved_count: int = -1
    #: Items sharing the boundary score, i.e. how much of the selection was arbitrary.
    n_ties_at_boundary: int = 0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["recall_ci"] = self.recall_ci.to_dict() if self.recall_ci else None
        return d


# --------------------------------------------------------------------------- rankings


def _random_scores(n: int, seed: int) -> FloatArray:
    return np.random.default_rng(seed).random(n)


def _word_length_scores(words: list[str]) -> FloatArray:
    """Longer words first. A non-personalized lexical heuristic (baseline B2).

    Deliberately weak but not trivial: word length correlates with segmental risk, so it
    is a fairer non-personalized comparator than random selection alone.
    """
    lengths = np.array([len(w) for w in words], dtype=np.float64)
    if lengths.size == 0:
        return lengths
    span = lengths.max() - lengths.min()
    return (lengths - lengths.min()) / span if span > 0 else np.full_like(lengths, 0.5)


def select_budget(
    scores: FloatArray,
    groups: StrArray,
    budget: float,
    *,
    per_listener: bool = True,
    tie_seed: int = 0,
) -> npt.NDArray[np.bool_]:
    """Select the top ``budget`` fraction of words by ``scores``.

    ``per_listener=True`` (the default and the primary analysis) applies the budget
    **within each listener**. This is what an actual deployment does: one listener watches
    one video and sees a fixed proportion of *their own* words captioned. It has an
    important consequence for RQ2 that the results must state plainly — features that are
    constant within a listener (PTA, SRT, WRS) cannot change the within-listener ranking
    at all, so under a per-listener budget only word-level and confusion-derived features
    can contribute.

    ``per_listener=False`` pools every listener and spends the budget globally. Listener-
    level features do move this ranking, because the model can direct captions toward the
    listeners who mishear most. It is reported as a contrast so that the mechanism behind
    the per-listener result is visible rather than merely asserted.
    """
    chosen = np.zeros(scores.shape, dtype=bool)
    # A deterministic tiny jitter breaks ties reproducibly instead of favouring
    # whichever word happens to come first in the array.
    jitter = np.random.default_rng(tie_seed).random(scores.shape) * 1e-9
    blocks = (
        [np.flatnonzero(groups == g) for g in np.unique(groups)]
        if per_listener
        else [np.arange(scores.size)]
    )
    for idx in blocks:
        n_show = round(budget * idx.size)
        if n_show <= 0:
            continue
        order = idx[np.argsort(-(scores[idx] + jitter[idx]), kind="stable")]
        chosen[order[:n_show]] = True
    return chosen


def select_exact_count(
    scores: FloatArray, n: int, *, tie_break: str = "index"
) -> npt.NDArray[np.bool_]:
    """Select exactly ``n`` items by descending score, with deterministic tie breaking.

    A threshold comparison (``p > tau``) only delivers the requested caption volume when
    the scores are essentially continuous. With heavy ties — which is exactly what the
    ``word_context_only`` arm produces, because identical words carry identical features
    and no listener information — a threshold either includes or excludes a whole tied
    block, so the achieved volume can miss the target badly. RQ3 claims the two arms are
    compared *at equal caption volume*, so that claim has to be enforced rather than
    hoped for.

    Ties are broken by ascending original index, which is stable and reproducible.
    """
    if n < 0 or n > scores.size:
        raise ValueError(f"선택 개수(count) {n} 이 0..{scores.size} 범위를 벗어났습니다")
    chosen = np.zeros(scores.shape, dtype=bool)
    if n == 0:
        return chosen
    if tie_break != "index":  # pragma: no cover - single supported policy today
        raise ValueError(f"unsupported tie_break {tie_break!r}")
    # `-scores` with a stable sort makes ascending index the tie-breaker.
    order = np.argsort(-scores, kind="stable")
    chosen[order[:n]] = True
    return chosen


def count_boundary_ties(scores: FloatArray, n: int) -> int:
    """How many items share the score of the last selected item.

    Reported so that a reader can see whether the exact-count selection had to make an
    arbitrary choice, rather than discovering it from a volume mismatch later.
    """
    if n <= 0 or n >= scores.size:
        return 0
    order = np.argsort(-scores, kind="stable")
    boundary = scores[order[n - 1]]
    return int(np.count_nonzero(scores == boundary))


def recall_by_listener(
    y: IntArray, groups: StrArray, selected: npt.NDArray[np.bool_]
) -> dict[str, float]:
    """Misheard-word recall for each listener separately.

    Aggregate recall can be maximised by serving the highest-risk listeners well and the
    rest badly. Reporting the per-listener distribution makes that trade-off visible, so
    an equity claim can be checked instead of assumed.
    """
    out: dict[str, float] = {}
    for g in np.unique(groups):
        mask = groups == g
        pos = int(y[mask].sum())
        out[str(g)] = float((selected[mask] & (y[mask] == 1)).sum() / pos) if pos else float("nan")
    return out


def _point(
    strategy: str,
    budget: float,
    selected: npt.NDArray[np.bool_],
    y: IntArray,
    groups: StrArray,
    *,
    n_bootstrap: int,
    seed: int,
    per_listener: bool = True,
    n_ties: int = 0,
) -> BudgetPoint:
    n = int(y.size)
    n_misheard = int(y.sum())
    hits = int(((selected) & (y == 1)).sum())
    n_shown = int(selected.sum())

    def recall_stat(idx: IntArray) -> float:
        sub_y, sub_sel = y[idx], selected[idx]
        pos = int(sub_y.sum())
        return float(((sub_sel) & (sub_y == 1)).sum() / pos) if pos else float("nan")

    ci = (
        bootstrap_metric(groups, recall_stat, n_resamples=n_bootstrap, seed=seed)
        if n_bootstrap > 0
        else None
    )
    per = np.array(
        [v for v in recall_by_listener(y, groups, selected).values() if np.isfinite(v)],
        dtype=np.float64,
    )
    return BudgetPoint(
        achieved_count=n_shown,
        n_ties_at_boundary=n_ties,
        strategy=strategy,
        budget=budget,
        per_listener=per_listener,
        recall_min=float(per.min()) if per.size else float("nan"),
        recall_median=float(np.median(per)) if per.size else float("nan"),
        recall_max=float(per.max()) if per.size else float("nan"),
        achieved_ratio=n_shown / n if n else 0.0,
        caption_reduction_ratio=1.0 - (n_shown / n if n else 0.0),
        misheard_recall=hits / n_misheard if n_misheard else float("nan"),
        precision=hits / n_shown if n_shown else float("nan"),
        missed_rate=(n_misheard - hits) / n if n else float("nan"),
        n_words=n,
        n_misheard=n_misheard,
        recall_ci=ci,
    )


def budget_frontier(
    y: IntArray,
    groups: StrArray,
    scores: FloatArray,
    *,
    strategy: str,
    budgets: tuple[float, ...] = STANDARD_BUDGETS,
    n_bootstrap: int = 0,
    seed: int = 0,
    per_listener: bool = True,
) -> list[BudgetPoint]:
    """Evaluate one ranking across every budget."""
    return [
        _point(
            strategy,
            b,
            select_budget(scores, groups, b, per_listener=per_listener, tie_seed=seed),
            y,
            groups,
            n_bootstrap=n_bootstrap,
            seed=seed,
            per_listener=per_listener,
        )
        for b in budgets
    ]


def compare_strategies(
    arm_results: dict[str, ArmResult],
    words: list[str],
    *,
    budgets: tuple[float, ...] = STANDARD_BUDGETS,
    n_bootstrap: int = 0,
    seed: int = 0,
    per_listener: bool = True,
) -> list[BudgetPoint]:
    """Run the full RQ2 comparison across models and non-personalized baselines.

    ``arm_results`` maps a display name to an :class:`ArmResult`; all of them must share
    the same evaluation rows, which the ablation runner guarantees.
    """
    if not arm_results:
        raise ValueError("no model arms supplied")
    reference = next(iter(arm_results.values()))
    y, groups = reference.y_true, reference.groups
    if len(words) != y.size:
        raise ValueError(f"got {len(words)} words but {y.size} evaluation rows")
    for name, r in arm_results.items():
        if not np.array_equal(r.y_true, y) or not np.array_equal(r.groups, groups):
            raise ValueError(f"arm {name!r} was evaluated on different rows; cannot compare")

    points: list[BudgetPoint] = []
    points += budget_frontier(
        y,
        groups,
        _random_scores(y.size, seed),
        strategy="random",
        budgets=budgets,
        n_bootstrap=n_bootstrap,
        seed=seed,
        per_listener=per_listener,
    )
    points += budget_frontier(
        y,
        groups,
        _word_length_scores(words),
        strategy="word_length",
        budgets=budgets,
        n_bootstrap=n_bootstrap,
        seed=seed,
        per_listener=per_listener,
    )
    for name, r in arm_results.items():
        points += budget_frontier(
            y,
            groups,
            r.y_prob,
            strategy=f"model:{name}",
            budgets=budgets,
            n_bootstrap=n_bootstrap,
            seed=seed,
            per_listener=per_listener,
        )

    log.info(
        "caption.frontier_done",
        n_strategies=len(arm_results) + 2,
        n_budgets=len(budgets),
        n_words=int(y.size),
    )
    return points


# --------------------------------------------------------------------------- RQ3


@dataclass(frozen=True, slots=True)
class ThresholdComparison:
    """RQ3 — personalized vs global threshold at a matched overall caption ratio."""

    target_ratio: float
    global_tau: float
    personalized_taus: dict[str, float]
    global_point: BudgetPoint
    personalized_point: BudgetPoint
    #: Exact number of words the target ratio asks for over the pooled evaluation set.
    target_count: int = -1
    #: How the caption volume was matched. Recorded so a reader can tell which semantics
    #: produced a given number (see the P0.6 note on compare_thresholds).
    selection: str = "exact_count"
    tie_break: str = "index"

    @property
    def recall_gain(self) -> float:
        return self.personalized_point.misheard_recall - self.global_point.misheard_recall

    @property
    def ratio_gap(self) -> float:
        """How far the global threshold's achieved ratio drifts from the target."""
        return abs(self.global_point.achieved_ratio - self.target_ratio)

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_ratio": self.target_ratio,
            "target_count": self.target_count,
            "selection": self.selection,
            "tie_break": self.tie_break,
            "global_tau": self.global_tau,
            "equity": {
                "global_recall_min": self.global_point.recall_min,
                "global_recall_median": self.global_point.recall_median,
                "personalized_recall_min": self.personalized_point.recall_min,
                "personalized_recall_median": self.personalized_point.recall_median,
                "note": (
                    "A global threshold maximises aggregate recall by directing captions "
                    "toward high-risk listeners; per-listener thresholds spread the same "
                    "total volume evenly. Compare the minima, not only the aggregates."
                ),
            },
            "personalized_tau_summary": {
                "min": min(self.personalized_taus.values(), default=float("nan")),
                "median": float(np.median(list(self.personalized_taus.values())))
                if self.personalized_taus
                else float("nan"),
                "max": max(self.personalized_taus.values(), default=float("nan")),
                "n_listeners": len(self.personalized_taus),
            },
            "global": self.global_point.to_dict(),
            "personalized": self.personalized_point.to_dict(),
            "recall_gain": self.recall_gain,
            "global_ratio_gap": self.ratio_gap,
        }


def compare_thresholds(
    result: ArmResult,
    target_ratio: float,
    *,
    n_bootstrap: int = 0,
    seed: int = 0,
) -> ThresholdComparison:
    """Compare one global rule against per-listener rules **at the same caption volume**.

    Semantics change (2026-08-16, P0.6). Previously both arms selected with ``p > tau``.
    That delivers the requested volume only when scores are near-continuous; with heavy
    ties a threshold includes or excludes a whole tied block, so the "equal volume" claim
    RQ3 rests on was hoped for rather than enforced.

    Both arms now use exact-count selection with deterministic tie breaking:

    * global — the top ``round(target_ratio * N)`` items over the pooled scores;
    * personalized — the top ``round(target_ratio * n_u)`` items *within each listener*.

    The per-listener arm can still differ from the target by at most one word per
    listener, because each listener's count is rounded independently. That residual is
    reported rather than hidden.

    **Effect on previously recorded results:** the recorded five-seed ``rq1_main`` run
    diverged by at most 0.0011 in achieved ratio between the two arms (22 words out of
    20,000), against a reported recall gap of ~0.15. Those numbers are therefore not
    materially affected. The thresholds themselves are still reported for continuity.
    """
    y, groups, p = result.y_true, result.groups, result.y_prob
    n_total = int(y.size)
    target_count = round(target_ratio * n_total)

    by_listener = {str(g): p[groups == g] for g in np.unique(groups)}
    # Retained for continuity with the previously reported figures and because a
    # deployment that streams words one at a time needs a threshold, not a count.
    tau_global = global_threshold(by_listener, target_ratio)
    taus = {g: personalized_threshold(v, target_ratio) for g, v in by_listener.items()}

    sel_global = select_exact_count(p, target_count)
    global_ties = count_boundary_ties(p, target_count)

    sel_personal = np.zeros(p.shape, dtype=bool)
    personal_ties = 0
    for g in by_listener:
        idx = np.flatnonzero(groups == g)
        k = round(target_ratio * idx.size)
        local = select_exact_count(p[idx], k)
        sel_personal[idx[local]] = True
        personal_ties += count_boundary_ties(p[idx], k)

    return ThresholdComparison(
        target_ratio=target_ratio,
        target_count=target_count,
        global_tau=tau_global,
        personalized_taus=taus,
        global_point=_point(
            "threshold:global",
            target_ratio,
            sel_global,
            y,
            groups,
            n_bootstrap=n_bootstrap,
            seed=seed,
            n_ties=global_ties,
        ),
        personalized_point=_point(
            "threshold:personalized",
            target_ratio,
            sel_personal,
            y,
            groups,
            n_bootstrap=n_bootstrap,
            seed=seed,
            n_ties=personal_ties,
        ),
    )


def pareto_table(points: list[BudgetPoint]) -> list[dict[str, Any]]:
    """Flatten frontier points into rows for a results table."""
    return [
        {
            "strategy": pt.strategy,
            "budget": pt.budget,
            "achieved_ratio": round(pt.achieved_ratio, 4),
            "crr": round(pt.caption_reduction_ratio, 4),
            "misheard_recall": round(pt.misheard_recall, 4),
            "precision": round(pt.precision, 4),
            "missed_rate": round(pt.missed_rate, 4),
            "per_listener_budget": pt.per_listener,
            "recall_min": round(pt.recall_min, 4),
            "recall_median": round(pt.recall_median, 4),
            "recall_max": round(pt.recall_max, 4),
            "recall_ci_lo": round(pt.recall_ci.lo, 4) if pt.recall_ci else None,
            "recall_ci_hi": round(pt.recall_ci.hi, 4) if pt.recall_ci else None,
        }
        for pt in points
    ]


def recall_statistic(y: IntArray, selected: npt.NDArray[np.bool_]) -> Callable[[IntArray], float]:
    """Adapt misheard-word recall into an index-taking statistic for the bootstrap."""

    def statistic(idx: IntArray) -> float:
        sub_y, sub_sel = y[idx], selected[idx]
        pos = int(sub_y.sum())
        return float(((sub_sel) & (sub_y == 1)).sum() / pos) if pos else float("nan")

    return statistic
