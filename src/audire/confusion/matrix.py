"""Per-position phoneme confusion matrices with explicit counts, priors and smoothing.

Scientific contract
-------------------
1. **Counts and probabilities are stored separately and are both always available.**
   A probability of 1.0 estimated from one trial and a probability of 1.0 estimated
   from forty trials are different pieces of evidence, so the estimator never returns a
   probability without also being able to return the sample size behind it.
2. **Rows sum to exactly 1 after smoothing**, including rows with zero observations
   (which fall back to the prior).
3. **No observation is ever dropped.** Omissions and additions are ordinary cells
   because ``NO_CODA`` is a real category; unusable responses land in ``NO_RESPONSE``.

Smoothing
---------
Row ``i`` is estimated as a Dirichlet posterior mean:

.. math::  \\hat p_{ij} = \\frac{n_{ij} + \\alpha\\,\\pi_{ij}}{n_i + \\alpha}

where :math:`\\pi_i` is a prior row summing to 1 and :math:`\\alpha` is the total prior
pseudo-count. With a uniform prior this is additive (Perks) smoothing; with a pooled
group matrix as prior it is hierarchical shrinkage toward the group. The prior is an
explicit, serialisable object -- never an implicit constant. See ADR-0006.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Self

import numpy as np
import numpy.typing as npt

from audire.hangul.inventory import Position, categories_for

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

#: Default total prior pseudo-count. Deliberately small: it regularises unseen cells
#: without overwhelming a realistic calibration of 25-100 trials. See ADR-0006.
DEFAULT_ALPHA: float = 1.0

PriorKind = Literal["uniform", "explicit"]


@dataclass(frozen=True, slots=True)
class SmoothingSpec:
    """A fully specified, serialisable smoothing configuration."""

    alpha: float = DEFAULT_ALPHA
    kind: PriorKind = "uniform"
    #: Row-stochastic prior of shape (n_target, n_perceived). Required when ``kind`` is
    #: ``"explicit"``, ignored otherwise.
    prior: FloatArray | None = None

    def __post_init__(self) -> None:
        if self.alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {self.alpha}")
        if self.kind == "explicit":
            if self.prior is None:
                raise ValueError("kind='explicit' requires a prior matrix")
            if not np.allclose(self.prior.sum(axis=1), 1.0, atol=1e-9):
                raise ValueError("explicit prior rows must each sum to 1")
            if np.any(self.prior < 0):
                raise ValueError("explicit prior must be non-negative")
        elif self.prior is not None:
            raise ValueError("prior matrix supplied but kind is not 'explicit'")

    def prior_matrix(self, shape: tuple[int, int]) -> FloatArray:
        """Return the prior for a matrix of ``shape``."""
        if self.kind == "explicit":
            assert self.prior is not None
            if self.prior.shape != shape:
                raise ValueError(f"prior shape {self.prior.shape} != matrix shape {shape}")
            return self.prior
        return np.full(shape, 1.0 / shape[1], dtype=np.float64)

    def describe(self) -> dict[str, Any]:
        """JSON-safe description for provenance records."""
        return {
            "alpha": self.alpha,
            "kind": self.kind,
            "has_explicit_prior": self.prior is not None,
        }


@dataclass(slots=True)
class ConfusionMatrix:
    """Confusion counts for one syllable position, plus a smoothing specification.

    The matrix is rectangular: rows are target categories, columns are perceived
    categories (target categories plus ``NO_RESPONSE``).
    """

    position: Position
    counts: IntArray
    smoothing: SmoothingSpec = field(default_factory=SmoothingSpec)

    def __post_init__(self) -> None:
        expected = (len(self.target_labels), len(self.perceived_labels))
        if self.counts.shape != expected:
            raise ValueError(
                f"{self.position} counts must have shape {expected}, got {self.counts.shape}"
            )
        if np.any(self.counts < 0):
            raise ValueError("confusion counts must be non-negative")
        if not np.issubdtype(self.counts.dtype, np.integer):
            raise TypeError(f"counts must be an integer array, got dtype {self.counts.dtype}")

    # ---------------------------------------------------------------- construction

    @classmethod
    def empty(cls, position: Position, smoothing: SmoothingSpec | None = None) -> Self:
        """Return an all-zero matrix for ``position``."""
        n_t = len(categories_for(position, axis="target"))
        n_p = len(categories_for(position, axis="perceived"))
        return cls(
            position=position,
            counts=np.zeros((n_t, n_p), dtype=np.int64),
            smoothing=smoothing or SmoothingSpec(),
        )

    # ---------------------------------------------------------------- labels/indexing

    @property
    def target_labels(self) -> tuple[str, ...]:
        return categories_for(self.position, axis="target")

    @property
    def perceived_labels(self) -> tuple[str, ...]:
        return categories_for(self.position, axis="perceived")

    def _row(self, target: str) -> int:
        try:
            return self.target_labels.index(target)
        except ValueError as exc:
            raise KeyError(f"{target!r} is not a valid {self.position} target category") from exc

    def _col(self, perceived: str) -> int:
        try:
            return self.perceived_labels.index(perceived)
        except ValueError as exc:
            raise KeyError(
                f"{perceived!r} is not a valid {self.position} perceived category"
            ) from exc

    # ---------------------------------------------------------------- accumulation

    def observe(self, target: str, perceived: str, weight: int = 1) -> None:
        """Record ``weight`` observations of ``target`` being reported as ``perceived``."""
        if weight < 0:
            raise ValueError("observation weight must be non-negative")
        self.counts[self._row(target), self._col(perceived)] += weight

    # ---------------------------------------------------------------- evidence

    @property
    def total_observations(self) -> int:
        """Total number of trials contributing to this matrix."""
        return int(self.counts.sum())

    def row_counts(self) -> IntArray:
        """Number of observations per target category. Never discarded."""
        return self.counts.sum(axis=1).astype(np.int64)

    def n_observations(self, target: str) -> int:
        """Number of trials in which ``target`` was presented."""
        return int(self.counts[self._row(target)].sum())

    @property
    def observed_targets(self) -> tuple[str, ...]:
        """Target categories with at least one observation."""
        rows = self.row_counts()
        return tuple(lbl for i, lbl in enumerate(self.target_labels) if rows[i] > 0)

    @property
    def unobserved_targets(self) -> tuple[str, ...]:
        """Target categories with zero observations; their rows equal the prior exactly."""
        rows = self.row_counts()
        return tuple(lbl for i, lbl in enumerate(self.target_labels) if rows[i] == 0)

    # ---------------------------------------------------------------- estimation

    def probabilities(self) -> FloatArray:
        """Return the smoothed row-stochastic probability matrix.

        Every row sums to 1, including rows with no observations.
        """
        shape = (len(self.target_labels), len(self.perceived_labels))
        prior = self.smoothing.prior_matrix(shape)
        alpha = self.smoothing.alpha
        numer = self.counts.astype(np.float64) + alpha * prior
        denom = self.counts.sum(axis=1, keepdims=True).astype(np.float64) + alpha
        if np.any(denom == 0.0):
            # alpha == 0 and an empty row: undefined. Fall back to the prior and say so.
            raise ValueError(
                "cannot estimate probabilities: alpha=0 leaves rows with no observations "
                f"undefined ({', '.join(self.unobserved_targets)}). Use alpha > 0."
            )
        return numer / denom

    def empirical_probabilities(self) -> FloatArray:
        """Unsmoothed row-normalised counts. Rows with no observations are ``NaN``.

        Provided so that raw evidence remains inspectable next to the smoothed estimate.
        """
        denom = self.counts.sum(axis=1, keepdims=True).astype(np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            out = self.counts.astype(np.float64) / denom
        out[denom[:, 0] == 0.0, :] = np.nan
        return out

    def p(self, target: str, perceived: str) -> float:
        """Smoothed ``P(perceived | target)``."""
        return float(self.probabilities()[self._row(target), self._col(perceived)])

    def p_correct(self, target: str) -> float:
        """Smoothed ``P(perceived == target | target)`` -- the diagonal element."""
        return self.p(target, target)

    def confusability(self, target: str) -> float:
        """``1 - p_correct(target)``: the probability this phoneme is misperceived."""
        return 1.0 - self.p_correct(target)

    def diagonal(self) -> FloatArray:
        """Smoothed correct-recognition probability for every target category."""
        probs = self.probabilities()
        n_t = len(self.target_labels)
        return probs[np.arange(n_t), np.arange(n_t)]

    def row_entropy(self) -> FloatArray:
        """Shannon entropy (nats) of each smoothed row: how diffuse the confusions are."""
        p = self.probabilities()
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = np.where(p > 0, p * np.log(p), 0.0)
        return -terms.sum(axis=1)

    def top_confusions(self, target: str, k: int = 3) -> list[tuple[str, float, int]]:
        """Return the ``k`` most likely *incorrect* responses to ``target``.

        Each entry is ``(perceived_label, smoothed_probability, raw_count)``. Used to
        explain why a word received its risk score.
        """
        r = self._row(target)
        probs = self.probabilities()[r]
        order = np.argsort(-probs)
        out: list[tuple[str, float, int]] = []
        for c in order:
            label = self.perceived_labels[c]
            if label == target:
                continue
            out.append((label, float(probs[c]), int(self.counts[r, c])))
            if len(out) >= k:
                break
        return out

    def with_smoothing(self, smoothing: SmoothingSpec) -> ConfusionMatrix:
        """Return a copy that shares the counts but uses a different smoothing spec."""
        return ConfusionMatrix(
            position=self.position, counts=self.counts.copy(), smoothing=smoothing
        )

    # ---------------------------------------------------------------- combination

    def __add__(self, other: ConfusionMatrix) -> ConfusionMatrix:
        """Pool counts from two matrices for the same position."""
        if other.position is not self.position:
            raise ValueError(f"cannot add {self.position} and {other.position} matrices")
        return ConfusionMatrix(
            position=self.position,
            counts=self.counts + other.counts,
            smoothing=self.smoothing,
        )

    # ---------------------------------------------------------------- serialisation

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe representation preserving raw counts."""
        return {
            "position": self.position.value,
            "target_labels": list(self.target_labels),
            "perceived_labels": list(self.perceived_labels),
            "counts": self.counts.tolist(),
            "row_counts": self.row_counts().tolist(),
            "total_observations": self.total_observations,
            "smoothing": self.smoothing.describe(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any], smoothing: SmoothingSpec | None = None) -> Self:
        position = Position(payload["position"])
        counts = np.asarray(payload["counts"], dtype=np.int64)
        spec = smoothing
        if spec is None:
            raw = payload.get("smoothing") or {}
            spec = SmoothingSpec(alpha=float(raw.get("alpha", DEFAULT_ALPHA)), kind="uniform")
        return cls(position=position, counts=counts, smoothing=spec)

    def save_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
