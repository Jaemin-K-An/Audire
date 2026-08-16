"""Phase D19 / E — residual architecture, additive spline model, and listener ranking.

Three candidates, all evaluated by the same listener-level harness as the existing
baselines. None of them replaces the logistic reference model; each has to beat it on
held-out listeners to be preferred (ADR-0012 stands until the evidence says otherwise).

``ResidualRiskModel`` (D19)
    ``logit P = f_base(word, clinical, context) + delta(confusion, word)``.
    ``f_base`` captures general word/acoustic difficulty and the listener's global
    ability; ``delta`` is fitted **with the base prediction as a true offset**, so it can
    only earn weight by explaining what the base model got wrong. That is the direct
    expression of the project's central hypothesis: does this listener find this word
    unusually hard *relative to* what their global scores already predict?

``SplineAdditiveRiskModel`` (E20)
    A GAM-style model: per-feature B-spline bases plus a small, explicitly listed set of
    two-way interactions, fitted by penalised logistic regression. Chosen over an
    Explainable Boosting Machine to avoid adding a heavy dependency for a cohort of ~80
    independent listeners; it delivers the same properties the mission asks for —
    interpretable, nonlinear response curves, a *controlled* number of interactions, and
    no unrestricted high-capacity fitting. See ADR-0013.

``ListenerRankingModel`` (E21)
    LightGBM LambdaMART with ``listener_id`` as the query group. Selective captioning
    under a per-listener budget is a ranking problem, and this is the only model here
    that optimises within-listener ordering directly. Its score is **not** a probability;
    :class:`~audire.risk.calibration.CalibratedRiskModel` must be applied on
    listener-held-out data before any threshold policy consumes it.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from audire.risk.features import FeatureMatrix
from audire.risk.models import MODEL_VERSION, RiskModel

FloatArray = npt.NDArray[np.float64]

#: Columns the residual stage may use: the listener's confusion structure and the word,
#: i.e. exactly the "is this word unusually hard for this person" question.
_PERSONAL_PREFIXES: tuple[str, ...] = ("x_", "ix_", "x2_")


def _split_columns(names: tuple[str, ...]) -> tuple[list[int], list[int]]:
    """Partition column indices into (base, personal)."""
    personal = [i for i, n in enumerate(names) if n.startswith(_PERSONAL_PREFIXES)]
    base = [i for i in range(len(names)) if i not in set(personal)]
    return base, personal


def _logit(p: FloatArray, eps: float = 1e-6) -> FloatArray:
    q = np.clip(p, eps, 1 - eps)
    out: FloatArray = np.log(q / (1 - q))
    return out


def _fit_offset_logistic(
    x: FloatArray, y: npt.NDArray[np.int64], offset: FloatArray, l2: float
) -> FloatArray:
    """Penalised logistic regression with a fixed per-row offset.

    scikit-learn has no offset support, and approximating one by adding the base logit as
    an ordinary feature would let the second stage rescale the base prediction instead of
    merely correcting it. Solving the penalised objective directly keeps the architecture
    honest: the base prediction enters with coefficient exactly 1.
    """
    n, d = x.shape
    design = np.hstack([np.ones((n, 1)), x])

    def objective(w: FloatArray) -> tuple[float, FloatArray]:
        z = offset + design @ w
        # Stable log(1 + exp(z)).
        loss = float(np.sum(np.logaddexp(0.0, z) - y * z))
        penalty = 0.5 * l2 * float(w[1:] @ w[1:])  # intercept unpenalised
        grad = design.T @ (1.0 / (1.0 + np.exp(-z)) - y)
        grad[1:] += l2 * w[1:]
        return loss + penalty, grad

    result = minimize(
        objective, np.zeros(d + 1), jac=True, method="L-BFGS-B", options={"maxiter": 500}
    )
    coefficients: FloatArray = result.x
    return coefficients


@dataclass
class ResidualRiskModel(RiskModel):
    """Base word/clinical difficulty plus a personal residual fitted on an offset."""

    name: str = "residual"
    C: float = 1.0
    residual_l2: float = 1.0
    max_iter: int = 2000
    random_state: int = 0
    _base: Pipeline | None = None
    _residual: FloatArray | None = None
    _base_idx: list[int] = field(default_factory=list)
    _personal_idx: list[int] = field(default_factory=list)
    _imputer: SimpleImputer | None = None
    _scaler: StandardScaler | None = None
    feature_names: tuple[str, ...] = ()
    n_train: int = 0

    def fit(self, matrix: FeatureMatrix) -> ResidualRiskModel:
        if matrix.y is None:
            raise ValueError("cannot fit without labels")
        y = matrix.y
        if np.unique(y).size < 2:
            raise ValueError(
                "training data has a single class; a probability model cannot be fitted"
            )

        self.feature_names = matrix.feature_names
        self._base_idx, self._personal_idx = _split_columns(matrix.feature_names)
        if not self._personal_idx:
            raise ValueError(
                "residual 모델은 개인 혼동 특징(x_/ix_/x2_)이 있는 arm 에서만 의미가 있습니다"
            )

        self._base = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scale", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=self.C,
                        l1_ratio=0.0,
                        max_iter=self.max_iter,
                        random_state=self.random_state,
                        solver="lbfgs",
                    ),
                ),
            ]
        )
        self._base.fit(matrix.X[:, self._base_idx], y)
        offset = _logit(self._base.predict_proba(matrix.X[:, self._base_idx])[:, 1])

        personal = matrix.X[:, self._personal_idx]
        self._imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(personal)
        self._scaler = StandardScaler().fit(self._imputer.transform(personal))
        z = self._scaler.transform(self._imputer.transform(personal))
        self._residual = _fit_offset_logistic(z, y, offset, self.residual_l2)
        self.n_train = len(matrix)
        return self

    def predict_proba(self, matrix: FeatureMatrix) -> FloatArray:
        if (
            self._base is None
            or self._residual is None
            or self._imputer is None
            or self._scaler is None
        ):
            raise ValueError("residual model is not fitted")
        if matrix.feature_names != self.feature_names:
            raise ValueError("feature columns changed between fit and predict")
        offset = _logit(self._base.predict_proba(matrix.X[:, self._base_idx])[:, 1])
        z = self._scaler.transform(self._imputer.transform(matrix.X[:, self._personal_idx]))
        design = np.hstack([np.ones((z.shape[0], 1)), z])
        out: FloatArray = 1.0 / (1.0 + np.exp(-(offset + design @ self._residual)))
        return out

    @property
    def is_fitted(self) -> bool:
        return self._base is not None and self._residual is not None

    def residual_strength(self) -> float:
        """L2 norm of the residual coefficients, excluding the intercept.

        Near zero means the personal block explained nothing beyond the base model — a
        negative result worth reporting rather than hiding.
        """
        if self._residual is None:
            raise ValueError("model is not fitted")
        return float(np.linalg.norm(self._residual[1:]))

    def describe(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "name": self.name,
            "family": "base_plus_personal_residual",
            "model_version": MODEL_VERSION,
            "fitted": self.is_fitted,
            "n_train": self.n_train,
            "n_base_features": len(self._base_idx),
            "n_personal_features": len(self._personal_idx),
            "residual_l2": self.residual_l2,
        }
        if self.is_fitted:
            out["residual_strength"] = self.residual_strength()
        return out


@dataclass
class SplineAdditiveRiskModel(RiskModel):
    """GAM-style additive model: per-feature splines plus listed two-way interactions."""

    name: str = "spline_gam"
    n_knots: int = 5
    degree: int = 3
    C: float = 0.5
    max_iter: int = 3000
    random_state: int = 0
    #: Explicitly listed interactions. Keeping this a short, named list is what prevents
    #: the model from becoming an unrestricted high-capacity fit on ~80 listeners.
    interactions: tuple[tuple[str, str], ...] = (
        ("h_wrs", "x_r_phon"),
        ("c_snr_db", "x_r_phon"),
        ("h_audiogram_slope", "x2_min_p_onset"),
        ("h_high_freq_mean", "w2_onset_manner_fricative"),
        ("x_mean_evidence", "x_r_phon"),
        ("w_n_syllables", "x_min_p_correct"),
    )
    _pipeline: Pipeline | None = None
    _interaction_idx: list[tuple[int, int]] = field(default_factory=list)
    feature_names: tuple[str, ...] = ()
    n_train: int = 0

    def _augment(self, x: FloatArray) -> FloatArray:
        if not self._interaction_idx:
            return x
        extra = np.column_stack([x[:, a] * x[:, b] for a, b in self._interaction_idx])
        return np.hstack([x, extra])

    def fit(self, matrix: FeatureMatrix) -> SplineAdditiveRiskModel:
        if matrix.y is None:
            raise ValueError("cannot fit without labels")
        if np.unique(matrix.y).size < 2:
            raise ValueError(
                "training data has a single class; a probability model cannot be fitted"
            )

        names = list(matrix.feature_names)
        self._interaction_idx = [
            (names.index(a), names.index(b))
            for a, b in self.interactions
            if a in names and b in names
        ]
        self.feature_names = matrix.feature_names
        self._pipeline = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                (
                    "spline",
                    SplineTransformer(n_knots=self.n_knots, degree=self.degree, include_bias=False),
                ),
                ("scale", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=self.C,
                        l1_ratio=0.0,
                        max_iter=self.max_iter,
                        random_state=self.random_state,
                        solver="lbfgs",
                    ),
                ),
            ]
        )
        self._pipeline.fit(self._augment(matrix.X), matrix.y)
        self.n_train = len(matrix)
        return self

    def predict_proba(self, matrix: FeatureMatrix) -> FloatArray:
        if self._pipeline is None:
            raise ValueError("spline model is not fitted")
        if matrix.feature_names != self.feature_names:
            raise ValueError("feature columns changed between fit and predict")
        out: FloatArray = self._pipeline.predict_proba(self._augment(matrix.X))[:, 1]
        return out

    @property
    def is_fitted(self) -> bool:
        return self._pipeline is not None

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": "additive_spline_gam",
            "model_version": MODEL_VERSION,
            "fitted": self.is_fitted,
            "n_train": self.n_train,
            "n_knots": self.n_knots,
            "degree": self.degree,
            "C": self.C,
            "n_interactions": len(self._interaction_idx),
            "interactions": [list(pair) for pair in self.interactions],
        }


def _canonical_order(
    x: FloatArray, y: npt.NDArray[np.int64], groups: npt.NDArray[np.str_]
) -> npt.NDArray[np.int64]:
    """Row order for the ranker: by listener, then by row content.

    LightGBM's ranking API requires rows to be contiguous by query, so sorting by listener
    is what makes "the query group is the listener" literally true rather than merely
    intended. Sorting *within* a listener is the less obvious half. LambdaMART truncates
    the pairs it considers (``lambdarank_truncation_level``), and which pairs survive
    depends on the incoming row order, so a plain stable sort leaves the fitted model a
    function of the order the caller happened to assemble the matrix in. Measured on a
    16-listener cohort, permuting the training rows moved the within-listener Spearman
    correlation of the scores to ~0.86 — far too large to dismiss as numerical noise, and
    enough to change which words a fixed budget selects.

    Ordering by a content hash of ``(row, label)`` makes the fit a deterministic function
    of the *set* of training examples. Rows that collide are identical in both features and
    label, so their relative order genuinely does not matter. The digest is
    :func:`hashlib.blake2b` rather than :func:`hash` because the latter is salted per
    process and would silently break reproducibility across runs.
    """
    digests = [
        hashlib.blake2b(x[i].tobytes() + bytes([int(y[i])]), digest_size=8).digest()
        for i in range(x.shape[0])
    ]
    keys = np.array(digests, dtype="S8")
    # lexsort applies the last key first, so this is "by listener, then by content".
    return np.lexsort((keys, groups)).astype(np.int64)


@dataclass
class ListenerRankingModel(RiskModel):
    """LightGBM LambdaMART with the listener as the query group.

    Optimises *within-listener* ordering, which is what a per-listener caption budget
    consumes. The output is a ranking score, not a probability; ``predict_proba`` returns
    a monotone squash of it purely so the model satisfies the common interface, and the
    docstring of that method says so. Any threshold policy must calibrate first.
    """

    name: str = "lambdamart"
    n_estimators: int = 300
    learning_rate: float = 0.05
    num_leaves: int = 15
    min_child_samples: int = 40
    reg_lambda: float = 1.0
    random_state: int = 0
    _model: Any = None
    _imputer: SimpleImputer | None = None
    feature_names: tuple[str, ...] = ()
    n_train: int = 0
    n_train_groups: int = 0

    def fit(self, matrix: FeatureMatrix) -> ListenerRankingModel:
        import lightgbm as lgb

        if matrix.y is None:
            raise ValueError("cannot fit without labels")
        if np.unique(matrix.y).size < 2:
            raise ValueError("training data has a single class; a ranker cannot be fitted")

        order = _canonical_order(matrix.X, matrix.y, matrix.groups)
        x = matrix.X[order]
        y = matrix.y[order]
        groups = matrix.groups[order]
        _, counts = np.unique(groups, return_counts=True)

        self._imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(x)
        self._model = lgb.LGBMRanker(
            objective="lambdarank",
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            min_child_samples=self.min_child_samples,
            reg_lambda=self.reg_lambda,
            random_state=self.random_state,
            verbose=-1,
        )
        self._model.fit(self._imputer.transform(x), y, group=counts.tolist())
        self.feature_names = matrix.feature_names
        self.n_train = len(matrix)
        self.n_train_groups = int(counts.size)
        return self

    def predict_score(self, matrix: FeatureMatrix) -> FloatArray:
        """Raw ranking score. Comparable **within** a listener, not across listeners."""
        if self._model is None or self._imputer is None:
            raise ValueError("ranking model is not fitted")
        if matrix.feature_names != self.feature_names:
            raise ValueError("feature columns changed between fit and predict")
        out: FloatArray = np.asarray(
            self._model.predict(self._imputer.transform(matrix.X)), dtype=np.float64
        )
        return out

    def predict_proba(self, matrix: FeatureMatrix) -> FloatArray:
        """A monotone squash of the ranking score — **not** a calibrated probability.

        Preserves the ordering the ranker learned so that budget policies and ranking
        metrics work unchanged, but the values carry no probabilistic meaning. Calibrate
        on listener-held-out data before any threshold policy consumes them; the
        calibration comparison reports Brier and ECE for both the raw and calibrated
        versions so the difference is visible rather than assumed.
        """
        out: FloatArray = 1.0 / (1.0 + np.exp(-self.predict_score(matrix)))
        return out

    @property
    def is_fitted(self) -> bool:
        return self._model is not None

    def describe(self) -> dict[str, Any]:
        import importlib.metadata as md

        try:
            version = md.version("lightgbm")
        except md.PackageNotFoundError:  # pragma: no cover - not installed
            version = "not-installed"
        return {
            "name": self.name,
            "family": "lambdamart_listener_grouped",
            "model_version": MODEL_VERSION,
            "lightgbm_version": version,
            "fitted": self.is_fitted,
            "n_train": self.n_train,
            "n_train_groups": self.n_train_groups,
            "params": {
                "n_estimators": self.n_estimators,
                "learning_rate": self.learning_rate,
                "num_leaves": self.num_leaves,
                "min_child_samples": self.min_child_samples,
                "reg_lambda": self.reg_lambda,
            },
            "output_is_probability": False,
            "note": (
                "ranking score; requires listener-held-out calibration before a threshold "
                "policy uses it"
            ),
        }
