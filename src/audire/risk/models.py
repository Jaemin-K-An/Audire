"""Word-mishearing risk models.

Model families, in the order the research plan requires them to be compared:

``PhonemeIndependenceRisk``
    The deterministic ``R_phon = 1 - prod_k C_u(phi_k, phi_k)``. Not fitted. Included as
    an interpretable comparator; phoneme independence is not assumed to be correct.

``LogisticRiskModel``
    Regularised logistic regression on a named feature set. The primary learned family,
    chosen for interpretability and for well-behaved probabilities.

``GradientBoostedRiskModel``
    Histogram gradient boosting, the single justified nonlinear comparator. It is
    reported alongside the logistic model, never instead of it, and it is only meaningful
    when the cohort is large enough — the runner records the training size with every
    result so that an over-fitted nonlinear win is visible.

Every model records its version, arm, feature names and fitted hyper-parameters so that a
prediction can be traced back to the exact estimator that produced it.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from audire.confusion.profile import ConfusionProfile
from audire.risk.features import (
    FeatureMatrix,
    FeatureSpec,
    WordContext,
    build_matrix,
    phoneme_independence_risk,
)

FloatArray = npt.NDArray[np.float64]

#: Bumped whenever prediction semantics change, so cached results become invalid rather
#: than silently shifting.
MODEL_VERSION = "1.0.0"


class RiskModel(ABC):
    """Common interface: fit on a :class:`FeatureMatrix`, predict a probability per row."""

    name: str = "abstract"

    @abstractmethod
    def fit(self, matrix: FeatureMatrix) -> RiskModel: ...

    @abstractmethod
    def predict_proba(self, matrix: FeatureMatrix) -> FloatArray:
        """Return ``P(misheard)`` for every row, in ``[0, 1]``."""

    @property
    @abstractmethod
    def is_fitted(self) -> bool: ...

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "model_version": MODEL_VERSION, "fitted": self.is_fitted}


# --------------------------------------------------------------------------- deterministic


@dataclass
class PhonemeIndependenceRisk(RiskModel):
    """``R_phon`` read straight out of the feature matrix.

    Requires the ``confusion`` block, which supplies ``x_r_phon``. It is deterministic:
    :meth:`fit` records the training size for provenance and changes nothing else.
    """

    name: str = "phoneme_independence"
    n_train: int = 0
    _fitted: bool = False

    def fit(self, matrix: FeatureMatrix) -> PhonemeIndependenceRisk:
        if "x_r_phon" not in matrix.feature_names:
            raise ValueError(
                "PhonemeIndependenceRisk needs the 'confusion' feature block "
                "(column 'x_r_phon'); the given arm does not include it"
            )
        self.n_train = len(matrix)
        self._fitted = True
        return self

    def predict_proba(self, matrix: FeatureMatrix) -> FloatArray:
        try:
            col = matrix.feature_names.index("x_r_phon")
        except ValueError as exc:
            raise ValueError("feature matrix has no 'x_r_phon' column") from exc
        out: FloatArray = np.clip(matrix.X[:, col].astype(np.float64), 0.0, 1.0)
        return out

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def describe(self) -> dict[str, Any]:
        return {
            **super().describe(),
            "family": "deterministic",
            "formula": "1 - prod_k C_u(phi_k, phi_k)",
            "n_train": self.n_train,
            "note": "phoneme independence is a comparator, not an assumed truth",
        }


# --------------------------------------------------------------------------- sklearn-backed


@dataclass
class _SklearnRiskModel(RiskModel):
    """Shared fitting/prediction machinery for the sklearn-backed families."""

    name: str = "sklearn"
    pipeline: Pipeline | None = None
    feature_names: tuple[str, ...] = ()
    n_train: int = 0
    train_positive_rate: float = float("nan")
    params: dict[str, Any] = field(default_factory=dict)

    def _build(self) -> Pipeline:  # pragma: no cover - overridden by subclasses
        raise NotImplementedError

    def fit(self, matrix: FeatureMatrix) -> _SklearnRiskModel:
        if matrix.y is None:
            raise ValueError("cannot fit without labels")
        y = matrix.y
        classes = np.unique(y)
        if classes.size < 2:
            raise ValueError(
                f"training data has a single class ({classes.tolist()}); a probability "
                f"model cannot be fitted. Widen the training split."
            )
        self.pipeline = self._build()
        self.pipeline.fit(matrix.X, y)
        self.feature_names = matrix.feature_names
        self.n_train = len(matrix)
        self.train_positive_rate = float(np.mean(y))
        return self

    def predict_proba(self, matrix: FeatureMatrix) -> FloatArray:
        if self.pipeline is None:
            raise ValueError(f"{self.name} is not fitted")
        if matrix.feature_names != self.feature_names:
            raise ValueError(
                "feature columns changed between fit and predict; "
                f"fitted on {len(self.feature_names)} columns, got {len(matrix.feature_names)}"
            )
        proba: FloatArray = self.pipeline.predict_proba(matrix.X)[:, 1].astype(np.float64)
        return proba

    @property
    def is_fitted(self) -> bool:
        return self.pipeline is not None

    def describe(self) -> dict[str, Any]:
        return {
            **super().describe(),
            "n_train": self.n_train,
            "n_features": len(self.feature_names),
            "train_positive_rate": self.train_positive_rate,
            "params": self.params,
        }


@dataclass
class LogisticRiskModel(_SklearnRiskModel):
    """L2-regularised logistic regression with fold-safe imputation and scaling."""

    name: str = "logistic"
    C: float = 1.0
    max_iter: int = 2000
    random_state: int = 0

    def _build(self) -> Pipeline:
        # scikit-learn >= 1.8 expresses the penalty through `l1_ratio`; 0.0 is pure L2.
        self.params = {"C": self.C, "l1_ratio": 0.0, "max_iter": self.max_iter}
        return Pipeline(
            [
                # Imputation is inside the pipeline so its statistics are fitted on the
                # training fold only and can never leak from the evaluation fold.
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

    def coefficients(self) -> dict[str, float]:
        """Standardised coefficients, largest absolute value first.

        Interpretable because the scaler puts every feature on a comparable scale.
        """
        if self.pipeline is None:
            raise ValueError("model is not fitted")
        clf = self.pipeline.named_steps["clf"]
        coefs = dict(zip(self.feature_names, clf.coef_[0].tolist(), strict=True))
        return dict(sorted(coefs.items(), key=lambda kv: -abs(kv[1])))

    def describe(self) -> dict[str, Any]:
        out = super().describe()
        out["family"] = "logistic_regression"
        if self.pipeline is not None:
            out["top_coefficients"] = dict(list(self.coefficients().items())[:12])
        return out


@dataclass
class GradientBoostedRiskModel(_SklearnRiskModel):
    """Histogram gradient boosting: the single justified nonlinear comparator.

    Handles ``NaN`` natively, so no imputation step is inserted; the missing-indicator
    columns from the feature builder still tell it what was absent.
    """

    name: str = "gradient_boosting"
    max_iter: int = 200
    learning_rate: float = 0.06
    max_leaf_nodes: int = 15
    min_samples_leaf: int = 40
    l2_regularization: float = 1.0
    random_state: int = 0

    def _build(self) -> Pipeline:
        self.params = {
            "max_iter": self.max_iter,
            "learning_rate": self.learning_rate,
            "max_leaf_nodes": self.max_leaf_nodes,
            "min_samples_leaf": self.min_samples_leaf,
            "l2_regularization": self.l2_regularization,
        }
        return Pipeline(
            [
                (
                    "clf",
                    HistGradientBoostingClassifier(
                        max_iter=self.max_iter,
                        learning_rate=self.learning_rate,
                        max_leaf_nodes=self.max_leaf_nodes,
                        min_samples_leaf=self.min_samples_leaf,
                        l2_regularization=self.l2_regularization,
                        random_state=self.random_state,
                        early_stopping=False,
                    ),
                )
            ]
        )

    def describe(self) -> dict[str, Any]:
        out = super().describe()
        out["family"] = "hist_gradient_boosting"
        out["caution"] = (
            "nonlinear comparator; interpret only alongside the logistic model and the "
            "recorded training size"
        )
        return out


#: Registry of model families available to experiment configs.
MODEL_REGISTRY: dict[str, type[RiskModel]] = {
    "phoneme_independence": PhonemeIndependenceRisk,
    "logistic": LogisticRiskModel,
    "gradient_boosting": GradientBoostedRiskModel,
}

#: Phase D/E families that live in :mod:`audire.risk.advanced`. They are resolved on first
#: use rather than imported here, because that module imports :class:`RiskModel` from this
#: one. Registering them lazily keeps a single registry without a circular import, and
#: :func:`known_models` still reports them so a config typo gets the full candidate list.
_LAZY_MODELS: dict[str, str] = {
    "residual": "ResidualRiskModel",
    "cross_fitted_residual": "CrossFittedResidualRiskModel",
    "spline_gam": "SplineAdditiveRiskModel",
    "lambdamart": "ListenerRankingModel",
}


def known_models() -> list[str]:
    """Every model name an experiment config may name, eager and lazy alike."""
    return sorted(set(MODEL_REGISTRY) | set(_LAZY_MODELS))


def resolve_model(name: str) -> type[RiskModel]:
    """Look up a model family, importing the advanced module only if it is needed."""
    if name in MODEL_REGISTRY:
        return MODEL_REGISTRY[name]
    if name in _LAZY_MODELS:
        from audire.risk import advanced

        cls: type[RiskModel] = getattr(advanced, _LAZY_MODELS[name])
        MODEL_REGISTRY[name] = cls
        return cls
    raise KeyError(f"unknown model {name!r}; known: {known_models()}")


def make_model(name: str, **kwargs: Any) -> RiskModel:
    """Instantiate a registered model family."""
    return resolve_model(name)(**kwargs)


# --------------------------------------------------------------------------- scoring helper


@dataclass(frozen=True, slots=True)
class WordScorer:
    """Bundles a fitted model with its feature spec so that a word can be scored directly.

    This is what the API and the caption engine use; it guarantees the same feature code
    path in research and in production.
    """

    model: RiskModel
    spec: FeatureSpec
    provenance: dict[str, Any] = field(default_factory=dict)

    def score(
        self,
        listener_id: str,
        words: list[str],
        contexts: list[WordContext],
        hearing: Any,
        confusion: ConfusionProfile | None,
    ) -> FloatArray:
        if len(words) != len(contexts):
            raise ValueError("words and contexts must have the same length")
        # Defended here as well as in the pipeline: the scorer is a public entry point,
        # so checking only the pipeline would leave a way to score one listener with
        # another listener's profile.
        if confusion is not None and confusion.listener_id != listener_id:
            raise ValueError(
                f"confusion profile belongs to listener {confusion.listener_id!r} but "
                f"{listener_id!r} was requested"
            )
        hearing_id = getattr(hearing, "listener_id", None)
        if hearing_id is not None and hearing_id != listener_id:
            raise ValueError(
                f"hearing profile belongs to listener {hearing_id!r} but "
                f"{listener_id!r} was requested"
            )
        if not words:
            return np.zeros(0, dtype=np.float64)
        matrix = build_matrix(
            self.spec,
            [(listener_id, w, c, hearing, confusion) for w, c in zip(words, contexts, strict=True)],
        )
        return self.model.predict_proba(matrix)

    def describe(self) -> dict[str, Any]:
        return {
            "model": self.model.describe(),
            "arm": self.spec.name,
            "blocks": list(self.spec.blocks),
            "provenance": self.provenance,
        }

    def save_description(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.describe(), ensure_ascii=False, indent=2), encoding="utf-8")


def independence_scores(words: list[str], confusion: ConfusionProfile) -> FloatArray:
    """``R_phon`` for a list of words, without building a feature matrix."""
    return np.array([phoneme_independence_risk(w, confusion) for w in words], dtype=np.float64)


__all__ = [
    "MODEL_REGISTRY",
    "MODEL_VERSION",
    "GradientBoostedRiskModel",
    "LogisticRiskModel",
    "PhonemeIndependenceRisk",
    "RiskModel",
    "WordScorer",
    "independence_scores",
    "known_models",
    "make_model",
    "resolve_model",
]
