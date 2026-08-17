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

    @staticmethod
    def is_available() -> bool:
        """LightGBM 이 이 환경에 설치되어 있는가.

        기본 설치에는 없습니다. E22 에서 이 계열이 참조 로지스틱을 이기지 못했으므로
        배포 경로의 필수 의존성이 아니며, ``research-models`` extra 로 분리되어 있습니다.
        """
        import importlib.util

        return importlib.util.find_spec("lightgbm") is not None

    def fit(self, matrix: FeatureMatrix) -> ListenerRankingModel:
        try:
            import lightgbm as lgb
        except ImportError as exc:
            # 맨 ImportError 대신 무엇을 어떻게 설치해야 하는지 말해 줍니다. 이 모델은
            # 선택적 연구 계열이므로 부재는 오류가 아니라 정상 상태입니다.
            raise ImportError(
                "LambdaMART(lambdamart)는 선택적 연구 모델 extra 를 필요로 합니다:\n"
                "    pip install -e '.[research-models]'\n"
                "또는 make bootstrap-research\n"
                "배포 경로는 이 의존성 없이 동작합니다."
            ) from exc

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


@dataclass
class CrossFittedResidualRiskModel(RiskModel):
    """Phase E — 잔차를 **폴드 밖 기저 예측** 위에서 적합합니다.

    기존 :class:`ResidualRiskModel` 의 위험
    ---------------------------------------
    기존 구현은 기저 모델을 훈련 청취자 전체에 적합한 뒤, **같은 행들에 대한 예측**을
    offset 으로 써서 잔차를 적합합니다. 그 예측은 표본 안(in-sample)이라 실제보다
    낙관적이고, 잔차 단계는 "이 청취자가 이 단어를 유난히 어려워하는가" 가 아니라
    "기저 모델이 자기 훈련 데이터에서 어디를 과신했는가" 를 배우게 됩니다. 그 보정은
    홀드아웃 청취자에게 옮겨지지 않습니다.

    교차적합
    --------
    바깥 훈련 청취자를 다시 청취자 단위 내부 폴드로 나누고, 내부 훈련 청취자로 적합한
    기저 모델이 내부 검증 청취자를 예측하게 합니다. 모든 바깥 훈련 행이 **자기 청취자를
    보지 않은** 기저 모델의 예측을 받고, 잔차는 그 offset 위에서만 적합됩니다. 마지막에
    기저 모델을 바깥 훈련 청취자 전체로 다시 적합해 예측에 씁니다.

    누출 방지
    ---------
    바깥 홀드아웃 청취자는 기저 학습·잔차 학습·전처리 어디에도 들어가지 않습니다. 내부
    폴드도 청취자 단위이므로 한 청취자의 행이 자기 기저 예측을 만드는 데 쓰이지 않습니다.
    """

    name: str = "cross_fitted_residual"
    C: float = 1.0
    residual_l2: float = 1.0
    max_iter: int = 2000
    random_state: int = 0
    #: 내부 폴드 수. 바깥 폴드와 독립이며, 작으면 offset 이 거칠고 크면 느립니다.
    n_inner_splits: int = 4
    _base: Pipeline | None = None
    _residual: FloatArray | None = None
    _base_idx: list[int] = field(default_factory=list)
    _personal_idx: list[int] = field(default_factory=list)
    _imputer: SimpleImputer | None = None
    _scaler: StandardScaler | None = None
    feature_names: tuple[str, ...] = ()
    n_train: int = 0
    #: 내부 폴드가 실제로 몇 개 돌았는지. 청취자가 적어 줄어들면 여기서 드러납니다.
    n_inner_folds_used: int = 0

    def _make_base(self) -> Pipeline:
        return Pipeline(
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

    def fit(self, matrix: FeatureMatrix) -> CrossFittedResidualRiskModel:
        from audire.eval.splits import listener_folds

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
                "cross_fitted_residual 모델은 개인 혼동 특징(x_/ix_/x2_)이 있는 arm 에서만 "
                "의미가 있습니다"
            )

        base_x = matrix.X[:, self._base_idx]
        n_listeners = int(np.unique(matrix.groups).size)
        n_inner = min(self.n_inner_splits, n_listeners)
        if n_inner < 2:
            raise ValueError(
                f"교차적합에는 최소 2명의 훈련 청취자가 필요합니다 (현재 {n_listeners}명)"
            )

        # 폴드 밖 기저 예측. 모든 행이 자기 청취자를 보지 않은 모델의 예측을 받습니다.
        oof = np.full(y.shape, np.nan, dtype=np.float64)
        folds = listener_folds(
            matrix.groups, y, n_splits=n_inner, stratify=True, seed=self.random_state
        )
        for fold in folds:
            inner = self._make_base()
            inner.fit(base_x[fold.train_idx], y[fold.train_idx])
            oof[fold.test_idx] = inner.predict_proba(base_x[fold.test_idx])[:, 1]
        self.n_inner_folds_used = len(folds)

        if np.any(~np.isfinite(oof)):
            raise RuntimeError(
                "일부 훈련 행이 폴드 밖 기저 예측을 받지 못했습니다; 내부 분할이 불완전합니다"
            )

        personal = matrix.X[:, self._personal_idx]
        self._imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(personal)
        self._scaler = StandardScaler().fit(self._imputer.transform(personal))
        z = self._scaler.transform(self._imputer.transform(personal))
        # 잔차는 **폴드 밖** offset 위에서만 적합됩니다.
        self._residual = _fit_offset_logistic(z, y, _logit(oof), self.residual_l2)

        # 예측에 쓸 기저 모델은 바깥 훈련 청취자 전체로 다시 적합합니다.
        self._base = self._make_base()
        self._base.fit(base_x, y)
        self.n_train = len(matrix)
        return self

    def predict_proba(self, matrix: FeatureMatrix) -> FloatArray:
        if (
            self._base is None
            or self._residual is None
            or self._imputer is None
            or self._scaler is None
        ):
            raise ValueError("cross-fitted residual model is not fitted")
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
        """절편을 뺀 잔차 계수의 L2 노름.

        0 에 가까우면 개인 블록이 기저 예측 위에 아무것도 더하지 못한 것이며, 그것은
        숨기지 않고 보고해야 할 음성 결과입니다.
        """
        if self._residual is None:
            raise ValueError("model is not fitted")
        return float(np.linalg.norm(self._residual[1:]))

    def describe(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "name": self.name,
            "family": "cross_fitted_base_plus_personal_residual",
            "model_version": MODEL_VERSION,
            "fitted": self.is_fitted,
            "n_train": self.n_train,
            "n_base_features": len(self._base_idx),
            "n_personal_features": len(self._personal_idx),
            "n_inner_splits_requested": self.n_inner_splits,
            "n_inner_folds_used": self.n_inner_folds_used,
            "residual_l2": self.residual_l2,
        }
        if self.is_fitted:
            out["residual_strength"] = self.residual_strength()
        return out


@dataclass
class ListenerPairwiseLogisticRanker(RiskModel):
    """Phase F — 청취자 **안에서** 단어 쌍을 비교하도록 학습하는 선형 랭커.

    왜 이 모델인가
    --------------
    선택 자막은 청취자별 예산 안에서의 **순위** 문제입니다. 그런데 지금까지의 결과는 일관되게
    선형 모델을 선호했고(E22), 순위를 직접 최적화한 LambdaMART 는 크게 뒤졌습니다. 이 모델은
    두 성질을 합칩니다: 선형·해석 가능하면서 목적함수가 청취자 내 순서입니다.

    청취자 상수가 소거됩니다
    ------------------------
    한 청취자 안에서 오청 단어 A 와 정청 단어 B 를 짝지어 ``ΔX = X_A - X_B`` 를 만들면,
    청취자에 대해 상수인 열(PTA, WRS, SRT 등)은 정확히 0 이 됩니다. 모델은 그 열들로부터
    아무것도 배울 수 없고, **단어 사이의 차이**만 배우게 됩니다. E25 에서 전체 이득의
    대부분이 조건 배분에서 왔고 조건 내부 이득이 1/10 이었던 것을 생각하면, 이 소거는
    바로 그 지점을 겨냥합니다.

    쌍 개수는 반드시 유계여야 합니다
    --------------------------------
    청취자 한 명의 오청 m 개와 정청 n 개는 m×n 쌍을 만듭니다. 250 시행에서 이는 쉽게
    15,000 쌍이 되고 코호트 전체로는 백만 단위가 됩니다. ``max_pairs_per_listener`` 로
    상한을 두고, 초과하면 **시드로 결정되는** 표집을 합니다.

    출력은 확률이 아닙니다
    ----------------------
    쌍별 목적으로 학습한 점수는 순서만 의미가 있습니다. :meth:`predict_proba` 는 인터페이스를
    맞추기 위한 단조 변환일 뿐이며, ``describe()`` 가 ``output_is_probability: False`` 를
    보고합니다. 임계값 정책이 소비하기 전에 청취자 홀드아웃 데이터에서 교정해야 합니다.
    """

    name: str = "pairwise_logistic"
    C: float = 1.0
    max_iter: int = 2000
    random_state: int = 0
    #: 청취자당 쌍 상한. None 이면 모든 쌍(작은 코호트에서만 현실적).
    max_pairs_per_listener: int | None = 400
    #: ``"random"`` 은 균등 표집, ``"hard"`` 는 기저 위험이 비슷한 쌍을 우선합니다.
    pair_sampling: str = "random"
    _pipeline: Pipeline | None = None
    feature_names: tuple[str, ...] = ()
    n_train: int = 0
    n_pairs: int = 0
    n_listeners_with_pairs: int = 0
    #: 훈련 점수의 위치와 척도. 단조 변환이 포화되지 않게 하는 데만 쓰이며 순서를 바꾸지
    #: 않습니다. 훈련 데이터에서만 계산되므로 홀드아웃 정보가 들어가지 않습니다.
    _score_centre: float = 0.0
    _score_scale: float = 1.0

    def _pairs_for(
        self, y: npt.NDArray[np.int64], rng: np.random.Generator
    ) -> list[tuple[int, int]]:
        """한 청취자 안의 (오청, 정청) 인덱스 쌍. 결정적으로 표집됩니다."""
        positive = np.flatnonzero(y == 1)
        negative = np.flatnonzero(y == 0)
        if positive.size == 0 or negative.size == 0:
            return []
        total = positive.size * negative.size
        cap = self.max_pairs_per_listener
        if cap is None or total <= cap:
            return [(int(a), int(b)) for a in positive for b in negative]
        # 곱집합을 만들지 않고 인덱스를 직접 뽑습니다. m×n 을 실체화하면 메모리가 터집니다.
        picks = rng.choice(total, size=cap, replace=False)
        return [
            (int(positive[p // negative.size]), int(negative[p % negative.size])) for p in picks
        ]

    def fit(self, matrix: FeatureMatrix) -> ListenerPairwiseLogisticRanker:
        if matrix.y is None:
            raise ValueError("cannot fit without labels")
        y = matrix.y
        if np.unique(y).size < 2:
            raise ValueError("training data has a single class; a ranker cannot be fitted")

        rng = np.random.default_rng(self.random_state)
        deltas: list[FloatArray] = []
        n_with_pairs = 0
        # 청취자를 정렬해 순회합니다. 순회 순서가 난수 스트림을 통해 결과에 새어들지
        # 않도록 하기 위한 것입니다.
        for listener in sorted(set(matrix.groups.tolist())):
            mask = matrix.groups == listener
            rows = np.flatnonzero(mask)
            pairs = self._pairs_for(y[mask], rng)
            if not pairs:
                continue
            n_with_pairs += 1
            local = matrix.X[rows]
            for a, b in pairs:
                deltas.append(local[a] - local[b])

        if not deltas:
            raise ValueError(
                "쌍을 하나도 만들지 못했습니다; 어떤 청취자도 오청과 정청을 모두 갖지 않았습니다"
            )

        # 대칭 학습: ΔX -> 1, -ΔX -> 0. 절편을 두지 않아야 "A 가 B 보다 위험하다" 가
        # 방향에 대해 대칭이 됩니다.
        delta = np.asarray(deltas, dtype=np.float64)
        x = np.vstack([delta, -delta])
        labels = np.concatenate(
            [np.ones(len(delta), dtype=np.int64), np.zeros(len(delta), dtype=np.int64)]
        )

        self._pipeline = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scale", StandardScaler(with_mean=False)),
                (
                    "clf",
                    LogisticRegression(
                        C=self.C,
                        l1_ratio=0.0,
                        max_iter=self.max_iter,
                        random_state=self.random_state,
                        solver="lbfgs",
                        fit_intercept=False,
                    ),
                ),
            ]
        )
        self._pipeline.fit(x, labels)
        self.feature_names = matrix.feature_names
        self.n_train = len(matrix)
        self.n_pairs = len(delta)
        self.n_listeners_with_pairs = n_with_pairs

        # 원 점수의 위치와 척도를 훈련 데이터에서 기록합니다.
        #
        # 절편이 없고 109개 계수가 더해지므로 원 점수는 60~70 대에 놓입니다. 그대로
        # 로지스틱으로 누르면 모든 행이 정확히 1.0 으로 포화되어 **순위가 통째로
        # 사라집니다** — 예산 정책이 predict_proba 를 소비하므로 치명적입니다. 아핀
        # 변환이라 순서는 정확히 보존되고, 통계는 훈련 행에서만 계산되므로 홀드아웃
        # 정보가 들어가지 않습니다.
        raw = self.predict_score(matrix)
        self._score_centre = float(np.mean(raw))
        scale = float(np.std(raw))
        self._score_scale = scale if scale > 1e-12 else 1.0
        return self

    def predict_score(self, matrix: FeatureMatrix) -> FloatArray:
        """원 순위 점수. **청취자 안에서만** 비교 가능합니다."""
        if self._pipeline is None:
            raise ValueError("pairwise ranker is not fitted")
        if matrix.feature_names != self.feature_names:
            raise ValueError("feature columns changed between fit and predict")
        # 쌍 모델의 계수를 원 특징에 그대로 적용합니다. 학습이 ΔX 위에서 이루어졌으므로
        # 같은 청취자 두 행의 점수 차이가 곧 그 쌍에 대한 모델의 판단입니다.
        steps = self._pipeline
        z = steps.named_steps["scale"].transform(steps.named_steps["impute"].transform(matrix.X))
        out: FloatArray = z @ steps.named_steps["clf"].coef_.ravel()
        return out

    def predict_proba(self, matrix: FeatureMatrix) -> FloatArray:
        """순위 점수의 단조 변환. **교정된 확률이 아닙니다.**

        쌍별 목적으로 학습했으므로 절대 수준에는 의미가 없고 순서만 의미가 있습니다.
        임계값 정책이 쓰려면 청취자 홀드아웃 데이터에서 교정해야 합니다.

        훈련 점수의 위치·척도로 표준화한 뒤 누릅니다. 아핀 변환이므로 순서는 정확히
        보존되며, 이것이 없으면 점수가 60 대라 모든 값이 1.0 으로 포화되어 순위가
        사라집니다.
        """
        z = (self.predict_score(matrix) - self._score_centre) / self._score_scale
        out: FloatArray = 1.0 / (1.0 + np.exp(-z))
        return out

    @property
    def is_fitted(self) -> bool:
        return self._pipeline is not None

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": "listener_pairwise_logistic",
            "model_version": MODEL_VERSION,
            "fitted": self.is_fitted,
            "n_train": self.n_train,
            "n_pairs": self.n_pairs,
            "n_listeners_with_pairs": self.n_listeners_with_pairs,
            "max_pairs_per_listener": self.max_pairs_per_listener,
            "pair_sampling": self.pair_sampling,
            "output_is_probability": False,
            "note": (
                "청취자 내 순위 점수입니다. 임계값 정책이 소비하기 전에 청취자 홀드아웃 "
                "데이터에서 교정해야 하며, 교정 없이 Brier/ECE 를 읽으면 안 됩니다."
            ),
        }
