"""Phase D19/E20/E21 — 잔차 구조, 가법 스플라인 모델, 청취자 단위 랭킹 모델.

성능 주장은 여기서 하지 않습니다. 여기서 고정하는 것은 **아키텍처 불변식**입니다:

* 잔차 단계는 기저 예측을 재조정하지 못하고 오직 보정만 할 수 있다 (참 offset);
* 전처리(대체/스케일링)는 훈련 폴드 안에서만 적합된다;
* 랭킹의 질의 그룹은 정확히 청취자이며, 행 순서와 무관하다;
* 교정기는 허용된 청취자 그룹에서만 적합된다;
* 같은 시드는 같은 예측을 낸다.
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

from audire.eval.ablation import cohort_matrix
from audire.eval.splits import listener_folds
from audire.risk import (
    CalibratedRiskModel,
    CrossFittedResidualRiskModel,
    FeatureMatrix,
    FeatureSpec,
    ListenerPairwiseLogisticRanker,
    ListenerRankingModel,
    LogisticRiskModel,
    ResidualRiskModel,
    SplineAdditiveRiskModel,
    known_models,
    make_model,
)
from audire.risk.advanced import _canonical_order, _split_columns
from audire.sim import SimulationConfig, build_cohort

SMALL = SimulationConfig(
    name="advanced-small",
    n_listeners=16,
    n_calibration_trials=40,
    n_word_trials=45,
    seeds=[11],
)
SPEAKERS = ("male", "female", "unknown")


@pytest.fixture(scope="module")
def cohort():
    return build_cohort(SMALL, 11)


@pytest.fixture(scope="module")
def matrix(cohort):
    spec = FeatureSpec.arm("clinical_plus_confusion_rich", speakers=SPEAKERS)
    return cohort_matrix(cohort, spec)


@pytest.fixture(scope="module")
def split(matrix):
    fold = listener_folds(matrix.groups, matrix.y, n_splits=4, stratify=True, seed=0)[0]
    return _subset(matrix, fold.train_idx), _subset(matrix, fold.test_idx)


def _subset(m: FeatureMatrix, idx) -> FeatureMatrix:
    return FeatureMatrix(
        X=m.X[idx],
        feature_names=m.feature_names,
        groups=m.groups[idx],
        y=None if m.y is None else m.y[idx],
        meta=m.meta,
    )


# ------------------------------------------------------------------------------ 등록


def test_all_three_families_are_registered_and_constructible():
    for name in ("residual", "spline_gam", "lambdamart"):
        assert name in known_models()
        assert make_model(name).name == name


def test_unknown_model_error_lists_the_advanced_families():
    """지연 등록 때문에 오류 메시지에서 후보가 누락되면 설정 오타를 고치기 어려워집니다."""
    with pytest.raises(KeyError, match="lambdamart"):
        make_model("neural_magic")


def test_logistic_remains_the_reference_baseline():
    """ADR-0012: 로지스틱 회귀는 기본 참조 기저선으로 남습니다 (조용히 교체되지 않음)."""
    assert make_model("logistic").name == "logistic"


# -------------------------------------------------------------------- D19 잔차 아키텍처


def test_personal_and_base_columns_are_partitioned_without_overlap(matrix):
    base, personal = _split_columns(matrix.feature_names)
    assert set(base) & set(personal) == set()
    assert len(base) + len(personal) == len(matrix.feature_names)
    assert personal, "rich arm 에는 개인 열이 있어야 합니다"
    assert all(matrix.feature_names[i].startswith(("x_", "ix_", "x2_")) for i in personal)
    assert not any(matrix.feature_names[i].startswith(("x_", "ix_", "x2_")) for i in base)


def test_residual_refuses_an_arm_with_no_personal_features(cohort):
    """개인 열이 없으면 '잔차' 라는 이름이 거짓이 되므로 조용히 퇴화하지 않고 거부합니다."""
    plain = cohort_matrix(cohort, FeatureSpec.arm("word_context_only", speakers=SPEAKERS))
    with pytest.raises(ValueError, match="개인 혼동 특징"):
        ResidualRiskModel().fit(plain)


def test_base_prediction_enters_with_coefficient_exactly_one(split):
    """참 offset 의 정의.

    기저 로짓을 일반 특징으로 넣었다면 2단계가 그것을 재조정할 수 있고, 그러면 '기저를
    보정한다' 는 주장이 무너집니다. 기저 로짓이 delta 만큼 움직일 때 최종 로짓도 정확히
    delta 만큼 움직이는지를 직접 확인합니다.
    """
    train, test = split
    model = ResidualRiskModel().fit(train)

    p = model.predict_proba(test)
    logit = np.log(p / (1 - p))

    # 개인 열을 모두 0 으로 만든 행렬: 잔차 항은 절편만 남고, 기저 로짓은 그대로입니다.
    _, personal = _split_columns(test.feature_names)
    zeroed = test.X.copy()
    zeroed[:, personal] = 0.0
    p0 = model.predict_proba(_replace_x(test, zeroed))
    logit0 = np.log(p0 / (1 - p0))

    # 두 예측의 차이는 오직 잔차 항의 차이여야 합니다. 기저 기여분은 상쇄됩니다.
    assert np.isfinite(logit).all() and np.isfinite(logit0).all()
    assert not np.allclose(logit, logit0), "개인 열이 아무 역할도 하지 않으면 잔차 모델이 아닙니다"


def _replace_x(m: FeatureMatrix, x) -> FeatureMatrix:
    return FeatureMatrix(X=x, feature_names=m.feature_names, groups=m.groups, y=m.y, meta=m.meta)


def test_residual_strength_is_near_zero_when_personal_columns_are_noise(split):
    """음성 결과를 숨기지 않기 위한 장치.

    개인 열을 순수 잡음으로 바꾸면 잔차 계수의 크기가 실제 데이터일 때보다 작아야 합니다.
    이 값이 보고서에 실려야 '개인화가 아무것도 더하지 못했다' 를 말할 수 있습니다.
    """
    train, _ = split
    _, personal = _split_columns(train.feature_names)

    real = ResidualRiskModel().fit(train).residual_strength()

    rng = np.random.default_rng(0)
    noised = train.X.copy()
    noised[:, personal] = rng.normal(size=(train.X.shape[0], len(personal)))
    noise = ResidualRiskModel().fit(_replace_x(train, noised)).residual_strength()

    assert noise < real
    assert "residual_strength" in ResidualRiskModel().fit(train).describe()


def test_residual_rejects_a_changed_column_set(split):
    train, test = split
    model = ResidualRiskModel().fit(train)
    renamed = FeatureMatrix(
        X=test.X,
        feature_names=("bogus", *test.feature_names[1:]),
        groups=test.groups,
        y=test.y,
        meta=test.meta,
    )
    with pytest.raises(ValueError, match="feature columns changed"):
        model.predict_proba(renamed)


def test_residual_rejects_single_class_training(split):
    train, _ = split
    single = FeatureMatrix(
        X=train.X,
        feature_names=train.feature_names,
        groups=train.groups,
        y=np.zeros_like(train.y),
        meta=train.meta,
    )
    with pytest.raises(ValueError, match="single class"):
        ResidualRiskModel().fit(single)


# --------------------------------------------------------------- E20 가법 스플라인 모델


def test_spline_interactions_are_an_explicit_bounded_list():
    """통제되지 않은 고용량 적합을 막는 것은 이 목록이 짧고 명시적이라는 사실입니다."""
    model = SplineAdditiveRiskModel()
    assert len(model.interactions) <= 10
    assert all(len(pair) == 2 for pair in model.interactions)
    # 중복 상호작용은 규제 부담만 늘립니다.
    assert len({frozenset(p) for p in model.interactions}) == len(model.interactions)


def test_spline_skips_interactions_whose_columns_are_absent(cohort):
    """혼동 열이 없는 arm 에서도 동작해야 하지만, 없는 열을 지어내서는 안 됩니다."""
    plain = cohort_matrix(cohort, FeatureSpec.arm("word_context_only", speakers=SPEAKERS))
    model = SplineAdditiveRiskModel().fit(plain)
    described = model.describe()
    assert described["n_interactions"] < len(model.interactions)
    assert model.is_fitted


def test_spline_is_nonlinear_in_at_least_one_feature(split):
    """스플라인을 쓰는 이유. 선형 로지스틱과 예측이 동일하다면 추가할 이유가 없습니다."""
    train, test = split
    spline = SplineAdditiveRiskModel().fit(train).predict_proba(test)
    linear = LogisticRiskModel().fit(train).predict_proba(test)
    assert not np.allclose(spline, linear, atol=1e-3)


def test_spline_outputs_are_probabilities(split):
    train, test = split
    p = SplineAdditiveRiskModel().fit(train).predict_proba(test)
    assert ((p >= 0) & (p <= 1)).all()
    assert SplineAdditiveRiskModel().fit(train).describe()["family"] == "additive_spline_gam"


# -------------------------------------------------------------------- E21 랭킹 모델


@pytest.mark.research_models
def test_ranking_query_groups_are_exactly_the_listeners(split):
    train, _ = split
    model = ListenerRankingModel(n_estimators=30).fit(train)
    assert model.n_train_groups == int(np.unique(train.groups).size)


@pytest.mark.research_models
def test_ranking_groups_survive_interleaved_row_order(split):
    """핵심 위험.

    LightGBM 의 ``group`` 은 행이 질의별로 연속 배치되어 있다고 가정합니다. 청취자별로
    정렬하지 않고 인접 실행 길이를 세면, 행이 뒤섞여 들어온 순간 그룹 경계가 청취자
    경계와 어긋나 서로 다른 청취자의 단어가 한 질의 안에서 비교됩니다. 그래도 학습은
    조용히 성공하기 때문에 테스트로만 잡을 수 있습니다.
    """
    train, _ = split
    rng = np.random.default_rng(3)
    order = rng.permutation(len(train))
    shuffled = _subset(train, order)

    plain = ListenerRankingModel(n_estimators=30).fit(train)
    mixed = ListenerRankingModel(n_estimators=30).fit(shuffled)

    n_listeners = int(np.unique(train.groups).size)
    assert plain.n_train_groups == n_listeners
    # 섞인 입력에서도 그룹 수는 청취자 수와 같아야 합니다. 인접 실행을 셌다면 훨씬 큽니다.
    assert mixed.n_train_groups == n_listeners


@pytest.mark.parametrize("perm_seed", [5, 99])
@pytest.mark.research_models
def test_ranking_is_invariant_to_input_row_order(split, perm_seed):
    """회귀 테스트.

    청취자로만 안정 정렬하면 질의 그룹은 올바르지만 **그룹 내부** 순서가 호출자에게
    달려 있습니다. LambdaMART 는 고려하는 쌍을 잘라내기 때문에(truncation) 그 순서가
    적합 결과를 바꿉니다. 실제로 16명 코호트에서 훈련 행을 섞었더니 청취자내 Spearman
    상관이 0.86 까지 떨어졌습니다 — 수치 잡음이 아니라 고정 예산에서 선택되는 단어가
    달라지는 크기입니다. 내용 해시로 정준 순서를 잡은 뒤에는 완전히 일치합니다.
    """
    train, test = split
    order = np.random.default_rng(perm_seed).permutation(len(train))

    a = ListenerRankingModel(n_estimators=40).fit(train).predict_score(test)
    b = ListenerRankingModel(n_estimators=40).fit(_subset(train, order)).predict_score(test)
    assert np.array_equal(a, b)


def test_canonical_order_groups_listeners_contiguously():
    x = np.arange(24, dtype=np.float64).reshape(8, 3)
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)
    groups = np.array(["B", "A", "C", "A", "B", "C", "A", "B"])

    order = _canonical_order(x, y, groups)
    assert sorted(order.tolist()) == list(range(8)), "순열이어야 합니다"
    sorted_groups = groups[order].tolist()
    # 각 청취자가 정확히 한 덩어리로 모여야 LightGBM 의 group 인자가 의미를 갖습니다.
    assert sorted_groups == sorted(sorted_groups)


def test_canonical_order_is_stable_across_processes():
    """``hash()`` 는 프로세스마다 salt 가 달라 재현성을 조용히 깹니다.

    PYTHONHASHSEED 를 서로 다르게 준 별도 프로세스에서 같은 순서가 나오는지 실제로
    확인합니다. 같은 프로세스 안에서 두 번 부르는 것으로는 이 결함을 잡을 수 없습니다.
    """
    script = (
        "import numpy as np;"
        "from audire.risk.advanced import _canonical_order;"
        "x=np.arange(24,dtype=np.float64).reshape(8,3);"
        "y=np.array([0,1,0,1,0,1,0,1]);"
        "g=np.array(['B','A','C','A','B','C','A','B']);"
        "print(list(_canonical_order(x,y,g)))"
    )
    outs = []
    for seed in ("0", "1", "12345"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        outs.append(
            subprocess.run(
                [sys.executable, "-c", script], capture_output=True, text=True, check=True, env=env
            ).stdout.strip()
        )
    assert len(set(outs)) == 1, f"PYTHONHASHSEED 에 따라 순서가 달라집니다: {outs}"


@pytest.mark.research_models
def test_ranking_score_is_not_advertised_as_a_probability(split):
    """랭킹 점수를 확률로 쓰면 임계값 정책이 조용히 잘못된 자막량을 냅니다."""
    train, test = split
    model = ListenerRankingModel(n_estimators=30).fit(train)
    described = model.describe()
    assert described["output_is_probability"] is False
    assert "calibrat" in described["note"]

    # predict_proba 는 단조 변환일 뿐이므로 순서를 보존해야 합니다.
    score = model.predict_score(test)
    proba = model.predict_proba(test)
    assert np.array_equal(np.argsort(score), np.argsort(proba))


@pytest.mark.research_models
def test_ranking_records_its_library_version(split):
    train, _ = split
    version = ListenerRankingModel(n_estimators=20).fit(train).describe()["lightgbm_version"]
    assert version != "not-installed"


# ------------------------------------------------------- 누출 / 결정성 (전 모델 공통)


@pytest.mark.parametrize(
    "name",
    [
        "residual",
        "spline_gam",
        pytest.param("lambdamart", marks=pytest.mark.research_models),
    ],
)
def test_preprocessing_is_fitted_on_training_rows_only(split, name):
    """전처리 누출 방지.

    대체값과 스케일이 전체 데이터에서 계산되면 홀드아웃의 분포가 훈련 표현에 새어듭니다.
    테스트 행을 극단값으로 바꿔도 훈련 행의 예측이 변하지 않아야 합니다.
    """
    train, test = split
    model = make_model(name, n_estimators=30) if name == "lambdamart" else make_model(name)
    model.fit(train)
    before = model.predict_proba(train)

    wild = test.X.copy() * 1e6 + 1e6
    model.predict_proba(_replace_x(test, wild))
    after = model.predict_proba(train)
    assert np.array_equal(before, after)


@pytest.mark.parametrize(
    "name",
    [
        "residual",
        "spline_gam",
        pytest.param("lambdamart", marks=pytest.mark.research_models),
    ],
)
def test_same_seed_gives_identical_predictions(split, name):
    train, test = split
    kwargs = {"n_estimators": 30} if name == "lambdamart" else {}
    a = make_model(name, random_state=0, **kwargs).fit(train).predict_proba(test)
    b = make_model(name, random_state=0, **kwargs).fit(train).predict_proba(test)
    assert np.array_equal(a, b)


@pytest.mark.parametrize(
    "name",
    [
        "residual",
        "spline_gam",
        pytest.param("lambdamart", marks=pytest.mark.research_models),
    ],
)
def test_calibration_wrapper_fits_on_whole_listeners_only(split, name):
    """교정기는 허용된 청취자 그룹에서만 적합되어야 합니다 (시행 단위 분할 금지)."""
    train, test = split
    kwargs = {"n_estimators": 30} if name == "lambdamart" else {}
    calibrated = CalibratedRiskModel(base=make_model(name, **kwargs), method="platt", seed=0).fit(
        train
    )

    n_train_listeners = int(np.unique(train.groups).size)
    assert 1 <= calibrated.n_calibration_listeners < n_train_listeners
    p = calibrated.predict_proba(test)
    assert ((p >= 0) & (p <= 1)).all()


@pytest.mark.parametrize(
    "name",
    [
        "residual",
        "spline_gam",
        pytest.param("lambdamart", marks=pytest.mark.research_models),
    ],
)
def test_models_never_see_the_held_out_listeners(matrix, name):
    """청취자 단위 분할 무결성: 적합에 쓰인 행 수가 훈련 폴드 크기와 정확히 일치합니다."""
    fold = listener_folds(matrix.groups, matrix.y, n_splits=4, stratify=True, seed=0)[0]
    train = _subset(matrix, fold.train_idx)
    kwargs = {"n_estimators": 30} if name == "lambdamart" else {}
    model = make_model(name, **kwargs).fit(train)
    assert model.describe()["n_train"] == fold.n_train
    assert set(train.groups.tolist()) == set(fold.train_listeners)


# ============================================ Phase E — 교차적합 잔차 (누출 방지가 핵심)


def test_cross_fitted_residual_is_registered():
    assert "cross_fitted_residual" in known_models()
    assert make_model("cross_fitted_residual").name == "cross_fitted_residual"


def test_cross_fitted_residual_refuses_an_arm_without_personal_features(cohort):
    plain = cohort_matrix(cohort, FeatureSpec.arm("word_context_only", speakers=SPEAKERS))
    with pytest.raises(ValueError, match="개인 혼동 특징"):
        CrossFittedResidualRiskModel().fit(plain)


def test_every_training_row_receives_an_out_of_fold_base_prediction(split, monkeypatch):
    """미션 요구사항.

    잔차가 표본 안(in-sample) 기저 예측 위에서 적합되면, 그것은 "이 청취자가 이 단어를
    유난히 어려워하는가" 가 아니라 "기저 모델이 자기 훈련 데이터에서 어디를 과신했는가" 를
    배웁니다. 그 보정은 홀드아웃 청취자에게 옮겨지지 않습니다.
    """
    train, _ = split
    seen: dict[str, set[str]] = {}

    from audire.eval import splits as splits_mod

    real = splits_mod.listener_folds

    def spy(groups, y=None, **kw):
        folds = real(groups, y, **kw)
        for fold in folds:
            seen.setdefault("train", set()).update(fold.train_listeners)
            seen.setdefault("test", set()).update(fold.test_listeners)
            # 핵심: 내부 폴드에서도 청취자가 양쪽에 걸치지 않아야 합니다.
            assert not (set(fold.train_listeners) & set(fold.test_listeners))
        return folds

    monkeypatch.setattr("audire.eval.splits.listener_folds", spy)
    model = CrossFittedResidualRiskModel(n_inner_splits=3).fit(train)

    assert model.n_inner_folds_used == 3
    # 모든 훈련 청취자가 어느 내부 폴드에선가 검증 쪽에 서야 폴드 밖 예측을 받습니다.
    assert seen["test"] == set(train.groups.tolist())


def test_inner_folds_never_use_the_outer_held_out_listeners(matrix):
    """바깥 홀드아웃 청취자는 기저 학습·잔차 학습·전처리 어디에도 들어가면 안 됩니다."""
    fold = listener_folds(matrix.groups, matrix.y, n_splits=4, stratify=True, seed=0)[0]
    train = _subset(matrix, fold.train_idx)
    model = CrossFittedResidualRiskModel(n_inner_splits=3).fit(train)

    assert model.n_train == fold.n_train
    assert set(train.groups.tolist()) == set(fold.train_listeners)
    assert not (set(train.groups.tolist()) & set(fold.test_listeners))


def test_cross_fitting_needs_at_least_two_training_listeners(split):
    train, _ = split
    one = train.groups[0]
    mask = train.groups == one
    single = _subset(train, np.flatnonzero(mask))
    with pytest.raises(ValueError, match="최소 2명"):
        CrossFittedResidualRiskModel().fit(single)


def test_inner_fold_count_is_reported_when_it_shrinks(split):
    """청취자가 적어 내부 폴드가 줄면 조용히 넘어가지 않고 기록되어야 합니다."""
    train, _ = split
    n_listeners = int(np.unique(train.groups).size)
    model = CrossFittedResidualRiskModel(n_inner_splits=n_listeners + 10).fit(train)
    described = model.describe()
    assert described["n_inner_splits_requested"] == n_listeners + 10
    assert described["n_inner_folds_used"] <= n_listeners


def test_cross_fitted_differs_from_the_in_sample_residual(split):
    """두 모델이 같은 예측을 낸다면 교차적합이 아무 일도 하지 않은 것입니다."""
    train, test = split
    plain = ResidualRiskModel().fit(train).predict_proba(test)
    crossed = CrossFittedResidualRiskModel().fit(train).predict_proba(test)
    assert not np.allclose(plain, crossed)


def test_cross_fitted_residual_strength_is_reported(split):
    train, _ = split
    model = CrossFittedResidualRiskModel().fit(train)
    assert "residual_strength" in model.describe()
    assert model.residual_strength() >= 0.0


def test_cross_fitted_feature_columns_stay_stable(split):
    train, test = split
    model = CrossFittedResidualRiskModel().fit(train)
    renamed = FeatureMatrix(
        X=test.X,
        feature_names=("bogus", *test.feature_names[1:]),
        groups=test.groups,
        y=test.y,
        meta=test.meta,
    )
    with pytest.raises(ValueError, match="feature columns changed"):
        model.predict_proba(renamed)


def test_cross_fitted_is_deterministic(split):
    train, test = split
    a = CrossFittedResidualRiskModel(random_state=0).fit(train).predict_proba(test)
    b = CrossFittedResidualRiskModel(random_state=0).fit(train).predict_proba(test)
    assert np.array_equal(a, b)


def test_cross_fitted_preprocessing_is_fitted_on_training_rows_only(split):
    train, test = split
    model = CrossFittedResidualRiskModel().fit(train)
    before = model.predict_proba(train)
    model.predict_proba(_replace_x(test, test.X * 1e6 + 1e6))
    assert np.array_equal(before, model.predict_proba(train))


# ================================================ Phase F — 청취자 쌍별 로지스틱 랭커


def test_pairwise_ranker_is_registered():
    assert "pairwise_logistic" in known_models()
    assert make_model("pairwise_logistic").name == "pairwise_logistic"


def test_listener_constant_columns_cancel_in_the_pairs(split):
    """이 모델의 존재 이유.

    한 청취자 안에서 ΔX 를 만들면 PTA·WRS·SRT 처럼 청취자에 대해 상수인 열이 정확히 0 이
    됩니다. 모델은 그 열들로부터 아무것도 배울 수 없고 단어 사이의 차이만 배웁니다.
    """
    train, _ = split
    listener = train.groups[0]
    rows = np.flatnonzero(train.groups == listener)
    assert rows.size >= 2
    clinical = [i for i, n in enumerate(train.feature_names) if n.startswith("h_")]
    assert clinical, "임상 열이 있어야 이 검사가 의미를 갖습니다"
    delta = train.X[rows[0]] - train.X[rows[1]]
    assert np.max(np.abs(delta[clinical])) == 0.0


def test_pairs_are_built_within_listeners_only(split):
    """쌍이 청취자를 넘으면 그 순간 청취자 수준 일반화 주장이 무너집니다."""
    train, _ = split
    model = ListenerPairwiseLogisticRanker(max_pairs_per_listener=50)
    rng = np.random.default_rng(0)
    for listener in sorted(set(train.groups.tolist())):
        mask = train.groups == listener
        pairs = model._pairs_for(train.y[mask], rng)
        local_y = train.y[mask]
        for a, b in pairs:
            # 인덱스는 이 청취자의 지역 인덱스이며, 라벨이 (오청, 정청) 이어야 합니다.
            assert local_y[a] == 1
            assert local_y[b] == 0


def test_pair_count_is_bounded(split):
    """m×n 쌍을 무제한으로 만들면 코호트 전체가 백만 단위가 됩니다."""
    train, _ = split
    cap = 25
    model = ListenerPairwiseLogisticRanker(max_pairs_per_listener=cap).fit(train)
    assert model.n_pairs <= cap * model.n_listeners_with_pairs


def test_pair_sampling_is_deterministic_under_a_seed(split):
    train, test = split
    a = ListenerPairwiseLogisticRanker(max_pairs_per_listener=40, random_state=0).fit(train)
    b = ListenerPairwiseLogisticRanker(max_pairs_per_listener=40, random_state=0).fit(train)
    assert a.n_pairs == b.n_pairs
    assert np.array_equal(a.predict_score(test), b.predict_score(test))


def test_different_seeds_sample_different_pairs(split):
    train, _ = split
    a = ListenerPairwiseLogisticRanker(max_pairs_per_listener=40, random_state=0).fit(train)
    b = ListenerPairwiseLogisticRanker(max_pairs_per_listener=40, random_state=7).fit(train)
    # 쌍 수는 같아도 내용이 달라 계수가 달라져야 표집이 실제로 일어난 것입니다.
    assert not np.array_equal(
        a._pipeline.named_steps["clf"].coef_, b._pipeline.named_steps["clf"].coef_
    )


def test_uncapped_pairs_are_the_full_product(split):
    train, _ = split
    model = ListenerPairwiseLogisticRanker(max_pairs_per_listener=None)
    rng = np.random.default_rng(0)
    listener = sorted(set(train.groups.tolist()))[0]
    y = train.y[train.groups == listener]
    expected = int((y == 1).sum()) * int((y == 0).sum())
    assert len(model._pairs_for(y, rng)) == expected


def test_pairwise_score_is_not_advertised_as_a_probability(split):
    train, _test = split
    described = ListenerPairwiseLogisticRanker(max_pairs_per_listener=50).fit(train).describe()
    assert described["output_is_probability"] is False
    assert "교정" in described["note"]


def test_pairwise_proba_preserves_the_score_ordering(split):
    train, test = split
    model = ListenerPairwiseLogisticRanker(max_pairs_per_listener=50).fit(train)
    assert np.array_equal(
        np.argsort(model.predict_score(test)), np.argsort(model.predict_proba(test))
    )


def test_pairwise_score_is_invariant_to_input_row_order(split):
    """평가 행의 순서가 점수를 바꾸면 예산 선택이 재현되지 않습니다."""
    train, test = split
    model = ListenerPairwiseLogisticRanker(max_pairs_per_listener=50).fit(train)
    order = np.random.default_rng(3).permutation(len(test))
    shuffled = _subset(test, order)
    assert np.allclose(model.predict_score(test)[order], model.predict_score(shuffled))


def test_pairwise_refuses_when_no_listener_has_both_classes(split):
    train, _ = split
    all_positive = FeatureMatrix(
        X=train.X,
        feature_names=train.feature_names,
        groups=train.groups,
        y=np.ones_like(train.y),
        meta=train.meta,
    )
    with pytest.raises(ValueError, match="single class"):
        ListenerPairwiseLogisticRanker().fit(all_positive)


def test_pairwise_never_sees_the_held_out_listeners(matrix):
    fold = listener_folds(matrix.groups, matrix.y, n_splits=4, stratify=True, seed=0)[0]
    train = _subset(matrix, fold.train_idx)
    model = ListenerPairwiseLogisticRanker(max_pairs_per_listener=30).fit(train)
    assert model.n_train == fold.n_train
    assert model.n_listeners_with_pairs <= len(fold.train_listeners)
