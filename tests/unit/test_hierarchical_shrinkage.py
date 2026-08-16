"""Phase D18 — 집단 사전분포 축소(shrinkage)와 그 누출 방지 장치.

이 파일에서 가장 중요한 것은 성능이 아니라 **누출 방지**입니다. 청취자 단위로 폴드를
나누는 것만으로는 부족합니다. 전체 코호트에서 추정한 집단 사전분포는 홀드아웃 청취자의
응답을 학습 표현 안으로 실어 나르고, 그 결과 모든 지표가 부풀려지지만 그룹 분할 검사는
아무것도 잡아내지 못합니다. 따라서 사전분포가 폴드마다 **훈련 청취자만으로** 다시
적합되는지를 직접 증명합니다.
"""

from __future__ import annotations

import numpy as np
import pytest

from audire.confusion import CalibrationTrial, ConfusionProfile
from audire.eval import LeakageError, assert_prior_fitted_on_train_only, evaluate_arm
from audire.eval.ablation import cohort_matrix
from audire.eval.splits import listener_folds
from audire.hangul.inventory import Position
from audire.risk import FeatureSpec
from audire.risk.hierarchical import (
    DEFAULT_GROUP_ALPHA,
    apply_group_prior,
    fit_group_prior,
    shrinkage_report,
)
from audire.sim import SimulationConfig, build_cohort

SPARSE = SimulationConfig(
    name="shrink-sparse",
    n_listeners=20,
    # 교정 시행을 적게 두어야 관측되지 않은 행이 많이 생기고, 축소가 실제로 할 일이
    # 생깁니다.
    n_calibration_trials=25,
    n_word_trials=40,
    seeds=[7],
)


@pytest.fixture(scope="module")
def cohort():
    return build_cohort(SPARSE, 7)


def _profile(listener_id: str, pairs: list[tuple[str, str]], *, synthetic: bool = True):
    return ConfusionProfile.from_trials(
        listener_id,
        [
            CalibrationTrial(stimulus_id=f"s{i}", target=t, response=r)
            for i, (t, r) in enumerate(pairs)
        ],
        is_synthetic=synthetic,
    )


# --------------------------------------------------------------------- 사전분포 추정 자체


def test_prior_records_exactly_who_contributed(cohort):
    """`fitted_from` 은 감사 기록입니다. 실제 기여자와 정확히 일치해야 합니다."""
    subset = cohort.records[:6]
    prior = fit_group_prior([r.estimated_confusion for r in subset])
    assert prior.fitted_from == tuple(sorted(r.listener_id for r in subset))
    assert prior.n_listeners == 6


def test_prior_rejects_empty_cohort():
    with pytest.raises(ValueError, match="빈 목록"):
        fit_group_prior([])


def test_prior_refuses_to_mix_synthetic_and_observed():
    """합성 증거를 실측 청취자의 사전분포로 세탁하는 경로를 차단합니다."""
    a = _profile("syn", [("각", "각")], synthetic=True)
    b = _profile("obs", [("각", "간")], synthetic=False)
    with pytest.raises(ValueError, match="합성"):
        fit_group_prior([a, b])


def test_prior_rows_are_proper_distributions(cohort):
    prior = fit_group_prior([r.estimated_confusion for r in cohort.records])
    for position in (Position.ONSET, Position.NUCLEUS, Position.CODA):
        matrix = prior.matrices[position]
        assert np.allclose(matrix.sum(axis=1), 1.0)
        assert (matrix >= 0).all()


# ----------------------------------------------------------------------------- 축소의 성질


def test_shrinkage_leaves_raw_counts_untouched(cohort):
    """원자료는 보존됩니다. 축소는 평활 명세만 바꿉니다."""
    record = cohort.records[0]
    prior = fit_group_prior([r.estimated_confusion for r in cohort.records[1:]])
    shrunk = apply_group_prior(record.estimated_confusion, prior)
    for position in (Position.ONSET, Position.NUCLEUS, Position.CODA):
        assert np.array_equal(
            shrunk.matrix(position).counts, record.estimated_confusion.matrix(position).counts
        )
    assert shrunk.n_trials == record.estimated_confusion.n_trials


def test_shrinkage_moves_unobserved_rows_more_than_well_observed(cohort):
    """정성적 요구사항: 증거가 없는 음소는 사전분포를 따르고, 많은 음소는 버팁니다.

    증거량을 직접 통제한 프로파일을 씁니다. 시뮬레이션 코호트에 의존하면 '관측 10회 이상'
    행이 우연히 하나도 없어 테스트가 조용히 건너뛰어지고, 요구사항은 검증되지 않은 채로
    통과한 것처럼 보입니다.
    """
    # ㄱ 초성은 40회 관측(그중 8회 오청), 나머지 초성은 한 번도 제시되지 않음.
    pairs = [("각", "각")] * 32 + [("각", "닥")] * 8
    subject = _profile("L-controlled", pairs)
    prior = fit_group_prior([r.estimated_confusion for r in cohort.records])
    report = shrinkage_report(subject, apply_group_prior(subject, prior), Position.ONSET)

    unobserved = report["mean_move_unobserved"]
    observed = report["mean_move_well_observed"]
    assert not np.isnan(unobserved) and not np.isnan(observed), "두 범주가 모두 존재해야 합니다"
    assert unobserved > observed
    # 잘 관측된 행은 거의 움직이지 않아야 합니다: 40회 관측 대 alpha=5.
    assert observed < 0.1


def test_shrinkage_records_its_own_provenance(cohort):
    record = cohort.records[0]
    prior = fit_group_prior([r.estimated_confusion for r in cohort.records[1:]])
    shrunk = apply_group_prior(record.estimated_confusion, prior, alpha=3.0)
    meta = shrunk.provenance["shrinkage"]
    assert meta["alpha"] == 3.0
    assert meta["prior"]["n_listeners"] == len(cohort.records) - 1
    # 사전분포 기여자가 산출물에 남아 있어야 사후에 누출을 감사할 수 있습니다.
    assert record.listener_id not in meta["prior"]["fitted_from"]


def test_larger_alpha_pulls_further_toward_the_prior(cohort):
    """alpha 는 총 의사관측수이므로, 커질수록 사후분포는 사전분포에 단조 수렴합니다.

    비교 기준은 반드시 **사전분포까지의 거리**여야 합니다. 균일평활 추정값까지의 거리로
    재면 사전분포가 우연히 균일값 근처에 있을 때 부호가 뒤집혀 성질을 잘못 검증합니다.
    """
    record = cohort.records[0]
    prior = fit_group_prior([r.estimated_confusion for r in cohort.records[1:]])
    matrix = record.estimated_confusion.matrix(Position.ONSET)
    target = matrix.target_labels[0]
    prior_p = float(
        prior.matrices[Position.ONSET][
            matrix.target_labels.index(target), matrix.perceived_labels.index(target)
        ]
    )

    distances = [
        abs(
            apply_group_prior(record.estimated_confusion, prior, alpha=a)
            .matrix(Position.ONSET)
            .p_correct(target)
            - prior_p
        )
        for a in (1.0, 5.0, 50.0, 500.0)
    ]
    assert distances == sorted(distances, reverse=True)
    assert distances[-1] < distances[0]


def test_smoothing_spec_survives_serialisation_round_trip(cohort):
    """P0.1 의 왕복 안전성이 명시적 집단 사전분포에도 적용되는지 확인합니다."""
    record = cohort.records[0]
    prior = fit_group_prior([r.estimated_confusion for r in cohort.records[1:]])
    shrunk = apply_group_prior(record.estimated_confusion, prior)

    restored = ConfusionProfile.from_dict(shrunk.to_dict())
    for position in (Position.ONSET, Position.NUCLEUS, Position.CODA):
        assert np.allclose(
            restored.matrix(position).probabilities(), shrunk.matrix(position).probabilities()
        )
        assert restored.matrix(position).smoothing.kind == "explicit"


# ------------------------------------------------------------------------------- 누출 방지


def test_guard_fires_when_prior_saw_a_held_out_listener():
    groups = np.array(["L1", "L1", "L2", "L2", "L3", "L3"])
    test_idx = np.array([4, 5], dtype=np.int64)
    # 정상: 훈련 청취자만 기여.
    assert_prior_fitted_on_train_only(("L1", "L2"), groups, test_idx)
    # 누출: 홀드아웃 청취자가 사전분포에 포함됨.
    with pytest.raises(LeakageError, match="held-out"):
        assert_prior_fitted_on_train_only(("L1", "L2", "L3"), groups, test_idx)


def test_guard_fires_on_a_cohort_wide_prior(cohort):
    """'전체 코호트로 사전분포를 적합한다' 는 가장 자연스러운 구현이 곧 누출입니다."""
    matrix = cohort_matrix(
        cohort, FeatureSpec.arm("clinical_plus_confusion", speakers=("male", "female", "unknown"))
    )
    fold = listener_folds(matrix.groups, matrix.y, n_splits=4, stratify=True, seed=0)[0]
    all_listeners = fit_group_prior([r.estimated_confusion for r in cohort.records])
    with pytest.raises(LeakageError):
        assert_prior_fitted_on_train_only(all_listeners.fitted_from, matrix.groups, fold.test_idx)


def test_evaluate_arm_refits_the_prior_per_fold(cohort):
    """하니스가 실제로 폴드마다 다시 적합하는지 — 기록된 산출물로 증명합니다."""
    result = evaluate_arm(
        cohort,
        "clinical_plus_confusion_rich",
        "logistic",
        seed=0,
        n_splits=4,
        n_bootstrap=25,
        group_shrinkage=True,
    )
    logged = result.model_description["group_shrinkage"]
    assert len(logged) == 4, "폴드마다 하나씩 기록되어야 합니다"
    assert [entry["fold"] for entry in logged] == [0, 1, 2, 3]

    matrix_groups = result.groups
    folds = listener_folds(matrix_groups, result.y_true, n_splits=4, stratify=True, seed=0)
    for entry, fold in zip(logged, folds, strict=True):
        contributors = set(entry["fitted_from"])
        # 핵심 주장: 기여자 집합 == 훈련 청취자 집합. 홀드아웃은 한 명도 없습니다.
        assert contributors == set(fold.train_listeners)
        assert not (contributors & set(fold.test_listeners))

    # 폴드마다 사전분포가 실제로 달라야 합니다. 모두 같다면 한 번만 적합한 것입니다.
    assert len({tuple(sorted(e["fitted_from"])) for e in logged}) == 4


def test_shrinkage_does_not_disturb_row_alignment(cohort):
    """짝지은 대조는 행 정렬에 의존합니다. 축소는 표현만 바꾸고 순서는 건드리지 않습니다."""
    spec = FeatureSpec.arm("confusion_only", speakers=("male", "female", "unknown"))
    plain = cohort_matrix(cohort, spec)
    prior = fit_group_prior([r.estimated_confusion for r in cohort.records])
    shrunk = cohort_matrix(
        cohort,
        spec,
        profiles={
            r.listener_id: apply_group_prior(r.estimated_confusion, prior) for r in cohort.records
        },
    )
    assert np.array_equal(plain.groups, shrunk.groups)
    assert np.array_equal(plain.y, shrunk.y)
    assert plain.feature_names == shrunk.feature_names
    # 표현은 달라져야 합니다. 같다면 축소가 아무 일도 하지 않은 것입니다.
    assert not np.allclose(plain.X, shrunk.X)


def test_shrinkage_off_is_the_default_and_changes_nothing(cohort):
    """기본값은 꺼짐입니다. 기존에 기록된 결과의 의미가 조용히 바뀌지 않습니다."""
    plain = evaluate_arm(
        cohort, "clinical_plus_confusion", "logistic", seed=0, n_splits=4, n_bootstrap=25
    )
    assert "group_shrinkage" not in plain.model_description


def test_default_alpha_is_documented_and_moderate():
    """alpha 는 총 의사관측수입니다. 50회 관측이 사전분포를 압도할 수 있어야 합니다."""
    assert 1.0 <= DEFAULT_GROUP_ALPHA <= 10.0
