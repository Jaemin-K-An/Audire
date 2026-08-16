"""공용 픽스처."""

from __future__ import annotations

import pytest

from audire.eval.ablation import cohort_matrix
from audire.risk import FeatureSpec, LogisticRiskModel, WordScorer
from audire.sim import SimulationConfig, build_cohort

_SCORER_CFG = SimulationConfig(
    name="conftest-scorer",
    n_listeners=16,
    n_calibration_trials=40,
    n_word_trials=40,
    seeds=[17],
)


@pytest.fixture(scope="session")
def fitted_scorer() -> WordScorer:
    """`clinical_plus_confusion` arm 으로 적합된 채점기.

    신원 불변식 테스트는 '적합된' 채점기를 요구합니다. 적합되지 않은 모델은 신원 검사에
    도달하기 전에 다른 이유로 거부되어 테스트가 자명하게 통과할 수 있기 때문입니다.
    """
    cohort = build_cohort(_SCORER_CFG, 17)
    spec = FeatureSpec.arm("clinical_plus_confusion", speakers=("male", "female", "unknown"))
    model = LogisticRiskModel().fit(cohort_matrix(cohort, spec))
    return WordScorer(model=model, spec=spec)
