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


def pytest_collection_modifyitems(config, items):
    """`research_models` 표시가 붙은 테스트는 extra 가 없으면 건너뜁니다.

    LightGBM 은 선택적 연구 모델 extra 입니다(E22 에서 이 계열이 참조 로지스틱을 이기지
    못했으므로 배포 필수 의존성이 아닙니다). 부재는 오류가 아니라 정상 상태이므로,
    ImportError 로 실패하는 대신 **이유가 적힌 skip** 으로 처리합니다.

    다만 CI 에는 이 extra 를 설치하고 이 테스트만 따로 돌리는 잡이 있습니다. 건너뛰는
    것으로 끝내면 랭킹 모델의 불변식이 아무 데서도 검증되지 않기 때문입니다.
    """
    import importlib.util

    if importlib.util.find_spec("lightgbm") is not None:
        return
    skip = pytest.mark.skip(
        reason="research-models extra 없음: pip install -e '.[research-models]'"
    )
    for item in items:
        if "research_models" in item.keywords:
            item.add_marker(skip)
