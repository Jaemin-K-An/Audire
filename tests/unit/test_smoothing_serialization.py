"""P0.1 — 평활 명세의 직렬화 왕복 안전성.

혼동행렬은 연구 결과의 입력입니다. 저장했다가 불러온 행렬이 **다른 확률**을 내놓으면
보고된 모든 수치가 무의미해집니다.

이 파일이 강제하는 불변식:

1. `deserialize(serialize(m))` 은 position·라벨·counts·alpha·kind·explicit prior·
   probabilities·row_entropy·p_correct 를 모두 보존한다.
2. explicit prior 가 **조용히 uniform 으로 강등되지 않는다.**
3. 저장된 라벨이 현재 자모 목록과 다르면 **크게 실패한다.** 새 의미론으로 옛 counts 를
   해석하는 것이 가장 위험한 실패 방식이기 때문이다.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from audire.confusion.matrix import ConfusionMatrix, SmoothingSpec
from audire.confusion.profile import ConfusionProfile
from audire.hangul.inventory import Position, categories_for


def _group_prior(position: Position) -> np.ndarray:
    """관측에서 유도한, 균등하지 않은 행 확률 사전분포."""
    group = ConfusionMatrix.empty(position)
    targets = group.target_labels
    perceived = group.perceived_labels
    # 각 목표를 '다음' 범주로 체계적으로 혼동하는 집단 — 균등과 확실히 구별된다.
    for i, t in enumerate(targets):
        wrong = perceived[(i + 1) % len(targets)]
        for _ in range(50):
            group.observe(t, wrong)
    return group.probabilities()


def _matrix_with_explicit_prior(position: Position = Position.ONSET) -> ConfusionMatrix:
    """관측이 있는 행과 없는 행을 모두 갖는, 명시적 사전분포를 쓰는 행렬."""
    prior = _group_prior(position)
    m = ConfusionMatrix.empty(position, SmoothingSpec(alpha=7.5, kind="explicit", prior=prior))
    targets = m.target_labels
    perceived = m.perceived_labels
    # 위치마다 자모 목록이 다르므로 인덱스로 고른다. 마지막 목표는 미관측으로 남겨
    # 사전분포가 그대로 드러나는 행을 확보한다.
    m.observe(targets[0], targets[0], weight=3)
    m.observe(targets[1], perceived[2], weight=2)
    return m


# =========================================================== 왕복 보존


def test_uniform_smoothing_survives_roundtrip() -> None:
    m = ConfusionMatrix.empty(Position.NUCLEUS, SmoothingSpec(alpha=2.25))
    m.observe("ㅏ", "ㅓ", weight=4)

    back = ConfusionMatrix.from_dict(m.to_dict())

    assert back.position is m.position
    assert back.target_labels == m.target_labels
    assert back.perceived_labels == m.perceived_labels
    assert np.array_equal(back.counts, m.counts)
    assert back.smoothing.alpha == pytest.approx(2.25)
    assert back.smoothing.kind == "uniform"
    assert np.allclose(back.probabilities(), m.probabilities())


def test_explicit_prior_survives_roundtrip() -> None:
    """가장 중요한 불변식: explicit prior 가 소실되면 확률이 달라진다."""
    m = _matrix_with_explicit_prior()

    back = ConfusionMatrix.from_dict(m.to_dict())

    assert back.smoothing.kind == "explicit", "explicit prior 가 uniform 으로 강등됐다"
    assert back.smoothing.alpha == pytest.approx(7.5)
    assert back.smoothing.prior is not None
    assert np.allclose(back.smoothing.prior, m.smoothing.prior)
    assert np.allclose(back.probabilities(), m.probabilities())
    assert np.allclose(back.row_entropy(), m.row_entropy())
    observed, unobserved = m.target_labels[0], m.target_labels[-1]
    assert m.n_observations(unobserved) == 0, "미관측 행이 있어야 사전분포 보존을 검증할 수 있다"
    assert back.p_correct(observed) == pytest.approx(m.p_correct(observed))
    assert back.p_correct(unobserved) == pytest.approx(m.p_correct(unobserved))


def test_explicit_prior_actually_differs_from_uniform() -> None:
    """이 테스트가 없으면 위 테스트가 자명하게 통과할 수 있다."""
    m = _matrix_with_explicit_prior()
    uniform = ConfusionMatrix(
        position=m.position, counts=m.counts.copy(), smoothing=SmoothingSpec(alpha=7.5)
    )
    assert not np.allclose(m.probabilities(), uniform.probabilities())


def test_roundtrip_through_actual_json_text() -> None:
    """dict 왕복이 아니라 실제 JSON 문자열을 통과해야 한다."""
    m = _matrix_with_explicit_prior(Position.CODA)
    back = ConfusionMatrix.from_dict(json.loads(json.dumps(m.to_dict(), ensure_ascii=False)))
    assert back.smoothing.kind == "explicit"
    assert np.allclose(back.probabilities(), m.probabilities())


@pytest.mark.parametrize("position", list(Position))
def test_every_position_roundtrips_with_an_explicit_prior(position: Position) -> None:
    m = _matrix_with_explicit_prior(position)
    back = ConfusionMatrix.from_dict(m.to_dict())
    assert np.allclose(back.probabilities(), m.probabilities()), position


def test_profile_roundtrip_preserves_explicit_priors(tmp_path) -> None:
    """프로파일 전체를 디스크에 쓰고 읽어도 계층적 사전분포가 살아 있어야 한다."""
    profile = ConfusionProfile.empty("L001", is_synthetic=True)
    for position in Position:
        profile.matrices[position] = _matrix_with_explicit_prior(position)

    path = tmp_path / "profile.json"
    profile.save_json(path)
    back = ConfusionProfile.load_json(path)

    for position in Position:
        assert back.matrix(position).smoothing.kind == "explicit", position
        assert np.allclose(
            back.matrix(position).probabilities(), profile.matrix(position).probabilities()
        ), position


def test_explicit_smoothing_override_still_wins_on_load() -> None:
    """호출자가 명시적으로 다른 평활을 주면 그것이 우선한다(기존 동작 유지)."""
    m = _matrix_with_explicit_prior()
    override = SmoothingSpec(alpha=0.5, kind="uniform")
    back = ConfusionMatrix.from_dict(m.to_dict(), smoothing=override)
    assert back.smoothing.kind == "uniform"
    assert back.smoothing.alpha == pytest.approx(0.5)


# =========================================================== 라벨 검증


def test_unknown_target_label_is_rejected() -> None:
    """옛 counts 를 새 자모 목록 의미로 재해석하는 것이 가장 위험한 실패다."""
    payload = ConfusionMatrix.empty(Position.ONSET).to_dict()
    payload["target_labels"][0] = "ㆁ"  # 현대 목록에 없는 옛한글
    with pytest.raises(ValueError, match=r"라벨|label"):
        ConfusionMatrix.from_dict(payload)


def test_reordered_labels_are_rejected() -> None:
    """순서가 바뀌면 같은 라벨 집합이라도 counts 의 의미가 완전히 달라진다."""
    payload = ConfusionMatrix.empty(Position.ONSET).to_dict()
    payload["target_labels"][0], payload["target_labels"][1] = (
        payload["target_labels"][1],
        payload["target_labels"][0],
    )
    with pytest.raises(ValueError, match=r"라벨|label|순서|order"):
        ConfusionMatrix.from_dict(payload)


def test_wrong_perceived_label_set_is_rejected() -> None:
    payload = ConfusionMatrix.empty(Position.CODA).to_dict()
    payload["perceived_labels"] = payload["perceived_labels"][:-1]  # NO_RESPONSE 제거
    with pytest.raises(ValueError, match=r"라벨|label|차원|shape"):
        ConfusionMatrix.from_dict(payload)


def test_counts_shape_mismatch_is_rejected() -> None:
    payload = ConfusionMatrix.empty(Position.ONSET).to_dict()
    payload["counts"] = [[0, 0], [0, 0]]
    with pytest.raises(ValueError, match=r"shape|차원"):
        ConfusionMatrix.from_dict(payload)


def test_missing_labels_are_rejected_rather_than_assumed() -> None:
    """라벨이 없는 옛 파일을 '현재 목록일 것'이라고 가정하면 안 된다."""
    payload = ConfusionMatrix.empty(Position.ONSET).to_dict()
    del payload["target_labels"]
    with pytest.raises(ValueError, match=r"라벨|label"):
        ConfusionMatrix.from_dict(payload)


def test_label_validation_accepts_the_current_inventory() -> None:
    """정상 경로가 검증 때문에 막히지 않아야 한다."""
    for position in Position:
        payload = ConfusionMatrix.empty(position).to_dict()
        assert payload["target_labels"] == list(categories_for(position, axis="target"))
        assert payload["perceived_labels"] == list(categories_for(position, axis="perceived"))
        ConfusionMatrix.from_dict(payload)


# =========================================================== 사전분포 무결성


def test_serialized_prior_rows_must_remain_stochastic() -> None:
    """손상된 사전분포를 조용히 받아들이면 확률이 아닌 값이 나온다."""
    m = _matrix_with_explicit_prior()
    payload = m.to_dict()
    payload["smoothing"]["prior"][0] = [0.0] * len(m.perceived_labels)
    with pytest.raises(ValueError, match=r"sum to 1|합"):
        ConfusionMatrix.from_dict(payload)


def test_explicit_kind_without_a_prior_is_rejected() -> None:
    m = _matrix_with_explicit_prior()
    payload = m.to_dict()
    del payload["smoothing"]["prior"]
    with pytest.raises(ValueError, match=r"prior|사전분포"):
        ConfusionMatrix.from_dict(payload)
