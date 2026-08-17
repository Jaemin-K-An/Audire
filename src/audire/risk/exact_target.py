"""Phase D — 목표 음소를 **개별적으로** 지목하는 특징 블록.

기존 블록과 무엇이 다른가
-------------------------
``confusion_features`` 는 ``C_u`` 를 16개 집계로 압축하고, ``confusion_rich_features`` 는
음운 *부류* 수준의 상호작용을 냅니다. 둘 다 "이 청취자가 파열음에 약하다" 는 말할 수
있지만 "이 청취자가 **ㅋ에** 약하다" 는 말할 수 없습니다. 같은 부류 안에서도 음소마다
청취자의 실제 정확도가 크게 다르면, 부류 평균은 그 차이를 지웁니다.

이 모듈은 두 블록을 추가합니다.

``exact_target`` (D1)
    고정된 한국어 음소 목록의 각 목표 음소 φ 마다 한 열:

        단어 안의 φ 출현 횟수  ×  이 청취자의 φ 오류 확률

    출현하지 않는 음소는 0 이므로 열은 희소하지만 **개수는 고정**입니다(68열: 초성 19,
    중성 21, 종성 28). 미션이 경고한 "수천 차원으로 평탄화" 를 피하면서, 아무 집계도
    표현할 수 없는 음소 개별 정보를 넣습니다.

``exact_target_offdiag`` (D2/D3)
    오류 질량이 **어디로** 가는지를 위치별로 요약합니다. 목표 음소마다 오프대각 질량을
    같은 음운 부류 안과 밖으로 나누고, 단어의 구성으로 가중해 위치별로 합칩니다. 여기에
    위치별 기대 오류 수, 증거량, 사후 불확실성을 더합니다(15열).

    이 블록은 Simulator V2 의 기전을 직접 겨냥합니다. V2 는 종성 오류를 잘 복구하고 모음
    오류를 못 하며 같은 부류 안의 치환에 가산점을 줍니다(docs/RESULTS.md §15). 그 구조를
    잡으려면 "이 단어에서 종성 오류가 얼마나 예상되는가" 와 "그 오류가 같은 부류 안에
    머물 가능성이 얼마인가" 를 **위치별로** 알아야 하는데, 기존 ``x2_`` 는 단어 전체로
    평균해 버려 그 구분이 사라집니다.

차원 통제
---------
D1 은 목록이 고정되어 68열, D2/D3 은 15열입니다. 응답 방향(perceived)으로는 평탄화하지
않습니다 — 그렇게 하면 19×20 + 21×22 + 28×29 ≈ 1,600열이 되어 청취자 80명 규모에서
규제 부담만 늘어납니다.
"""

from __future__ import annotations

from typing import Final

import numpy as np

from audire.confusion.grouping import NUCLEUS_SHAPE, ONSET_MANNER, neutralise_coda
from audire.confusion.profile import POSITIONS, ConfusionProfile
from audire.hangul.inventory import NO_CODA, Position, categories_for
from audire.risk.features import _EPS, word_syllables


def _inventory() -> dict[Position, tuple[str, ...]]:
    """고정된 목표 음소 목록. 열 개수와 순서를 이것이 결정합니다."""
    return {p: tuple(categories_for(p, axis="target")) for p in POSITIONS}


_INVENTORY: Final[dict[Position, tuple[str, ...]]] = _inventory()


def _safe_label(position: Position, jamo: str) -> str:
    """열 이름에 쓸 안정적인 식별자. 자모를 그대로 쓰면 인코딩 경로마다 깨집니다."""
    index = _INVENTORY[position].index(jamo)
    return f"{position.value}{index:02d}"


def word_phoneme_counts(word: str) -> dict[Position, dict[str, int]]:
    """단어 안에서 각 목표 음소가 몇 번 나타나는가."""
    counts: dict[Position, dict[str, int]] = {p: {} for p in POSITIONS}
    for syllable in word_syllables(word):
        for position in POSITIONS:
            jamo = syllable.get(position)
            counts[position][jamo] = counts[position].get(jamo, 0) + 1
    return counts


def exact_target_features(word: str, profile: ConfusionProfile) -> dict[str, float]:
    """D1 — 음소별 ``출현 횟수 × 오류 확률``.

    부류 평균이 지우는 것을 되살립니다. 같은 파열음이라도 이 청취자가 ㅋ 은 거의 못 듣고
    ㄱ 은 잘 듣는다면, 부류 평균은 그 차이를 없애지만 이 열은 남깁니다.
    """
    counts = word_phoneme_counts(word)
    out: dict[str, float] = {}
    for position in POSITIONS:
        matrix = profile.matrix(position)
        present = counts[position]
        for jamo in _INVENTORY[position]:
            n = float(present.get(jamo, 0))
            error = 1.0 - matrix.p_correct(jamo) if n else 0.0
            out[f"et_{_safe_label(position, jamo)}"] = n * error
    return out


def n_exact_target_features() -> int:
    """열 개수. 고정 목록에서 나오므로 데이터에 따라 변하지 않습니다."""
    return sum(len(v) for v in _INVENTORY.values())


def _same_class(position: Position, target: str, other: str) -> bool:
    if other == "?":
        return False
    if position is Position.ONSET:
        return ONSET_MANNER.get(other) is ONSET_MANNER.get(target)
    if position is Position.NUCLEUS:
        return NUCLEUS_SHAPE.get(other) is NUCLEUS_SHAPE.get(target)
    if NO_CODA in (target, other):
        return target == other
    return neutralise_coda(other) == neutralise_coda(target)


def exact_target_offdiag_features(word: str, profile: ConfusionProfile) -> dict[str, float]:
    """D2/D3 — 위치별로 오류 질량이 어디로 가는가.

    목표 음소마다 오프대각 질량을 같은 부류 안과 밖으로 나누고 단어의 구성으로 가중해
    위치별로 합칩니다. Simulator V2 의 복구 기전이 정확히 "어느 위치에서, 부류 안인가
    밖인가" 에 달려 있으므로(docs/RESULTS.md §15) 그 구분을 위치별로 유지합니다.
    """
    counts = word_phoneme_counts(word)
    out: dict[str, float] = {}

    for position in POSITIONS:
        matrix = profile.matrix(position)
        probabilities = matrix.probabilities()
        labels = matrix.perceived_labels
        present = counts[position]

        expected_errors = 0.0
        within = 0.0
        across = 0.0
        top1 = 0.0
        evidence = 0.0
        uncertainty = 0.0
        n_total = 0

        for jamo, n in present.items():
            row = probabilities[matrix.target_labels.index(jamo)]
            p_correct = matrix.p_correct(jamo)
            error_mass = max(1.0 - p_correct, _EPS)

            same = sum(
                float(row[j])
                for j, label in enumerate(labels)
                if label != jamo and _same_class(position, jamo, label)
            )
            off = [float(row[j]) for j, label in enumerate(labels) if label != jamo]

            expected_errors += n * error_mass
            within += n * same
            across += n * (error_mass - same)
            top1 += n * (max(off) / error_mass if off else 0.0)

            observations = matrix.n_observations(jamo)
            evidence += n * float(observations)
            alpha = matrix.smoothing.alpha
            p = min(max(p_correct, _EPS), 1 - _EPS)
            uncertainty += n * float(np.sqrt(p * (1 - p) / (observations + alpha + 1.0)))
            n_total += n

        scale = float(n_total) if n_total else 1.0
        key = position.value
        # 기대 오류 수는 **합** 으로 둡니다. 이 위치에서 오류가 몇 개나 날지가 복구
        # 난이도를 정하므로, 평균을 내면 단어 길이 정보가 사라집니다.
        out[f"eo_{key}_expected_errors"] = expected_errors
        # 나머지는 평균입니다. 비율이므로 길이에 비례해 커지면 해석이 어긋납니다.
        out[f"eo_{key}_within_class"] = within / scale
        out[f"eo_{key}_across_class"] = across / scale
        out[f"eo_{key}_top1_share"] = top1 / scale
        out[f"eo_{key}_evidence"] = evidence / scale
        out[f"eo_{key}_uncertainty"] = uncertainty / scale
    return out


def n_exact_target_offdiag_features() -> int:
    return 6 * len(POSITIONS)
