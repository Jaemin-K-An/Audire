"""라이브 임계값 보정에 쓰는 참조 어휘.

왜 필요한가
-----------
ADR-0021 은 라이브 소비 측에 **청취자별 임계값**을 강제합니다. 전역 임계값을 쓰면 E30 에서
관측된 대로 normal 청취자가 자막률 0.0004 를 받고 중앙값 청취자의 재현율이 0 이 됩니다.

그런데 라이브 자막은 큐 단위로 도착하므로, 그 청취자의 점수 분포를 미리 알 수 없습니다.
분위수를 실시간으로 추정하면 상태가 생기고 초반에 흔들리며 세션마다 달라집니다.

해결은 **큐가 도착하기 전에** 그 청취자의 임계값을 정하는 것입니다. 이 청취자의 프로파일로
고정된 참조 어휘를 채점하고 그 분포의 분위수를 임계값으로 씁니다. 결정적이고, 상태가 없고,
같은 프로파일이면 언제나 같은 값이 나옵니다.

무엇이 가정인가
---------------
임계값의 대표성은 **참조 어휘가 사용자가 실제로 볼 자막과 얼마나 닮았는가** 에 달려
있습니다. 그것은 알 수 없으므로 여기서는 명시적 가정으로 둡니다.

* 음절은 프로젝트의 결정적 음소 균형 목록에서 가져옵니다(`build_balanced_catalog`).
* 단어 길이 분포는 **선언된 가정**입니다. 실제 한국어 자막의 어절 길이 분포를 측정한
  값이 아니며, 그 사실을 버전 이름과 provenance 에 적어 둡니다.

참조 어휘가 바뀌면 임계값의 의미가 바뀌므로 버전을 올리고, 아티팩트/응답에 그 버전이
실려 나갑니다.
"""

from __future__ import annotations

from typing import Any

from audire.data.stimuli import build_balanced_catalog

#: 참조 어휘 버전. 이것이 바뀌면 같은 프로파일이라도 임계값이 달라지므로, 응답과 캐시 키에
#: 함께 실립니다.
REFERENCE_VOCABULARY_VERSION = "live-ref-v1"

#: 참조 어휘의 어절 길이 분포. **선언된 가정**이며 측정값이 아닙니다. 한국어 자막 어절이
#: 대체로 2–3 음절에 몰린다는 통상적 관찰을 반영했을 뿐, 이 프로젝트가 측정한 값은
#: 아닙니다. 임계값의 대표성이 여기에 달려 있으므로 숫자를 사실로 제시하지 않습니다.
SYLLABLE_LENGTH_MIX: tuple[tuple[int, int], ...] = ((1, 20), (2, 45), (3, 25), (4, 10))

#: 참조 어휘 크기. 분위수가 안정될 만큼 크되 매 페어링마다 채점해도 즉시 끝날 만큼 작게.
REFERENCE_SIZE = 400


def build_reference_vocabulary() -> tuple[str, ...]:
    """결정적 참조 어휘.

    음소 균형 음절 목록을 순서대로 이어 붙여 선언된 길이 분포대로 어절을 만듭니다. 난수를
    쓰지 않으므로 같은 버전은 언제나 같은 목록을 냅니다.
    """
    catalog = build_balanced_catalog()
    syllables = [stimulus.syllable for stimulus in catalog.stimuli]
    if not syllables:
        raise ValueError("음소 균형 목록이 비어 있어 참조 어휘를 만들 수 없습니다")

    words: list[str] = []
    cursor = 0
    total_weight = sum(weight for _length, weight in SYLLABLE_LENGTH_MIX)
    for length, weight in SYLLABLE_LENGTH_MIX:
        count = round(REFERENCE_SIZE * weight / total_weight)
        for _ in range(count):
            word = "".join(syllables[(cursor + i) % len(syllables)] for i in range(length))
            words.append(word)
            cursor += length
    return tuple(words)


def reference_provenance() -> dict[str, Any]:
    """참조 어휘의 출처와 가정. 응답과 사이드카에 실립니다."""
    return {
        "version": REFERENCE_VOCABULARY_VERSION,
        "size": len(build_reference_vocabulary()),
        "syllable_source": "audire.data.stimuli.build_balanced_catalog (결정적 음소 균형)",
        "syllable_length_mix": {str(length): weight for length, weight in SYLLABLE_LENGTH_MIX},
        "evidence": "assumption",
        "rationale": (
            "어절 길이 분포는 선언된 가정이며 한국어 자막을 측정한 값이 아닙니다. 임계값의 "
            "대표성이 이 목록에 달려 있으므로, 실제 자막 분포가 크게 다르면 달성 자막률이 "
            "목표에서 벗어날 수 있습니다. 그 경우 버전을 올려 재보정해야 합니다."
        ),
    }
