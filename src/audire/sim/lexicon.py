"""어휘 구조: 지각형이 실재 단어인가, 그 주변에 경쟁 단어가 몇 개인가.

Simulator V2 가 필요로 하는 것
------------------------------
V1 의 복구 확률은 오류 **개수**만 봅니다. 그래서 목표 "각" 을 "닥" 으로 듣든 "삭" 으로
듣든 결과 분포가 같습니다. 실제 청자에게 두 경우는 전혀 다릅니다.

* 지각형이 **실재 단어**이면 청자는 무언가 잘못됐다는 신호를 받지 못합니다. 잘못 들은
  단어를 그대로 받아들이므로 복구가 잘 일어나지 않습니다(어휘 함정).
* 지각형이 **비단어**이면 청자는 오류를 알아채고 주변 단어를 탐색합니다. 다만 후보가
  많을수록 어느 것이 의도된 단어인지 고르기 어렵습니다.

이 모듈은 그 두 가지 — 어휘 실재성과 이웃 밀도 — 를 어휘 목록에서 계산합니다.

왜 순환이 아닌가
----------------
예측 모델은 음소별 ``p_correct`` 의 곱과 그 집계를 계산합니다. "지각형이 실재 단어인가"
라는 개념 자체가 없습니다. 따라서 V2 가 이 신호로 결과를 만들면, 모델은 그것을 자기
수식의 재진술이 아니라 **추정해야 할 대상**으로 마주하게 됩니다.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from audire.hangul.inventory import Position
from audire.hangul.syllable import decompose_syllable, is_hangul_syllable


def jamo_sequence(word: str) -> tuple[str, ...]:
    """단어를 위치별 자모 열로 폅니다 (음절당 초성·중성·종성 3개)."""
    out: list[str] = []
    for ch in word:
        if not is_hangul_syllable(ch):
            continue
        syl = decompose_syllable(ch)
        out.extend(syl.get(p) for p in (Position.ONSET, Position.NUCLEUS, Position.CODA))
    return tuple(out)


@dataclass(frozen=True, slots=True)
class Lexicon:
    """단어 목록과 그 위에서 계산되는 어휘 구조.

    이웃은 **같은 길이·한 자리 치환**으로 정의합니다. 자모 한 자리가 바뀐 형태가 곧 한
    번의 음소 혼동이 만들어내는 결과이고, 이 시뮬레이터에서 오류가 실제로 그렇게
    발생하기 때문입니다. 삽입·삭제는 자모 열 길이를 바꾸는데, 여기서는 위치별 치환만
    일어나므로 정의에서 제외합니다.
    """

    words: frozenset[str]
    #: (자모 열, 가려진 위치) -> 그 패턴에 맞는 단어들. 한 자리 치환 이웃을 O(L) 에
    #: 찾기 위한 색인입니다. 어휘 전체를 매번 훑으면 시행 수 x 어휘 수가 되어 느립니다.
    _index: dict[tuple[tuple[str, ...], int], frozenset[str]] = field(repr=False)
    provenance: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.words)

    def is_word(self, form: str) -> bool:
        """이 형태가 어휘에 실재하는가."""
        return form in self.words

    def neighbours(self, form: str, *, exclude: str | None = None) -> frozenset[str]:
        """``form`` 과 자모 한 자리만 다른 어휘 항목들.

        ``exclude`` 는 보통 목표 단어입니다. "지각형 주변에 목표 말고 다른 후보가 몇 개
        있는가" 가 복구 난이도를 정하기 때문입니다.
        """
        seq = jamo_sequence(form)
        if not seq:
            return frozenset()
        found: set[str] = set()
        for i in range(len(seq)):
            masked = (*seq[:i], "\0", *seq[i + 1 :])
            found |= self._index.get((masked, i), frozenset())
        found.discard(form)
        if exclude is not None:
            found.discard(exclude)
        return frozenset(found)

    def n_competitors(self, form: str, *, exclude: str | None = None) -> int:
        return len(self.neighbours(form, exclude=exclude))

    def describe(self) -> dict[str, Any]:
        return {"n_words": len(self.words), "provenance": dict(self.provenance)}


def build_lexicon(words: tuple[str, ...], provenance: dict[str, Any] | None = None) -> Lexicon:
    """치환-이웃 색인과 함께 어휘를 만듭니다.

    색인은 각 단어의 자모 열에서 한 자리씩 가린 패턴을 열쇠로 씁니다. 같은 패턴에 걸리는
    두 단어는 정확히 그 자리 하나만 다르므로, 조회 한 번이 그 위치의 이웃 전부를 줍니다.
    """
    index: dict[tuple[tuple[str, ...], int], set[str]] = defaultdict(set)
    for word in words:
        seq = jamo_sequence(word)
        for i in range(len(seq)):
            index[((*seq[:i], "\0", *seq[i + 1 :]), i)].add(word)
    return Lexicon(
        words=frozenset(words),
        _index={k: frozenset(v) for k, v in index.items()},
        provenance=dict(provenance or {}),
    )
