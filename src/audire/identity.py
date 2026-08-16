"""청취자 식별자의 단일 검증 규칙.

이 모듈이 존재하는 이유는 규칙이 계층마다 달랐기 때문입니다. 저장소는 안전한 알파벳을
강제했지만 스키마는 길이만 봤고 혼동 프로파일은 아무 검증도 하지 않았습니다. 그 결과
스키마가 받아들인 식별자를 저장소가 거부할 수 있었고, 사람 이름을 그대로 식별자로 쓰는
것도 일부 경로에서는 가능했습니다.

식별자가 만족해야 하는 성질:

* **불투명해야 한다.** 이름·생년월일 등 직접 식별자가 들어가면 안 됩니다. 알파벳을
  ASCII 영숫자와 ``._-`` 로 제한하는 것이 이를 강제하지는 못하지만, 한글 이름을 그대로
  넣는 가장 흔한 실수는 막습니다.
* **파일 이름으로 안전해야 한다.** 저장소가 식별자를 디렉터리 이름으로 쓰므로
  ``../`` 나 ``/`` 가 들어가면 경로 탈출이 됩니다.
* **로그로 안전해야 한다.** 개행이나 널 바이트가 들어가면 구조화 로그를 위조할 수
  있습니다.
"""

from __future__ import annotations

import re
from typing import Annotated

from pydantic import AfterValidator

#: 실제 청취자 식별자. 영숫자로 시작하고 1~64자.
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")

#: 집단/집계 프로파일용 예약 형태. 청취자가 **아님**을 이름으로 드러냅니다.
_AGGREGATE_ID = re.compile(r"^__[a-z][a-z0-9_]*__\Z")

MAX_LISTENER_ID_LENGTH = 64


def validate_listener_id(listener_id: str, *, allow_aggregate: bool = False) -> str:
    """식별자를 검증하고 그대로 돌려줍니다.

    Parameters
    ----------
    listener_id:
        검사할 식별자.
    allow_aggregate:
        ``__pooled__`` 같은 예약된 집계 식별자를 허용할지 여부. 기본값은 ``False`` 이며,
        집계 프로파일을 실제 청취자로 오인하는 것을 막습니다.

    Raises
    ------
    ValueError
        형식에 맞지 않거나, 예약 형태인데 ``allow_aggregate`` 가 꺼져 있는 경우.
    """
    if not isinstance(listener_id, str):  # pragma: no cover - 타입 검사기가 먼저 잡음
        raise ValueError(f"청취자 식별자는 문자열이어야 합니다: {type(listener_id).__name__}")

    if _AGGREGATE_ID.match(listener_id):
        if allow_aggregate:
            return listener_id
        raise ValueError(
            f"{listener_id!r} 는 예약된(reserved) 집단(aggregate) 식별자입니다. "
            f"개별 청취자 식별자로 사용할 수 없습니다."
        )

    if not _SAFE_ID.match(listener_id):
        raise ValueError(
            f"잘못된 청취자 식별자 {listener_id!r}: 영숫자로 시작하는 "
            f"[A-Za-z0-9._-] 1~{MAX_LISTENER_ID_LENGTH}자여야 합니다. "
            f"이름이나 그 밖의 직접 식별자를 사용하지 마십시오."
        )
    return listener_id


def _validate(value: str) -> str:
    return validate_listener_id(value)


def _validate_allowing_aggregate(value: str) -> str:
    return validate_listener_id(value, allow_aggregate=True)


#: pydantic 모델에서 쓰는 검증된 식별자 타입.
ListenerId = Annotated[str, AfterValidator(_validate)]

#: 집계 프로파일도 담을 수 있는 변형.
ListenerOrAggregateId = Annotated[str, AfterValidator(_validate_allowing_aggregate)]


def is_aggregate_id(listener_id: str) -> bool:
    """``__pooled__`` 처럼 개별 청취자가 아닌 집계 식별자인지."""
    return bool(_AGGREGATE_ID.match(listener_id))
