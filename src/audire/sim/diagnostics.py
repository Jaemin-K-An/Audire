"""생성 과정에 실제로 존재하는 단어 수준 신호를 측정합니다.

왜 이 진단이 필요한가
---------------------
더 정교한 추정기를 붙이기 전에, **생성 과정이 애초에 무엇을 알 수 있게 만들어 두었는지**를
알아야 합니다. 시뮬레이터가 단어 결과를 소수의 요약량만으로 결정한다면, 아무리 풍부한
개인 혼동 표현을 넣어도 그 요약량을 더 잘 추정하는 것 이상은 할 수 없습니다. 그 상한을
모르고 모델을 늘리면 "왜 이득이 없는가" 를 모델 탓으로 오해하게 됩니다.

Simulator V1 의 구조
--------------------
:func:`audire.sim.trials.simulate_word_trial` 에서 결과는

    misheard = not (exact or repaired)

이고 ``exact`` 는 ``n_segment_errors == 0`` 과 같으며, ``repaired`` 의 확률은
:func:`audire.sim.trials._repair_probability` 가 정하는데 그 함수는 **오류 개수와 음절
수만** 받습니다. ``perceived_word`` 는 계산되어 기록되지만 복구 판정에 쓰이지 않습니다.

따라서 목표 "각" 을 "닥" 으로 듣는 것과 "삭" 으로 듣는 것은 결과 분포가 정확히 같습니다.
어느 위치가 틀렸는지, 지각형이 실재하는 한국어 단어인지, 음운적으로 얼마나 먼지가 모두
무관합니다. 이것이 V2 의 동기입니다.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score

from audire.sim.cohort import Cohort


def outcome_signal_ceiling(cohort: Cohort) -> dict[str, Any]:
    """생성 과정의 요약량만 아는 오라클이 도달하는 상한.

    ``(n_segment_errors, n_syllables)`` 칸마다 관측된 오청률을 그대로 예측값으로 쓰는
    오라클을 만듭니다. V1 에서는 결과가 그 두 값으로 완전히 결정되므로, 이 오라클의
    PR-AUC 가 **어떤 추정기도 넘을 수 없는 상한**입니다(추정 잡음을 제외하면).

    상한과 실제 모델 성능의 간격이 좁다면, 남은 개선 여지는 모델이 아니라 생성 과정에
    있습니다.
    """
    trials = [t for r in cohort.records for t in r.word_trials]
    if not trials:
        raise ValueError("빈 코호트에서는 상한을 측정할 수 없습니다")

    cells: dict[tuple[int, int], list[int]] = defaultdict(list)
    for t in trials:
        cells[(t.n_segment_errors, t.n_syllables)].append(int(t.misheard))

    y = np.asarray([int(t.misheard) for t in trials], dtype=np.int64)
    oracle = np.asarray(
        [float(np.mean(cells[(t.n_segment_errors, t.n_syllables)])) for t in trials],
        dtype=np.float64,
    )
    length = np.asarray([float(t.n_syllables) for t in trials], dtype=np.float64)

    # 오류가 없으면 결과가 결정적으로 "정상 청취" 인가. V1 에서는 그래야 합니다.
    zero_error = [int(t.misheard) for t in trials if t.n_segment_errors == 0]

    return {
        "n_trials": len(trials),
        "prevalence": float(y.mean()),
        "oracle_pr_auc": float(average_precision_score(y, oracle)),
        "word_length_pr_auc": float(average_precision_score(y, length)),
        "n_cells": len(cells),
        "p_misheard_given_zero_errors": float(np.mean(zero_error)) if zero_error else float("nan"),
        "cell_rates": {
            f"{errors}e_{syllables}s": float(np.mean(v))
            for (errors, syllables), v in sorted(cells.items())
        },
    }


def perceived_form_influence(cohort: Cohort, *, min_count: int = 5) -> dict[str, Any]:
    """같은 ``(오류 수, 음절 수)`` 칸 안에서 **지각형**이 결과를 바꾸는가.

    V1 에서는 바꾸지 않습니다 — 복구 확률이 지각형을 보지 않기 때문입니다. V2 는 이 값이
    0 이 아니게 만드는 것이 목표이고, 이 함수가 그 차이를 측정합니다.

    ``min_count`` 미만으로 관측된 지각형은 제외합니다. 어휘가 크면 지각형이 거의 유일해져
    비율 추정이 불가능하기 때문이며, 몇 개가 남았는지도 함께 보고합니다.
    """
    trials = [t for r in cohort.records for t in r.word_trials if t.perceived_word]
    by_cell: dict[tuple[int, int], dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for t in trials:
        by_cell[(t.n_segment_errors, t.n_syllables)][t.perceived_word].append(int(t.misheard))

    spreads: list[float] = []
    n_comparable = 0
    for forms in by_cell.values():
        rates = [float(np.mean(v)) for v in forms.values() if len(v) >= min_count]
        if len(rates) >= 2:
            n_comparable += 1
            spreads.append(float(np.std(rates)))

    return {
        "n_cells_with_comparable_forms": n_comparable,
        "mean_rate_sd_across_forms": float(np.mean(spreads)) if spreads else float("nan"),
        "max_rate_sd_across_forms": float(np.max(spreads)) if spreads else float("nan"),
        "min_count": min_count,
    }
