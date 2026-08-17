"""Phase C — Simulator V2 의 불변식.

미션이 요구하는 항목을 그대로 고정합니다: V1/V2 가 따로 버전 관리되는가, 같은 시드가
결정론적인가, V2 에서 지각형·오류 위치가 복구에 실제로 영향을 주는가, 생성기가 채점기를
불러 라벨을 만들지 않는가, 모든 파라미터가 출처 분류를 갖는가.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pytest

from audire.hangul.inventory import NO_CODA, Position
from audire.sim import SimulationConfig, build_cohort
from audire.sim.config import LexicalRepairV2
from audire.sim.lexicon import build_lexicon, jamo_sequence
from audire.sim.trials import SegmentError, _same_phonological_class, repair_probability_v2

BASE = {
    "seeds": [101],
    "n_listeners": 20,
    "n_calibration_trials": 60,
    "n_word_trials": 80,
    "snr_conditions_db": [20.0],
    "speakers": ["male", "female"],
}


def _cfg(version: str, **overrides) -> SimulationConfig:
    return SimulationConfig(name=f"sim-{version}", simulator_version=version, **BASE, **overrides)


# ------------------------------------------------------------------------ 버전 관리


def test_v1_is_the_default_so_recorded_runs_keep_their_meaning():
    """기본값이 바뀌면 기존 설정과 기록된 실행의 의미론이 조용히 달라집니다."""
    assert SimulationConfig(name="x").simulator_version == "v1"


def test_both_versions_remain_runnable():
    for version in ("v1", "v2"):
        cohort = build_cohort(_cfg(version), 101)
        assert cohort.summary()["simulator_version"] == version


def test_simulator_version_appears_in_every_artifact():
    """버전 없이 기록된 수치는 해석할 수 없습니다."""
    cohort = build_cohort(_cfg("v2"), 101)
    assert cohort.summary()["simulator_version"] == "v2"
    assert cohort.provenance["simulator_version"] == "v2"


def test_an_unknown_simulator_version_is_rejected():
    with pytest.raises(ValueError):
        SimulationConfig(name="x", simulator_version="v3")


# --------------------------------------------------------------------------- 결정성


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_same_seed_is_deterministic(version):
    a = build_cohort(_cfg(version), 101)
    b = build_cohort(_cfg(version), 101)
    ta = [(t.word, t.perceived_word, t.misheard) for r in a.records for t in r.word_trials]
    tb = [(t.word, t.perceived_word, t.misheard) for r in b.records for t in r.word_trials]
    assert ta == tb


def test_versions_share_the_perception_samples_and_differ_only_in_repair():
    """V2 가 지각 표본까지 바꿔 버리면 두 조건의 비교가 생성 과정 비교가 아니게 됩니다."""
    v1 = build_cohort(_cfg("v1"), 101)
    v2 = build_cohort(_cfg("v2"), 101)
    pairs = list(
        zip(
            [t for r in v1.records for t in r.word_trials],
            [t for r in v2.records for t in r.word_trials],
            strict=True,
        )
    )
    for a, b in pairs:
        assert a.word == b.word
        assert a.perceived_word == b.perceived_word
        assert a.n_segment_errors == b.n_segment_errors
    # 복구 판정은 달라져야 합니다.
    assert any(a.misheard != b.misheard for a, b in pairs)


# ----------------------------------------------------------- V2 가 실제로 무엇을 바꾸는가


def test_error_position_changes_the_outcome_in_v2_but_not_in_v1():
    """V1 의 핵심 한계이자 V2 의 존재 이유.

    V1 의 복구 확률은 오류 개수와 음절 수만 받으므로 어느 위치가 틀렸는지가 무관합니다.
    V2 는 한국어 종성 중화를 반영해 종성 오류를 잘 복구하고 모음 오류를 잘 복구하지
    못합니다.
    """
    big = BASE | {"n_listeners": 40, "n_word_trials": 200}
    rates: dict[str, dict[str, float]] = {}
    for version in ("v1", "v2"):
        cohort = build_cohort(
            SimulationConfig(name=f"s-{version}", simulator_version=version, **big), 101
        )
        by_pos: dict[str, list[int]] = defaultdict(list)
        for record in cohort.records:
            for trial in record.word_trials:
                if trial.n_segment_errors != 1 or not trial.perceived_word:
                    continue
                errors = _errors_of(trial)
                if len(errors) != 1:
                    continue
                by_pos[errors[0].position.value].append(int(trial.misheard))
        rates[version] = {k: float(np.mean(v)) for k, v in by_pos.items() if len(v) >= 50}

    assert {"onset", "nucleus", "coda"} <= set(rates["v1"])
    assert {"onset", "nucleus", "coda"} <= set(rates["v2"])
    # V1: 위치가 결과를 거의 바꾸지 않습니다.
    assert max(rates["v1"].values()) - min(rates["v1"].values()) < 0.10
    # V2: 종성이 가장 잘 복구되고 중성이 가장 안 됩니다.
    assert rates["v2"]["coda"] < rates["v2"]["onset"] < rates["v2"]["nucleus"]


def _errors_of(trial) -> list[SegmentError]:
    from audire.hangul.syllable import decompose_syllable, is_hangul_syllable

    target = [c for c in trial.word if is_hangul_syllable(c)]
    perceived = [c for c in trial.perceived_word if is_hangul_syllable(c)]
    if len(target) != len(perceived):
        return []
    out: list[SegmentError] = []
    for a, b in zip(target, perceived, strict=True):
        sa, sb = decompose_syllable(a), decompose_syllable(b)
        for position in (Position.ONSET, Position.NUCLEUS, Position.CODA):
            if sa.get(position) != sb.get(position):
                out.append(
                    SegmentError(
                        position=position, target=sa.get(position), perceived=sb.get(position)
                    )
                )
    return out


def test_same_class_substitution_repairs_more_easily():
    cfg = _cfg("v2")
    lexicon = build_lexicon(("가나",))
    same = SegmentError(position=Position.ONSET, target="ㄱ", perceived="ㅋ")  # 둘 다 파열음
    across = SegmentError(position=Position.ONSET, target="ㄱ", perceived="ㅁ")  # 파열음 -> 비음
    assert _same_phonological_class(same)
    assert not _same_phonological_class(across)
    p_same = repair_probability_v2(cfg, "가", "카", [same], 1, lexicon)
    p_across = repair_probability_v2(cfg, "가", "마", [across], 1, lexicon)
    assert p_same > p_across


def test_no_errors_means_no_repair_decision():
    cfg = _cfg("v2")
    assert repair_probability_v2(cfg, "가", "가", [], 1, build_lexicon(("가",))) == 0.0


def test_repair_probability_stays_within_declared_bounds():
    cfg = _cfg("v2")
    lexicon = build_lexicon(("가나다",))
    coda = SegmentError(position=Position.CODA, target=NO_CODA, perceived="ㄱ")
    many = [coda] * 8
    for errors in ([coda], many):
        p = repair_probability_v2(cfg, "가나다", "각나다", errors, 3, lexicon)
        assert cfg.lexical_repair_v2.repair_floor <= p <= cfg.lexical_repair_v2.repair_ceiling


def test_v2_requires_a_lexicon():
    """어휘 없이 v2 를 돌리면 조용히 v1 로 떨어지지 않고 실패해야 합니다."""
    from audire.sim.listener import generate_cohort
    from audire.sim.trials import simulate_word_trial

    cfg = _cfg("v2")
    listener = generate_cohort(cfg, 101)[0]
    with pytest.raises(ValueError, match="어휘 구조"):
        simulate_word_trial(
            listener,
            "가나",
            np.random.default_rng(0),
            cfg,
            snr_db=20.0,
            speaker="male",
            lexicon=None,
        )


# ------------------------------------------------------------------------ 순환 방지


def test_the_generator_never_calls_the_scoring_model():
    """생성기가 채점기를 불러 라벨을 만들면 실험이 순환이 됩니다."""
    import inspect

    from audire.sim import cohort, trials

    for module in (trials, cohort):
        source = inspect.getsource(module)
        for forbidden in ("audire.risk", "WordScorer", "phoneme_independence_risk", "make_model"):
            assert forbidden not in source, f"{module.__name__} 이 {forbidden} 를 참조합니다"


# ------------------------------------------------------------------------ 출처 분류


def test_every_v2_parameter_block_declares_its_evidence():
    """수치를 사실처럼 제시하지 않기 위한 장치입니다."""
    block = LexicalRepairV2()
    assert block.evidence in ("literature", "clinical_convention", "assumption")
    assert block.rationale, "근거 설명이 비어 있습니다"
    # 문헌이라고 주장하면 출처가 있어야 합니다 (기반 클래스가 강제).
    if block.evidence == "literature":
        assert block.evidence_source


def test_v2_evidence_appears_in_the_cohort_evidence_report():
    report = build_cohort(_cfg("v2"), 101).summary()["evidence"]
    assert report, "출처 보고가 비어 있습니다"


def test_lexical_trap_rate_is_recorded_even_when_it_never_fires():
    """발동하지 않는 항을 효과가 있는 것처럼 읽지 않도록 실제 발동률을 남깁니다."""
    summary = build_cohort(_cfg("v2"), 101).summary()
    assert "lexical_trap_rate" in summary
    assert 0.0 <= summary["lexical_trap_rate"] <= 1.0


# --------------------------------------------------------------------------- 어휘 구조


def test_lexicon_finds_single_substitution_neighbours():
    lexicon = build_lexicon(("각", "닥", "박", "간", "가족"))
    assert lexicon.is_word("각")
    assert not lexicon.is_word("좍")
    neighbours = lexicon.neighbours("각")
    assert {"닥", "박", "간"} <= neighbours
    assert "가족" not in neighbours  # 길이가 다름


def test_lexicon_excludes_the_target_from_competitors():
    lexicon = build_lexicon(("각", "닥", "박"))
    assert lexicon.n_competitors("각", exclude="닥") == 1


def test_jamo_sequence_is_three_per_syllable():
    assert len(jamo_sequence("가족")) == 6
    assert jamo_sequence("ABC") == ()
