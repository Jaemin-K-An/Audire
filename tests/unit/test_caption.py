"""G4 — caption policies, exports and the budget/threshold studies."""

from __future__ import annotations

import json

import numpy as np
import pytest

from audire.caption import (
    STANDARD_BUDGETS,
    BudgetPolicy,
    CaptionDecision,
    FullCaptionPolicy,
    ThresholdPolicy,
    WordRisk,
    build_cues,
    caption_ratio,
    caption_reduction_ratio,
    global_threshold,
    personalized_threshold,
    to_ass,
    to_json,
    to_srt,
    validate_cues,
)
from audire.caption.export import Cue
from audire.eval.caption import (
    budget_frontier,
    compare_strategies,
    compare_thresholds,
    pareto_table,
    recall_by_listener,
    select_budget,
)
from audire.hangul.inventory import Position
from audire.risk.features import PhonemeRisk

MODEL_V = "1.0.0"


def _w(text: str, start: float, risk: float, *, conf: float | None = None) -> WordRisk:
    return WordRisk(
        text=text,
        start_s=start,
        end_s=start + 0.3,
        listener_risk=risk,
        asr_confidence=conf,
        model_version=MODEL_V,
        model_arm="clinical_plus_confusion",
        decision=CaptionDecision.HIDDEN,
        policy="none",
    )


def _words() -> list[WordRisk]:
    return [
        _w("오늘", 0.0, 0.10, conf=0.95),
        _w("날씨가", 0.4, 0.80, conf=0.90),
        _w("정말", 0.8, 0.45, conf=0.30),
        _w("좋습니다", 1.2, 0.95, conf=0.99),
    ]


# =========================================================== WordRisk


def test_asr_confidence_and_listener_risk_are_separate_fields() -> None:
    """ADR-0010: a low-confidence ASR token must never become listener risk."""
    w = _w("정말", 0.0, 0.05, conf=0.02)
    assert w.listener_risk == 0.05
    assert w.asr_confidence == 0.02
    exp = w.explanation()
    assert exp["listener_risk"] == 0.05
    assert exp["asr_confidence"] == 0.02
    assert "does not contribute" in exp["asr_note"]


def test_word_risk_validates_its_fields() -> None:
    with pytest.raises(ValueError, match=r"ends .* before it starts"):
        WordRisk("가", 1.0, 0.5, 0.5, None, MODEL_V, "a", CaptionDecision.HIDDEN, "p")
    with pytest.raises(ValueError, match="listener_risk must be in"):
        WordRisk("가", 0.0, 1.0, 1.5, None, MODEL_V, "a", CaptionDecision.HIDDEN, "p")
    with pytest.raises(ValueError, match="asr_confidence must be in"):
        WordRisk("가", 0.0, 1.0, 0.5, 2.0, MODEL_V, "a", CaptionDecision.HIDDEN, "p")


def test_explanation_reports_weakest_phonemes_with_their_evidence() -> None:
    contributions = (
        PhonemeRisk(Position.ONSET, "ㄱ", 0.95, 30, False, (("ㅋ", 0.03, 1),)),
        PhonemeRisk(Position.CODA, "ㄱ", 0.40, 2, False, (("-", 0.5, 1),)),
        PhonemeRisk(Position.NUCLEUS, "ㅏ", 0.99, 0, False, ()),
    )
    w = WordRisk(
        "각",
        0.0,
        0.4,
        0.62,
        0.9,
        MODEL_V,
        "arm",
        CaptionDecision.SHOWN_HIGH_RISK,
        "p",
        contributions=contributions,
    )
    weakest = w.explanation(top_k=2)["weakest_phonemes"]
    assert [p["phoneme"] for p in weakest] == ["ㄱ", "ㄱ"]
    assert weakest[0]["p_correct"] == 0.40
    assert weakest[0]["n_calibration_observations"] == 2
    assert weakest[0]["likely_confusions"][0]["perceived"] == "-"


def test_explanation_flags_estimates_that_rest_only_on_the_prior() -> None:
    w = WordRisk(
        "각",
        0.0,
        0.4,
        0.5,
        None,
        MODEL_V,
        "arm",
        CaptionDecision.HIDDEN,
        "p",
        contributions=(PhonemeRisk(Position.ONSET, "ㅎ", 0.5, 0, False, ()),),
    )
    assert w.explanation()["weakest_phonemes"][0]["estimate_from_prior_only"] is True


# =========================================================== policies


def test_full_policy_shows_everything() -> None:
    out = FullCaptionPolicy().apply(_words())
    assert all(w.is_shown for w in out)
    assert caption_ratio(out) == 1.0
    assert caption_reduction_ratio(out) == 0.0
    assert all(w.decision is CaptionDecision.SHOWN_FULL_MODE for w in out)


def test_threshold_policy_selects_above_tau_only() -> None:
    out = ThresholdPolicy(tau=0.5).apply(_words())
    assert [w.text for w in out if w.is_shown] == ["날씨가", "좋습니다"]
    assert out[0].policy == "threshold(tau=0.5000)"


def test_budget_policy_shows_exactly_the_requested_fraction() -> None:
    words = [_w(f"w{i}", i * 0.5, i / 20) for i in range(20)]
    for b in STANDARD_BUDGETS:
        out = BudgetPolicy(budget=b).apply(words)
        assert sum(w.is_shown for w in out) == round(b * 20)
        assert caption_ratio(out) == pytest.approx(b)


def test_budget_policy_shows_the_highest_risk_words() -> None:
    out = BudgetPolicy(budget=0.5).apply(_words())
    assert sorted(w.text for w in out if w.is_shown) == sorted(["날씨가", "좋습니다"])


def test_budget_policy_is_deterministic_under_ties() -> None:
    words = [_w(f"w{i}", i * 0.5, 0.5) for i in range(10)]
    a = [w.is_shown for w in BudgetPolicy(budget=0.3).apply(words)]
    b = [w.is_shown for w in BudgetPolicy(budget=0.3).apply(words)]
    assert a == b
    assert sum(a) == 3


def test_low_asr_confidence_is_shown_but_recorded_distinctly() -> None:
    """Surfacing an unreliable ASR token must never be scored as a personalization hit."""
    out = ThresholdPolicy(tau=0.5, asr_confidence_floor=0.4).apply(_words())
    by_text = {w.text: w for w in out}
    assert by_text["정말"].is_shown
    assert by_text["정말"].decision is CaptionDecision.SHOWN_LOW_ASR_CONFIDENCE
    assert by_text["날씨가"].decision is CaptionDecision.SHOWN_HIGH_RISK
    assert by_text["오늘"].decision is CaptionDecision.HIDDEN


def test_asr_floor_is_ignored_when_confidence_is_unavailable() -> None:
    words = [_w("가", 0.0, 0.1, conf=None)]
    assert not ThresholdPolicy(tau=0.5, asr_confidence_floor=0.9).apply(words)[0].is_shown


def test_policies_validate_their_parameters() -> None:
    with pytest.raises(ValueError, match="tau must be in"):
        ThresholdPolicy(tau=1.5)
    with pytest.raises(ValueError, match="budget must be in"):
        BudgetPolicy(budget=-0.1)
    with pytest.raises(ValueError, match="asr_confidence_floor"):
        BudgetPolicy(budget=0.2, asr_confidence_floor=1.5)


def test_empty_word_list_is_handled() -> None:
    assert BudgetPolicy(budget=0.2).apply([]) == []
    assert caption_ratio([]) == 0.0


# =========================================================== thresholds


def test_personalized_threshold_hits_the_requested_ratio() -> None:
    risks = np.linspace(0.0, 1.0, 1000)
    for ratio in (0.1, 0.25, 0.5, 0.9):
        tau = personalized_threshold(risks, ratio)
        assert float(np.mean(risks > tau)) == pytest.approx(ratio, abs=0.01)


def test_threshold_edge_cases() -> None:
    risks = np.linspace(0.0, 1.0, 100)
    assert personalized_threshold(risks, 0.0) == 1.0
    assert personalized_threshold(risks, 1.0) == 0.0
    assert personalized_threshold(np.zeros(0), 0.5) == 1.0
    with pytest.raises(ValueError, match="target_ratio must be in"):
        personalized_threshold(risks, 1.5)


def test_global_threshold_pools_listeners() -> None:
    by_listener = {
        "low": np.full(100, 0.1),
        "high": np.full(100, 0.9),
    }
    tau = global_threshold(by_listener, 0.5)
    # One threshold captions the high-risk listener entirely and the other not at all.
    assert float(np.mean(by_listener["high"] > tau)) == 1.0
    assert float(np.mean(by_listener["low"] > tau)) == 0.0


# =========================================================== export


def test_srt_snapshot() -> None:
    """날씨가 ends at 0.7 s and 좋습니다 starts at 1.2 s: the 0.5 s gap exceeds
    MERGE_GAP_S, so they become two cues. Each 0.3 s word is also extended to the
    0.4 s minimum cue duration."""
    words = ThresholdPolicy(tau=0.5).apply(_words())
    assert to_srt(words) == (
        "1\n00:00:00,400 --> 00:00:00,800\n날씨가\n\n2\n00:00:01,200 --> 00:00:01,600\n좋습니다\n\n"
    )


def test_srt_merges_words_within_the_gap_threshold() -> None:
    words = FullCaptionPolicy().apply([_w("가", 0.0, 0.9), _w("나", 0.4, 0.9)])
    assert to_srt(words) == "1\n00:00:00,000 --> 00:00:00,700\n가 나\n\n"


def test_srt_omits_hidden_words() -> None:
    srt = to_srt(ThresholdPolicy(tau=0.5).apply(_words()))
    assert "오늘" not in srt
    assert "좋습니다" in srt


def test_srt_of_nothing_shown_is_empty() -> None:
    assert to_srt(ThresholdPolicy(tau=0.99).apply(_words())) == ""


def test_srt_timestamp_format_at_hour_boundaries() -> None:
    """A 0.3 s word is extended to the 0.4 s minimum cue duration."""
    srt = to_srt(FullCaptionPolicy().apply([_w("가", 3661.5, 0.9)]))
    assert "01:01:01,500 --> 01:01:01,900" in srt


def test_ass_snapshot_colours_by_risk_band() -> None:
    words = FullCaptionPolicy().apply([_w("낮음", 0.0, 0.10), _w("높음", 0.4, 0.95)])
    ass = to_ass(words)
    assert "[Script Info]" in ass and "Style: Audire," in ass
    assert "{\\c&H00FFFFFF}낮음" in ass  # low  -> white
    assert "{\\c&H004040FF}높음" in ass  # high -> red
    assert ass.count("Dialogue:") == 1


def test_ass_escapes_brace_characters() -> None:
    ass = to_ass(FullCaptionPolicy().apply([_w("a{b}c", 0.0, 0.1)]))
    assert "a\\{b\\}c" in ass


def test_json_export_contains_hidden_words_and_full_provenance() -> None:
    words = BudgetPolicy(budget=0.5).apply(_words())
    payload = json.loads(
        to_json(
            words,
            listener_id="L001",
            policy=BudgetPolicy(budget=0.5).describe(),
            provenance={"cohort": "test", "seed": 1},
        )
    )
    assert payload["schema"] == "audire.caption.v1"
    assert payload["n_words"] == 4 and payload["n_shown"] == 2
    assert payload["caption_ratio"] == 0.5
    assert payload["caption_reduction_ratio"] == 0.5
    assert len(payload["words"]) == 4  # hidden words are exported too
    assert payload["policy"]["budget"] == 0.5
    assert "not a medical device" in payload["disclaimer"]
    assert payload["words"][0]["model_version"] == MODEL_V
    assert payload["words"][0]["explanation"]["asr_confidence"] is not None


def test_cues_are_ordered_non_overlapping_and_positive() -> None:
    words = FullCaptionPolicy().apply([_w("가", 0.0, 0.5), _w("나", 0.31, 0.5), _w("다", 5.0, 0.5)])
    cues = build_cues(words)
    assert len(cues) == 2
    validate_cues(cues)
    assert cues[0].text == "가 나"


def test_short_words_are_extended_without_overlapping_the_next_cue() -> None:
    words = FullCaptionPolicy().apply(
        [
            WordRisk("가", 0.0, 0.05, 0.5, None, MODEL_V, "a", CaptionDecision.HIDDEN, "p"),
            WordRisk("나", 0.2, 0.25, 0.5, None, MODEL_V, "a", CaptionDecision.HIDDEN, "p"),
        ]
    )
    cues = build_cues(words, merge_gap_s=0.0)
    validate_cues(cues)
    assert cues[0].end_s < cues[1].start_s


def test_validate_cues_rejects_overlap_and_zero_duration() -> None:
    a = Cue(1, 0.0, 1.0, ())
    with pytest.raises(ValueError, match=r"starts at .* before cue"):
        validate_cues([a, Cue(2, 0.5, 1.5, ())])
    with pytest.raises(ValueError, match="non-positive duration"):
        validate_cues([Cue(1, 1.0, 1.0, ())])


def test_export_rejects_negative_timestamps() -> None:
    from audire.caption.export import _ass_timestamp, _srt_timestamp

    with pytest.raises(ValueError, match="cannot be negative"):
        _srt_timestamp(-1.0)
    with pytest.raises(ValueError, match="cannot be negative"):
        _ass_timestamp(-1.0)


# =========================================================== budget study


def _study_data(n_listeners: int = 8, n_words: int = 50, seed: int = 0):
    rng = np.random.default_rng(seed)
    groups = np.array(
        [f"L{i:02d}" for i in range(n_listeners) for _ in range(n_words)], dtype=np.str_
    )
    signal = rng.random(groups.size)
    y = (rng.random(groups.size) < signal * 0.8).astype(np.int64)
    return y, groups, signal


def test_a_perfect_ranking_reaches_the_recall_ceiling() -> None:
    y, groups, _ = _study_data()
    oracle = y.astype(np.float64) + np.random.default_rng(1).random(y.size) * 1e-6
    pts = {p.budget: p for p in budget_frontier(y, groups, oracle, strategy="oracle")}
    # At a 20 % budget with a ~40 % base rate, an oracle captions half the misheard words.
    assert pts[0.20].misheard_recall > pts[0.10].misheard_recall
    assert pts[0.20].precision == pytest.approx(1.0, abs=0.01)


def test_random_ranking_recall_tracks_the_budget() -> None:
    y, groups, _ = _study_data()
    rand = np.random.default_rng(7).random(y.size)
    for pt in budget_frontier(y, groups, rand, strategy="random"):
        assert pt.misheard_recall == pytest.approx(pt.budget, abs=0.06)


def test_informative_ranking_beats_random_at_every_budget() -> None:
    y, groups, signal = _study_data()
    rand = np.random.default_rng(7).random(y.size)
    good = {p.budget: p for p in budget_frontier(y, groups, signal, strategy="signal")}
    base = {p.budget: p for p in budget_frontier(y, groups, rand, strategy="random")}
    for b in STANDARD_BUDGETS:
        assert good[b].misheard_recall > base[b].misheard_recall, b


def test_achieved_ratio_matches_the_budget_and_crr_is_its_complement() -> None:
    y, groups, signal = _study_data()
    for pt in budget_frontier(y, groups, signal, strategy="s"):
        assert pt.achieved_ratio == pytest.approx(pt.budget, abs=0.01)
        assert pt.caption_reduction_ratio == pytest.approx(1 - pt.achieved_ratio)


def test_per_listener_budget_gives_every_listener_the_same_share() -> None:
    _y, groups, signal = _study_data()
    sel = select_budget(signal, groups, 0.2, per_listener=True)
    shares = [float(sel[groups == g].mean()) for g in np.unique(groups)]
    assert max(shares) - min(shares) < 0.02


def test_pooled_budget_can_starve_individual_listeners() -> None:
    """Aggregate recall hides who is served badly; the pooled contrast must show it."""
    groups = np.array(["low"] * 100 + ["high"] * 100, dtype=np.str_)
    scores = np.concatenate([np.full(100, 0.1), np.full(100, 0.9)])
    y = np.concatenate([np.ones(100), np.ones(100)]).astype(np.int64)
    pooled = select_budget(scores, groups, 0.5, per_listener=False)
    per = select_budget(scores, groups, 0.5, per_listener=True)
    assert min(recall_by_listener(y, groups, pooled).values()) == 0.0
    assert min(recall_by_listener(y, groups, per).values()) == pytest.approx(0.5)


def test_per_listener_recall_distribution_is_reported() -> None:
    y, groups, signal = _study_data()
    pt = budget_frontier(y, groups, signal, strategy="s", budgets=(0.2,))[0]
    assert pt.recall_min <= pt.recall_median <= pt.recall_max
    assert np.isfinite([pt.recall_min, pt.recall_median, pt.recall_max]).all()


def test_compare_strategies_requires_aligned_rows() -> None:
    import dataclasses

    from audire.eval.ablation import ArmResult
    from audire.eval.metrics import compute_metrics

    y, groups, signal = _study_data(n_listeners=4, n_words=20)
    words = ["가" * (1 + i % 3) for i in range(y.size)]

    def _arm(name: str, p: np.ndarray) -> ArmResult:
        m = compute_metrics(y, p)
        return ArmResult(
            arm=name,
            model="m",
            seed=0,
            calibration="none",
            n_listeners=4,
            n_trials=int(y.size),
            n_features=1,
            metrics=m,
            prevalence_floor=m,
            intervals={},
            reliability={},
            model_description={},
            fold_sizes=[],
            y_true=y,
            y_prob=p,
            groups=groups,
        )

    arms = {"a": _arm("a", signal)}
    pts = compare_strategies(arms, words, seed=0)
    assert {p.strategy for p in pts} == {"random", "word_length", "model:a"}

    with pytest.raises(ValueError, match=r"but .* evaluation rows"):
        compare_strategies(arms, words[:-1], seed=0)
    misaligned = _arm("b", signal)
    misaligned = dataclasses.replace(misaligned, groups=groups[::-1])
    with pytest.raises(ValueError, match="different rows"):
        compare_strategies({**arms, "b": misaligned}, words, seed=0)
    with pytest.raises(ValueError, match="no model arms"):
        compare_strategies({}, words)

    frontier = pareto_table(pts)
    assert {"strategy", "budget", "misheard_recall", "recall_min"} <= set(frontier[0])

    cmp = compare_thresholds(arms["a"], 0.2)
    assert cmp.target_ratio == 0.2
    assert len(cmp.personalized_taus) == 4
    d = cmp.to_dict()
    assert "equity" in d
    assert d["personalized_tau_summary"]["n_listeners"] == 4
