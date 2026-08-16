"""E2 — confusion-matrix construction, invariants, omission/addition and missing data."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from audire.confusion import (
    CalibrationTrial,
    ConfusionMatrix,
    ConfusionProfile,
    PositionErrorType,
    ResponseQuality,
    SmoothingSpec,
    TrialErrorType,
    neutralise_coda,
    parse_response,
    pool_profiles,
)
from audire.confusion.grouping import (
    NEUTRALISED_CODA_CATEGORIES,
    Phonation,
    coda_features,
    onset_features,
)
from audire.hangul import NO_CODA, NO_RESPONSE, Position, categories_for
from audire.hangul.syllable import HANGUL_SYLLABLE_END, HANGUL_SYLLABLE_START

ALL_SYLLABLES = [chr(c) for c in range(HANGUL_SYLLABLE_START, HANGUL_SYLLABLE_END + 1)]


# =========================================================== response parsing


def test_correct_response_is_correct_at_every_position() -> None:
    t = parse_response("각", "각")
    assert t.is_correct
    assert t.trial_error is TrialErrorType.CORRECT
    assert t.quality is ResponseQuality.OK
    assert all(o.is_correct for o in t.observations)
    assert len(t.observations) == 3


@pytest.mark.parametrize(
    ("target", "response", "expected"),
    [
        ("각", "닥", TrialErrorType.ONSET_ERROR),
        ("각", "극", TrialErrorType.NUCLEUS_ERROR),
        ("각", "간", TrialErrorType.CODA_ERROR),
        ("각", "가", TrialErrorType.CODA_ERROR),  # omission
        ("가", "각", TrialErrorType.CODA_ERROR),  # addition
        ("각", "던", TrialErrorType.COMPOUND),
        ("각", "", TrialErrorType.NO_RESPONSE),
        ("각", "?", TrialErrorType.NO_RESPONSE),
        ("각", "abc", TrialErrorType.NO_RESPONSE),
    ],
)
def test_trial_error_classification(target: str, response: str, expected: TrialErrorType) -> None:
    assert parse_response(target, response).trial_error is expected


def test_coda_omission_is_recorded_not_dropped() -> None:
    """각 -> 가 must land in the (ㄱ, NO_CODA) cell, not vanish."""
    t = parse_response("각", "가")
    coda = next(o for o in t.observations if o.position is Position.CODA)
    assert (coda.target, coda.perceived) == ("ㄱ", NO_CODA)
    assert coda.error_type is PositionErrorType.OMISSION


def test_coda_addition_is_recorded_not_dropped() -> None:
    """가 -> 각 must land in the (NO_CODA, ㄱ) cell."""
    t = parse_response("가", "각")
    coda = next(o for o in t.observations if o.position is Position.CODA)
    assert (coda.target, coda.perceived) == (NO_CODA, "ㄱ")
    assert coda.error_type is PositionErrorType.ADDITION


def test_onset_omission_and_addition_use_the_null_onset() -> None:
    """ㅇ is the null onset: ㄱ->ㅇ is an omission, ㅇ->ㄱ is an addition."""
    omit = next(o for o in parse_response("가", "아").observations if o.position is Position.ONSET)
    assert omit.error_type is PositionErrorType.OMISSION
    add = next(o for o in parse_response("아", "가").observations if o.position is Position.ONSET)
    assert add.error_type is PositionErrorType.ADDITION


def test_unusable_response_still_produces_three_observations() -> None:
    """A blank answer is evidence, not an absence of evidence."""
    t = parse_response("각", "   ")
    assert t.response_syllable is None
    assert t.quality is ResponseQuality.BLANK
    assert len(t.observations) == 3
    assert all(o.perceived == NO_RESPONSE for o in t.observations)
    assert all(o.error_type is PositionErrorType.NO_RESPONSE for o in t.observations)


def test_multi_syllable_response_is_flagged_and_first_syllable_scored() -> None:
    t = parse_response("각", "각각")
    assert t.quality is ResponseQuality.MULTI_SYLLABLE
    assert t.response_syllable is not None
    assert t.is_correct
    assert t.raw_response == "각각"


def test_raw_response_is_preserved_verbatim() -> None:
    assert parse_response("각", "  닥  ").raw_response == "  닥  "


def test_parse_rejects_non_syllable_target() -> None:
    with pytest.raises(ValueError, match="must be one Hangul syllable"):
        parse_response("가나", "가")


@given(st.sampled_from(ALL_SYLLABLES), st.sampled_from(ALL_SYLLABLES))
def test_property_parse_always_yields_three_observations(target: str, response: str) -> None:
    t = parse_response(target, response)
    assert len(t.observations) == 3
    assert [o.position for o in t.observations] == [
        Position.ONSET,
        Position.NUCLEUS,
        Position.CODA,
    ]
    assert t.is_correct == (target == response)


# =========================================================== matrix invariants


@pytest.mark.parametrize("position", list(Position))
def test_empty_matrix_shape_and_labels(position: Position) -> None:
    m = ConfusionMatrix.empty(position)
    assert m.counts.shape == (
        len(categories_for(position, axis="target")),
        len(categories_for(position, axis="perceived")),
    )
    assert m.perceived_labels[-1] == NO_RESPONSE
    assert NO_RESPONSE not in m.target_labels
    assert m.total_observations == 0


@pytest.mark.parametrize("position", list(Position))
def test_rows_sum_to_one_after_smoothing_even_with_no_data(position: Position) -> None:
    m = ConfusionMatrix.empty(position)
    p = m.probabilities()
    assert np.allclose(p.sum(axis=1), 1.0)
    assert np.all(p >= 0.0)


@pytest.mark.parametrize("position", list(Position))
def test_rows_sum_to_one_after_observations(position: Position) -> None:
    m = ConfusionMatrix.empty(position)
    tgt = m.target_labels
    rng = np.random.default_rng(7)
    for _ in range(200):
        t = tgt[int(rng.integers(len(tgt)))]
        p = m.perceived_labels[int(rng.integers(len(m.perceived_labels)))]
        m.observe(t, p)
    assert np.allclose(m.probabilities().sum(axis=1), 1.0)


def test_counts_are_never_replaced_by_probabilities() -> None:
    m = ConfusionMatrix.empty(Position.ONSET)
    m.observe("ㄱ", "ㄱ")
    assert m.p_correct("ㄱ") > 0.5
    assert m.n_observations("ㄱ") == 1
    assert m.n_observations("ㄷ") == 0
    # Same probability shape, very different evidence.
    m2 = ConfusionMatrix.empty(Position.ONSET)
    for _ in range(40):
        m2.observe("ㄱ", "ㄱ")
    assert m2.n_observations("ㄱ") == 40
    assert m2.p_correct("ㄱ") > m.p_correct("ㄱ")


def test_unobserved_rows_equal_the_prior_exactly() -> None:
    m = ConfusionMatrix.empty(Position.ONSET)
    m.observe("ㄱ", "ㄱ")
    probs = m.probabilities()
    row = m.target_labels.index("ㅎ")
    assert m.n_observations("ㅎ") == 0
    assert "ㅎ" in m.unobserved_targets
    assert np.allclose(probs[row], 1.0 / len(m.perceived_labels))


def test_observed_and_unobserved_partition_the_alphabet() -> None:
    m = ConfusionMatrix.empty(Position.CODA)
    m.observe(NO_CODA, NO_CODA)
    assert set(m.observed_targets) | set(m.unobserved_targets) == set(m.target_labels)
    assert not set(m.observed_targets) & set(m.unobserved_targets)


def test_empirical_probabilities_are_nan_for_unobserved_rows() -> None:
    m = ConfusionMatrix.empty(Position.NUCLEUS)
    m.observe("ㅏ", "ㅓ")
    emp = m.empirical_probabilities()
    r = m.target_labels.index("ㅏ")
    assert emp[r, m.perceived_labels.index("ㅓ")] == 1.0
    assert np.isnan(emp[m.target_labels.index("ㅣ")]).all()


def test_explicit_prior_shrinks_toward_the_group() -> None:
    group = ConfusionMatrix.empty(Position.ONSET)
    for _ in range(100):
        group.observe("ㄱ", "ㅋ")  # a group that reliably confuses ㄱ with ㅋ
    prior = group.probabilities()

    individual = ConfusionMatrix.empty(
        Position.ONSET, SmoothingSpec(alpha=5.0, kind="explicit", prior=prior)
    )
    individual.observe("ㄱ", "ㄱ")  # one contrary observation
    p = individual.probabilities()
    r = individual.target_labels.index("ㄱ")
    # Shrinkage pulls the estimate toward the group's ㄱ->ㅋ mass.
    assert (
        p[r, individual.perceived_labels.index("ㅋ")] > p[r, individual.target_labels.index("ㄴ")]
    )
    assert np.isclose(p.sum(axis=1), 1.0).all()


def test_smoothing_spec_validation() -> None:
    with pytest.raises(ValueError, match="alpha must be >= 0"):
        SmoothingSpec(alpha=-1.0)
    with pytest.raises(ValueError, match="requires a prior matrix"):
        SmoothingSpec(kind="explicit")
    bad = np.full((19, 20), 0.5)
    with pytest.raises(ValueError, match="rows must each sum to 1"):
        SmoothingSpec(kind="explicit", prior=bad)
    ok = np.full((19, 20), 1.0 / 20)
    with pytest.raises(ValueError, match="prior matrix supplied but kind is not"):
        SmoothingSpec(kind="uniform", prior=ok)


def test_zero_alpha_with_empty_row_raises_rather_than_returning_nonsense() -> None:
    m = ConfusionMatrix.empty(Position.ONSET, SmoothingSpec(alpha=0.0))
    with pytest.raises(ValueError, match="alpha=0 leaves rows with no observations"):
        m.probabilities()


def test_matrix_rejects_bad_shape_and_negative_counts() -> None:
    with pytest.raises(ValueError, match="must have shape"):
        ConfusionMatrix(position=Position.ONSET, counts=np.zeros((2, 2), dtype=np.int64))
    bad = np.zeros((19, 20), dtype=np.int64)
    bad[0, 0] = -1
    with pytest.raises(ValueError, match="non-negative"):
        ConfusionMatrix(position=Position.ONSET, counts=bad)
    with pytest.raises(TypeError, match="integer array"):
        ConfusionMatrix(position=Position.ONSET, counts=np.zeros((19, 20)))  # type: ignore[arg-type]


def test_unknown_category_raises_keyerror() -> None:
    m = ConfusionMatrix.empty(Position.ONSET)
    with pytest.raises(KeyError, match="not a valid onset target"):
        m.observe("ㅏ", "ㄱ")
    with pytest.raises(KeyError, match="not a valid onset perceived"):
        m.observe("ㄱ", "ㅏ")


def test_top_confusions_excludes_the_correct_answer_and_reports_counts() -> None:
    m = ConfusionMatrix.empty(Position.ONSET)
    for _ in range(10):
        m.observe("ㄱ", "ㅋ")
    for _ in range(3):
        m.observe("ㄱ", "ㄲ")
    top = m.top_confusions("ㄱ", k=2)
    assert [label for label, _, _ in top] == ["ㅋ", "ㄲ"]
    assert [count for _, _, count in top] == [10, 3]
    assert all(label != "ㄱ" for label, _, _ in top)


def test_matrix_addition_pools_counts() -> None:
    a = ConfusionMatrix.empty(Position.CODA)
    b = ConfusionMatrix.empty(Position.CODA)
    a.observe("ㄱ", NO_CODA)
    b.observe("ㄱ", NO_CODA)
    assert (a + b).n_observations("ㄱ") == 2
    with pytest.raises(ValueError, match="cannot add"):
        _ = a + ConfusionMatrix.empty(Position.ONSET)


def test_row_entropy_is_lower_for_a_confident_row() -> None:
    m = ConfusionMatrix.empty(Position.ONSET)
    for _ in range(50):
        m.observe("ㄱ", "ㄱ")
    for _ in range(50):
        m.observe("ㄷ", "ㄷ")
        m.observe("ㄷ", "ㄸ")
    ent = m.row_entropy()
    assert ent[m.target_labels.index("ㄱ")] < ent[m.target_labels.index("ㄷ")]


def test_matrix_json_roundtrip_preserves_counts() -> None:
    m = ConfusionMatrix.empty(Position.CODA)
    m.observe("ㄱ", NO_CODA, weight=4)
    back = ConfusionMatrix.from_dict(m.to_dict())
    assert np.array_equal(back.counts, m.counts)
    assert back.position is Position.CODA


# =========================================================== profile


def _trials(pairs: list[tuple[str, str]]) -> list[CalibrationTrial]:
    return [
        CalibrationTrial(stimulus_id=f"s{i}", target=t, response=r)
        for i, (t, r) in enumerate(pairs)
    ]


def test_profile_from_trials_accumulates_all_positions() -> None:
    prof = ConfusionProfile.from_trials(
        "L001", _trials([("각", "각"), ("각", "닥"), ("가", "각")]), is_synthetic=False
    )
    assert prof.n_trials == 3
    assert prof.total_observations == 9
    assert prof.matrix(Position.ONSET).n_observations("ㄱ") == 3
    assert (
        prof.matrix(Position.CODA).counts[
            prof.matrix(Position.CODA).target_labels.index(NO_CODA),
            prof.matrix(Position.CODA).perceived_labels.index("ㄱ"),
        ]
        == 1
    )


def test_profile_counts_unusable_responses() -> None:
    prof = ConfusionProfile.from_trials(
        "L002", _trials([("각", "각"), ("각", ""), ("각", "xyz")]), is_synthetic=False
    )
    assert prof.n_unusable_responses == 2
    m = prof.matrix(Position.ONSET)
    assert m.counts[m.target_labels.index("ㄱ"), m.perceived_labels.index(NO_RESPONSE)] == 2


def test_profile_coverage_is_honest_about_short_calibrations() -> None:
    prof = ConfusionProfile.from_trials("L003", _trials([("각", "각")]), is_synthetic=False)
    cov = prof.coverage
    assert cov["onset"] == pytest.approx(1 / 19)
    assert cov["nucleus"] == pytest.approx(1 / 21)
    assert cov["coda"] == pytest.approx(1 / 28)


def test_profile_overall_accuracy_and_empty_case() -> None:
    empty = ConfusionProfile.empty("L004", is_synthetic=True)
    assert empty.overall_accuracy() is None
    prof = ConfusionProfile.from_trials(
        "L005", _trials([("각", "각"), ("각", "닥")]), is_synthetic=True
    )
    assert prof.overall_accuracy() == pytest.approx(5 / 6)


def test_profile_requires_all_three_matrices() -> None:
    prof = ConfusionProfile.empty("L006", is_synthetic=True)
    with pytest.raises(ValueError, match="missing matrices"):
        ConfusionProfile(
            listener_id="L006",
            matrices={Position.ONSET: prof.matrix(Position.ONSET)},
            is_synthetic=True,
        )


def test_profile_json_roundtrip(tmp_path) -> None:
    prof = ConfusionProfile.from_trials(
        "L007", _trials([("각", "닥"), ("고", "구")]), is_synthetic=True
    )
    path = tmp_path / "p.json"
    prof.save_json(path)
    back = ConfusionProfile.load_json(path)
    assert back.listener_id == "L007"
    assert back.is_synthetic is True
    assert back.n_trials == 2
    for pos in Position:
        assert np.array_equal(back.matrix(pos).counts, prof.matrix(pos).counts)


def test_pooling_refuses_to_mix_synthetic_and_real() -> None:
    a = ConfusionProfile.from_trials("A", _trials([("각", "각")]), is_synthetic=True)
    b = ConfusionProfile.from_trials("B", _trials([("각", "각")]), is_synthetic=False)
    with pytest.raises(ValueError, match="refusing to pool synthetic and non-synthetic"):
        pool_profiles([a, b])
    with pytest.raises(ValueError, match="cannot pool an empty"):
        pool_profiles([])


def test_pooling_sums_counts_and_records_membership() -> None:
    a = ConfusionProfile.from_trials("A", _trials([("각", "각")]), is_synthetic=True)
    b = ConfusionProfile.from_trials("B", _trials([("각", "닥")]), is_synthetic=True)
    pooled = pool_profiles([a, b])
    assert pooled.matrix(Position.ONSET).n_observations("ㄱ") == 2
    assert pooled.provenance["n_listeners"] == 2
    assert pooled.is_synthetic is True


def test_with_smoothing_shares_counts_but_changes_estimates() -> None:
    prof = ConfusionProfile.from_trials("L008", _trials([("각", "각")]), is_synthetic=True)
    strong = prof.with_smoothing(SmoothingSpec(alpha=100.0))
    assert prof.evidence(Position.ONSET, "ㄱ") == strong.evidence(Position.ONSET, "ㄱ")
    assert strong.p_correct(Position.ONSET, "ㄱ") < prof.p_correct(Position.ONSET, "ㄱ")


# =========================================================== phonology tables


def test_coda_neutralisation_yields_exactly_seven_surface_consonants() -> None:
    from audire.hangul import CODA_JAMO

    surfaces = {neutralise_coda(c) for c in CODA_JAMO}
    assert surfaces == {"ㄱ", "ㄴ", "ㄷ", "ㄹ", "ㅁ", "ㅂ", "ㅇ"}
    assert len(NEUTRALISED_CODA_CATEGORIES) == 8  # seven surfaces + NO_CODA


def test_neutralise_passes_through_special_categories() -> None:
    assert neutralise_coda(NO_CODA) == NO_CODA
    assert neutralise_coda(NO_RESPONSE) == NO_RESPONSE
    with pytest.raises(KeyError, match="not a coda jamo"):
        neutralise_coda("ㄸ")


def test_onset_phonation_three_way_contrast() -> None:
    assert onset_features("ㄱ")["phonation"] == Phonation.LAX
    assert onset_features("ㄲ")["phonation"] == Phonation.TENSE
    assert onset_features("ㅋ")["phonation"] == Phonation.ASPIRATED
    assert onset_features("ㄱ")["place"] == onset_features("ㅋ")["place"] == "velar"


def test_onset_inventory_minus_null_is_eighteen() -> None:
    """Consistency check with Ma et al. (2026): 18 onset consonants."""
    from audire.hangul import ONSET_JAMO

    assert len([j for j in ONSET_JAMO if j != "ㅇ"]) == 18


def test_coda_cluster_features() -> None:
    assert coda_features("ㅄ") == {"surface": "ㅂ", "is_cluster": True, "present": True}
    assert coda_features(NO_CODA) == {"surface": NO_CODA, "is_cluster": False, "present": False}
