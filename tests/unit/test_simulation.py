"""E3 — synthetic cohort generation, determinism, provenance and parameter recovery."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from audire.confusion.profile import POSITIONS
from audire.data.stimuli import build_balanced_catalog
from audire.hangul.inventory import Position, categories_for
from audire.profile.schema import ProfileSource
from audire.sim import SimulationConfig, build_cohort, generate_cohort, similarity
from audire.sim.config import SeverityMix, TrialModel
from audire.sim.listener import solve_position_error_mass
from audire.sim.trials import build_vocabulary, simulate_word_trial

SMALL = SimulationConfig(
    name="test-small", n_listeners=8, n_calibration_trials=60, n_word_trials=40, seeds=[3]
)


# =========================================================== config contract


def test_literature_evidence_requires_a_registry_source() -> None:
    with pytest.raises(ValidationError, match="requires evidence_source"):
        SeverityMix(evidence="literature", evidence_source=None)


def test_evidence_report_labels_every_parameter_block() -> None:
    report = SMALL.evidence_report()
    assert {r["block"] for r in report} == {
        "severity",
        "audiogram",
        "speech",
        "confusion",
        "trials",
        "words",
    }
    for row in report:
        assert row["evidence"] in {"literature", "clinical_convention", "assumption"}
        assert row["rationale"], row["block"]
        if row["evidence"] == "literature":
            assert row["evidence_source"]


def test_evidence_sources_exist_in_the_registry() -> None:
    """A literature claim must point at something actually registered."""
    from audire.data.sources import registry

    known = set(registry().literature)
    for row in SMALL.evidence_report():
        if row["evidence"] == "literature":
            assert row["evidence_source"] in known, row


def test_config_rejects_strata_without_parameters() -> None:
    with pytest.raises(ValidationError, match="has no entry for strata"):
        SimulationConfig(name="bad", severity=SeverityMix(weights={"catastrophic": 1.0}))


def test_config_rejects_unknown_speakers_and_duplicate_seeds() -> None:
    with pytest.raises(ValidationError, match="speaker_effects has no entry"):
        SimulationConfig(name="bad", speakers=["martian"])
    with pytest.raises(ValidationError, match="seeds must be distinct"):
        SimulationConfig(name="bad", seeds=[1, 1])


# =========================================================== accuracy anchoring


@pytest.mark.parametrize("accuracy", [0.817, 0.722, 0.516, 0.196, 0.05, 0.99])
def test_position_error_mass_reproduces_the_syllable_accuracy(accuracy: float) -> None:
    """The literature reports whole-syllable rates; matrices are parameterised per position."""
    m = (1.15, 0.70, 1.15)
    e = solve_position_error_mass(accuracy, m)
    assert float(np.prod([1 - e * mi for mi in m])) == pytest.approx(accuracy, abs=1e-6)


def test_position_error_mass_rejects_non_positive_multipliers() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        solve_position_error_mass(0.8, (1.0, 0.0, 1.0))


def test_cohort_reproduces_the_configured_stratum_accuracies() -> None:
    """The generator must land near the literature-anchored accuracies it was given."""
    cfg = SimulationConfig(
        name="anchor", n_listeners=240, n_calibration_trials=1, n_word_trials=1, seeds=[5]
    )
    cohort = build_cohort(cfg, 5)
    achieved = cohort.syllable_accuracy_by_stratum()
    configured = cfg.speech.accuracy_by_stratum
    for stratum, value in achieved.items():
        # The between-listener spread is on the logit scale, so the mean accuracy sits
        # slightly below the configured centre for strata above 50 % (Jensen). A 0.06
        # tolerance keeps the check meaningful without asserting the gap away.
        assert value == pytest.approx(configured[stratum], abs=0.06), stratum


def test_mishear_rate_increases_monotonically_with_severity() -> None:
    cfg = SimulationConfig(
        name="mono", n_listeners=120, n_calibration_trials=20, n_word_trials=60, seeds=[9]
    )
    rates = build_cohort(cfg, 9).mishear_rate_by_stratum()
    ordered = [rates[s] for s in ("normal", "mild", "moderate", "severe") if s in rates]
    assert ordered == sorted(ordered), rates


# =========================================================== determinism


def test_cohort_generation_is_deterministic() -> None:
    a, b = build_cohort(SMALL, 3), build_cohort(SMALL, 3)
    assert [r.listener.ability_logit for r in a.records] == [
        r.listener.ability_logit for r in b.records
    ]
    assert [t.response for t in a.records[0].calibration] == [
        t.response for t in b.records[0].calibration
    ]
    assert [t.misheard for t in a.records[2].word_trials] == [
        t.misheard for t in b.records[2].word_trials
    ]


def test_cohort_generation_is_seed_sensitive() -> None:
    a, b = build_cohort(SMALL, 3), build_cohort(SMALL, 4)
    assert [r.listener.ability_logit for r in a.records] != [
        r.listener.ability_logit for r in b.records
    ]


def test_adding_a_listener_does_not_perturb_earlier_listeners_trials() -> None:
    """Independent per-listener substreams keep experiments comparable across cohort sizes."""
    small = build_cohort(SMALL, 3)
    bigger = build_cohort(SMALL.model_copy(update={"n_listeners": 12}), 3)
    assert [t.response for t in small.records[0].calibration] == [
        t.response for t in bigger.records[0].calibration
    ]


# =========================================================== synthetic provenance


def test_every_generated_object_is_marked_synthetic() -> None:
    cohort = build_cohort(SMALL, 3)
    assert cohort.is_synthetic
    for r in cohort.records:
        assert r.listener.is_synthetic
        assert r.hearing.is_synthetic
        assert r.hearing.source is ProfileSource.SYNTHETIC
        assert r.estimated_confusion.is_synthetic
        assert all(t.is_synthetic for t in r.word_trials)
    assert all(row["is_synthetic"] for row in cohort.word_trial_rows())


def test_model_inputs_never_expose_the_true_confusion_structure() -> None:
    """A model must only ever see what a real calibration could have produced."""
    cohort = build_cohort(SMALL, 3)
    for hearing, estimated, trials in cohort.model_inputs():
        assert not hasattr(estimated, "true_confusion")
        assert hearing.is_synthetic
        assert trials
    # The true structure is reachable only through the listener object itself.
    assert cohort.records[0].listener.true_confusion is not None


def test_cohort_summary_reports_evidence_and_base_rates() -> None:
    s = build_cohort(SMALL, 3).summary()
    assert s["is_synthetic"] is True
    assert 0.0 <= s["mishear_rate"] <= 1.0
    assert s["mishear_rate_by_stratum"]
    assert s["evidence"]
    assert s["vocabulary"]["license"] == "none (generated)"


# =========================================================== parameter recovery


@pytest.mark.slow
def test_estimated_diagonals_converge_to_the_truth_as_calibration_lengthens() -> None:
    """E3 — longer calibration must reduce the error between estimate and truth."""
    errors: dict[int, float] = {}
    for n in (25, 100, 400, 1600):
        cfg = SimulationConfig(
            name=f"recover-{n}",
            n_listeners=6,
            n_calibration_trials=n,
            n_word_trials=1,
            seeds=[21],
        )
        cohort = build_cohort(cfg, 21)
        diffs: list[float] = []
        for r in cohort.records:
            for position in POSITIONS:
                targets = categories_for(position, axis="target")
                est = r.estimated_confusion.matrix(position)
                true = r.listener.true_confusion
                for t in est.observed_targets:
                    if est.n_observations(t) < 3:
                        continue
                    diffs.append(abs(est.p_correct(t) - true.p_correct(position, t)))
                assert set(est.target_labels) == set(targets)
        errors[n] = float(np.mean(diffs))

    assert errors[1600] < errors[400] < errors[100] < errors[25], errors
    assert errors[1600] < 0.10, errors


def test_short_calibration_leaves_most_categories_unobserved() -> None:
    """The estimator must be honest that a 10-item list cannot cover 19 onsets."""
    cfg = SimulationConfig(
        name="short", n_listeners=3, n_calibration_trials=10, n_word_trials=1, seeds=[2]
    )
    cohort = build_cohort(cfg, 2)
    est = cohort.records[0].estimated_confusion
    assert len(est.matrix(Position.ONSET).unobserved_targets) >= 9
    assert est.coverage["onset"] <= 10 / 19 + 1e-9


# =========================================================== context effects


def test_lower_snr_increases_mishearing() -> None:
    cohort = build_cohort(SMALL, 3)
    listener = cohort.records[0].listener
    cfg = SMALL
    rates: dict[float, float] = {}
    for snr in (25.0, 5.0, -10.0):
        rng = np.random.default_rng(1)
        trials = [
            simulate_word_trial(listener, "가족", rng, cfg, snr_db=snr, speaker="male")
            for _ in range(600)
        ]
        rates[snr] = sum(t.misheard for t in trials) / len(trials)
    assert rates[-10.0] > rates[5.0] > rates[25.0], rates


def test_speaker_effect_moves_outcomes_in_the_declared_direction() -> None:
    cohort = build_cohort(SMALL, 3)
    listener = cohort.records[0].listener
    rates: dict[str, float] = {}
    for speaker in ("male", "female"):
        rng = np.random.default_rng(4)
        trials = [
            simulate_word_trial(listener, "가족", rng, SMALL, snr_db=20.0, speaker=speaker)
            for _ in range(600)
        ]
        rates[speaker] = sum(t.misheard for t in trials) / len(trials)
    # trials.speaker_effects declares female = +0.10 logit (easier), male = -0.10.
    assert rates["male"] > rates["female"], rates


def test_lexical_repair_is_the_declared_non_independence_mechanism() -> None:
    """With repair disabled, every segmental error must produce a misheard word."""
    cfg = SMALL.model_copy(
        update={
            "trials": TrialModel(
                lexical_repair_base=0.0,
                lexical_repair_multi_error=0.0,
                lexical_repair_max=0.0,
                evidence="assumption",
            )
        }
    )
    listener = generate_cohort(cfg, 3)[0]
    rng = np.random.default_rng(6)
    trials = [
        simulate_word_trial(listener, "가족", rng, cfg, snr_db=20.0, speaker="male")
        for _ in range(300)
    ]
    assert all(t.misheard == (t.n_segment_errors > 0) for t in trials)
    assert not any(t.repaired for t in trials)
    assert any(t.n_segment_errors > 0 for t in trials)


def test_repair_makes_longer_words_more_recoverable_from_a_single_error() -> None:
    from audire.sim.trials import _repair_probability

    assert _repair_probability(SMALL, 1, 1) < _repair_probability(SMALL, 3, 1)
    assert _repair_probability(SMALL, 3, 2) < _repair_probability(SMALL, 3, 1)
    assert _repair_probability(SMALL, 3, 0) == 0.0
    assert _repair_probability(SMALL, 99, 1) == SMALL.trials.lexical_repair_max


# =========================================================== similarity kernel


def test_similarity_is_reflexive_bounded_and_symmetric() -> None:
    for position in Position:
        cats = categories_for(position, axis="target")
        for a in cats:
            assert similarity(position, a, a) == 1.0
            for b in cats:
                s = similarity(position, a, b)
                assert 0.0 <= s <= 1.0
                assert s == similarity(position, b, a)


def test_similarity_reflects_korean_phonology() -> None:
    # ㄱ / ㅋ share place and manner, differ in phonation.
    assert similarity(Position.ONSET, "ㄱ", "ㅋ") > similarity(Position.ONSET, "ㄱ", "ㅁ")
    # Adding or dropping a coda is categorically different from swapping one.
    assert similarity(Position.CODA, "-", "ㄱ") == 0.0
    assert similarity(Position.CODA, "ㄱ", "ㄲ") == 1.0  # same neutralised surface


def test_confusions_concentrate_within_phonetic_classes() -> None:
    """The simulator must produce the qualitative pattern its config claims to encode."""
    cfg = SimulationConfig(
        name="conc", n_listeners=40, n_calibration_trials=1, n_word_trials=1, seeds=[8]
    )
    listeners = generate_cohort(cfg, 8)
    within: list[float] = []
    across: list[float] = []
    labels = categories_for(Position.ONSET, axis="perceived")
    for lst in listeners:
        row = lst.true_confusion.row(Position.ONSET, "ㄱ")
        for j, label in enumerate(labels):
            if label == "ㄱ" or label == "?":
                continue
            (within if similarity(Position.ONSET, "ㄱ", label) >= 2 / 3 else across).append(
                float(row[j])
            )
    assert float(np.mean(within)) > float(np.mean(across))


# =========================================================== vocabulary


def test_vocabulary_is_deterministic_unique_and_licence_free() -> None:
    v1 = build_vocabulary(SMALL, 3)
    v2 = build_vocabulary(SMALL, 3)
    assert v1.words == v2.words
    assert len(set(v1.words)) == len(v1.words) == SMALL.words.vocabulary_size
    assert v1.provenance["license"] == "none (generated)"
    assert build_vocabulary(SMALL, 4).words != v1.words


def test_vocabulary_syllable_lengths_follow_the_configured_weights() -> None:
    v = build_vocabulary(SMALL, 3)
    lengths = {len(w) for w in v.words}
    assert lengths <= set(SMALL.words.syllable_count_weights)


def test_calibration_uses_the_balanced_catalog_by_default() -> None:
    cohort = build_cohort(SMALL, 3)
    expected = [s.syllable for s in build_balanced_catalog(SMALL.n_calibration_trials)]
    assert [t.target for t in cohort.records[0].calibration] == expected
    assert cohort.calibration_catalog.provenance["deterministic"] is True


def test_cohort_lookup_by_listener_id() -> None:
    cohort = build_cohort(SMALL, 3)
    assert cohort.record(cohort.listener_ids[1]).listener_id == cohort.listener_ids[1]
    with pytest.raises(KeyError, match="no listener"):
        cohort.record("nobody")
