"""HearingProfile schema: missingness, derived measures, validation and private storage."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from audire.profile import (
    Audiogram,
    AudiogramPoint,
    Ear,
    EarProfile,
    HearingProfile,
    LoudnessLevels,
    PIFunction,
    PIPoint,
    ProfileSource,
    ProfileStore,
    ProfileStoreError,
    PTAMethod,
    SeverityScheme,
    SpeechScores,
    severity_stratum,
)
from audire.profile.schema import PTA_CALC_VERSION


def _flat_audiogram(
    ear: Ear, db: float, freqs: tuple[int, ...] = (500, 1000, 2000, 4000)
) -> Audiogram:
    return Audiogram(ear=ear, thresholds={f: AudiogramPoint(db_hl=db) for f in freqs})


def _profile(**kw: object) -> HearingProfile:
    db = float(kw.pop("db", 30.0))
    speech_kw = dict(kw.pop("speech", {}))  # type: ignore[arg-type]
    right = EarProfile(
        ear=Ear.RIGHT,
        audiogram=_flat_audiogram(Ear.RIGHT, db),
        speech=SpeechScores(ear=Ear.RIGHT, **speech_kw),
    )
    return HearingProfile(
        listener_id="L1", source=ProfileSource.MANUAL, is_synthetic=False, right=right, **kw
    )


# =========================================================== audiogram / PTA


def test_pta_is_none_when_a_required_frequency_is_missing() -> None:
    """A partial average would be incomparable with a PTA computed elsewhere."""
    a = Audiogram(
        ear=Ear.LEFT, thresholds={500: AudiogramPoint(db_hl=30), 1000: AudiogramPoint(db_hl=30)}
    )
    assert a.pta(PTAMethod.PTA4) is None
    assert a.pta(PTAMethod.PTA3) is None
    detail = a.pta_detail(PTAMethod.PTA4)
    assert detail["frequencies_missing"] == [2000, 4000]
    assert detail["value_db_hl"] is None
    assert detail["calc_version"] == PTA_CALC_VERSION


def test_pta_methods_disagree_and_each_names_itself() -> None:
    a = Audiogram(
        ear=Ear.LEFT,
        thresholds={
            500: AudiogramPoint(db_hl=20),
            1000: AudiogramPoint(db_hl=20),
            2000: AudiogramPoint(db_hl=40),
            3000: AudiogramPoint(db_hl=60),
            4000: AudiogramPoint(db_hl=80),
        },
    )
    assert a.pta(PTAMethod.PTA3) == pytest.approx(80 / 3)
    assert a.pta(PTAMethod.PTA4) == pytest.approx(40.0)
    assert a.pta(PTAMethod.PTA4_3K) == pytest.approx(35.0)
    assert a.pta(PTAMethod.PTA4) != a.pta(PTAMethod.PTA3)
    assert a.pta_detail(PTAMethod.PTA3)["method"] == "pta3_500_1000_2000"


def test_no_response_is_not_a_threshold() -> None:
    """ "No response at max output" must not be averaged in as if it were a threshold."""
    a = Audiogram(
        ear=Ear.LEFT,
        thresholds={
            500: AudiogramPoint(db_hl=60),
            1000: AudiogramPoint(db_hl=70),
            2000: AudiogramPoint(db_hl=90),
            4000: AudiogramPoint(db_hl=120, no_response=True),
        },
    )
    assert a.pta(PTAMethod.PTA4) is None
    assert 4000 not in a.measured_frequencies()
    assert a.pta(PTAMethod.PTA3) == pytest.approx((60 + 70 + 90) / 3)


def test_no_response_requires_the_level_at_which_it_was_obtained() -> None:
    with pytest.raises(ValidationError, match="no_response=True requires db_hl"):
        AudiogramPoint(no_response=True)


def test_thresholds_outside_plausible_bounds_are_rejected() -> None:
    with pytest.raises(ValidationError):
        AudiogramPoint(db_hl=-40)
    with pytest.raises(ValidationError):
        AudiogramPoint(db_hl=200)


def test_non_standard_frequency_is_rejected() -> None:
    with pytest.raises(ValidationError, match="non-standard audiometric frequencies"):
        Audiogram(ear=Ear.LEFT, thresholds={1500: AudiogramPoint(db_hl=20)})


def test_slope_detects_a_sloping_configuration() -> None:
    sloping = Audiogram(
        ear=Ear.LEFT,
        thresholds={500: AudiogramPoint(db_hl=20), 4000: AudiogramPoint(db_hl=65)},
    )
    assert sloping.slope_db_per_octave() == pytest.approx(15.0)
    flat = _flat_audiogram(Ear.LEFT, 40.0)
    assert flat.slope_db_per_octave() == pytest.approx(0.0)
    assert Audiogram(ear=Ear.LEFT, thresholds={}).slope_db_per_octave() is None


# =========================================================== severity strata


@pytest.mark.parametrize(
    ("pta", "expected"),
    [
        (0.0, "normal"),
        (19.9, "normal"),
        (20.0, "mild"),
        (34.9, "mild"),
        (35.0, "moderate"),
        (49.9, "moderate"),
        (50.0, "moderately_severe"),
        (65.0, "severe"),
        (80.0, "profound"),
        (95.0, "complete"),
        (130.0, "complete"),
    ],
)
def test_who_severity_boundaries(pta: float, expected: str) -> None:
    assert severity_stratum(pta, SeverityScheme.WHO2021) == expected


def test_korean_four_group_scheme_collapses_the_top() -> None:
    assert severity_stratum(70.0, SeverityScheme.KOREAN_STUDY_4GROUP) == "severe"
    assert severity_stratum(100.0, SeverityScheme.KOREAN_STUDY_4GROUP) == "severe"
    assert severity_stratum(10.0, SeverityScheme.KOREAN_STUDY_4GROUP) == "normal"


def test_summary_carries_the_non_diagnosis_disclaimer() -> None:
    s = _profile().summary()
    assert "not a diagnosis" in s["disclaimer"]
    assert s["severity_scheme"] == "who2021"
    assert s["pta_method"] == "pta4_500_1000_2000_4000"


# =========================================================== speech scores


def test_wrs_without_a_presentation_level_is_rejected() -> None:
    """A word recognition score is uninterpretable without its level."""
    with pytest.raises(ValidationError, match="requires wrs_presentation_level_db_hl"):
        SpeechScores(ear=Ear.LEFT, wrs_percent=72.0)
    SpeechScores(ear=Ear.LEFT, wrs_percent=72.0, wrs_presentation_level_db_hl=70.0)


def test_pi_function_derived_measures() -> None:
    pi = PIFunction(
        ear=Ear.LEFT,
        points=[
            PIPoint(level_db_hl=40, score_percent=40),
            PIPoint(level_db_hl=60, score_percent=80),
            PIPoint(level_db_hl=80, score_percent=50),
        ],
    )
    assert pi.pbmax_percent == 80.0
    assert pi.pbmax_level_db_hl == 60.0
    assert pi.rollover_index == pytest.approx((80 - 50) / 80)


def test_pi_rollover_is_none_without_a_level_above_the_peak() -> None:
    pi = PIFunction(
        ear=Ear.LEFT,
        points=[
            PIPoint(level_db_hl=40, score_percent=40),
            PIPoint(level_db_hl=60, score_percent=80),
        ],
    )
    assert pi.rollover_index is None


def test_pi_function_rejects_duplicate_levels() -> None:
    with pytest.raises(ValidationError, match="duplicate presentation levels"):
        PIFunction(
            ear=Ear.LEFT,
            points=[
                PIPoint(level_db_hl=60, score_percent=80),
                PIPoint(level_db_hl=60, score_percent=70),
            ],
        )


def test_ucl_below_mcl_is_rejected() -> None:
    with pytest.raises(ValidationError, match="cannot be below MCL"):
        LoudnessLevels(ear=Ear.LEFT, mcl_db_hl=70, ucl_db_hl=60)
    lv = LoudnessLevels(ear=Ear.LEFT, mcl_db_hl=60, ucl_db_hl=95)
    assert lv.dynamic_range_db == pytest.approx(35.0)
    assert LoudnessLevels(ear=Ear.LEFT, mcl_db_hl=60).dynamic_range_db is None


# =========================================================== profile-level


def test_missingness_is_explicit_and_never_imputed() -> None:
    p = _profile()
    missing = set(p.missing())
    assert {"srt", "wrs", "pbmax", "mcl"} <= missing
    assert p.available()["better_ear_pta"] is True
    assert 0.0 < p.completeness() < 1.0
    # Nothing was silently filled in.
    assert p.best_srt() is None and p.best_wrs() is None and p.pbmax() is None


def test_full_profile_reports_no_missing_clinical_fields() -> None:
    ear = EarProfile(
        ear=Ear.RIGHT,
        audiogram=Audiogram(
            ear=Ear.RIGHT,
            thresholds={
                f: AudiogramPoint(db_hl=v)
                for f, v in ((500, 30), (1000, 35), (2000, 45), (4000, 60))
            },
        ),
        speech=SpeechScores(
            ear=Ear.RIGHT,
            srt_db_hl=35,
            wrs_percent=76,
            wrs_presentation_level_db_hl=70,
            wrs_word_list="KS-MWL-A list 1",
            wrs_n_words=50,
        ),
        pi_function=PIFunction(
            ear=Ear.RIGHT,
            points=[
                PIPoint(level_db_hl=70, score_percent=76),
                PIPoint(level_db_hl=85, score_percent=82),
            ],
        ),
        loudness=LoudnessLevels(ear=Ear.RIGHT, mcl_db_hl=70, ucl_db_hl=100),
    )
    p = HearingProfile(
        listener_id="L2", source=ProfileSource.CLINICAL_EXPORT, is_synthetic=False, right=ear
    )
    assert p.missing() == ()
    assert p.completeness() == 1.0
    assert p.pbmax() == 82.0
    assert p.severity() == "moderate"


def test_better_and_worse_ear_pta() -> None:
    left = EarProfile(
        ear=Ear.LEFT, audiogram=_flat_audiogram(Ear.LEFT, 60.0), speech=SpeechScores(ear=Ear.LEFT)
    )
    right = EarProfile(
        ear=Ear.RIGHT,
        audiogram=_flat_audiogram(Ear.RIGHT, 25.0),
        speech=SpeechScores(ear=Ear.RIGHT),
    )
    p = HearingProfile(
        listener_id="L3",
        source=ProfileSource.MANUAL,
        is_synthetic=False,
        left=left,
        right=right,
    )
    assert p.better_ear_pta() == 25.0
    assert p.worse_ear_pta() == 60.0
    assert p.severity() == "mild"


def test_profile_requires_at_least_one_ear() -> None:
    with pytest.raises(ValidationError, match="at least one ear"):
        HearingProfile(listener_id="L4", source=ProfileSource.MANUAL, is_synthetic=False)


def test_ear_fields_must_hold_the_matching_ear() -> None:
    right_shaped = EarProfile(
        ear=Ear.RIGHT,
        audiogram=_flat_audiogram(Ear.RIGHT, 20.0),
        speech=SpeechScores(ear=Ear.RIGHT),
    )
    with pytest.raises(ValidationError, match="`left` field must hold a left-ear profile"):
        HearingProfile(
            listener_id="L5",
            source=ProfileSource.MANUAL,
            is_synthetic=False,
            left=right_shaped,
        )


def test_ear_profile_rejects_internally_mismatched_parts() -> None:
    with pytest.raises(ValidationError, match="ear mismatch"):
        EarProfile(
            ear=Ear.RIGHT,
            audiogram=_flat_audiogram(Ear.LEFT, 20.0),
            speech=SpeechScores(ear=Ear.RIGHT),
        )


def test_synthetic_provenance_cannot_be_partially_declared() -> None:
    ear = EarProfile(
        ear=Ear.RIGHT,
        audiogram=_flat_audiogram(Ear.RIGHT, 20.0),
        speech=SpeechScores(ear=Ear.RIGHT),
    )
    with pytest.raises(ValidationError, match="cannot be partially declared"):
        HearingProfile(
            listener_id="L6", source=ProfileSource.SYNTHETIC, is_synthetic=False, right=ear
        )
    with pytest.raises(ValidationError, match="cannot be partially declared"):
        HearingProfile(listener_id="L7", source=ProfileSource.MANUAL, is_synthetic=True, right=ear)


def test_unknown_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        SpeechScores(ear=Ear.LEFT, wrs=80)  # type: ignore[call-arg]


def test_profile_json_roundtrip() -> None:
    p = _profile(
        speech={"srt_db_hl": 30.0, "wrs_percent": 80.0, "wrs_presentation_level_db_hl": 65.0}
    )
    back = HearingProfile.model_validate_json(p.model_dump_json())
    assert back.better_ear_pta() == p.better_ear_pta()
    assert back.best_wrs() == 80.0


# =========================================================== private store


def test_store_roundtrip_and_listing(tmp_path) -> None:
    store = ProfileStore(tmp_path)
    p = _profile()
    store.save_hearing(p)
    assert store.exists("L1")
    assert store.list_ids() == ["L1"]
    assert store.load("L1").hearing.listener_id == "L1"
    assert store.load("L1").has_calibration is False


def test_store_rejects_unsafe_listener_ids(tmp_path) -> None:
    store = ProfileStore(tmp_path)
    for bad in ("../escape", "a/b", "", "x" * 65, ".hidden", "김철수"):
        with pytest.raises(ProfileStoreError, match="invalid listener id"):
            store.hearing_path(bad)


def test_store_export_and_delete(tmp_path) -> None:
    store = ProfileStore(tmp_path)
    store.save_hearing(_profile())
    store.append_responses("L1", [{"stimulus_id": "s0", "target": "각", "response": "닥"}])

    exported = store.export("L1")
    assert exported["hearing_profile"]["listener_id"] == "L1"
    assert exported["calibration_responses"][0]["response"] == "닥"

    removed = store.delete("L1")
    assert any("calibration_responses.jsonl" in r for r in removed)
    assert store.list_ids() == []
    with pytest.raises(ProfileStoreError, match="nothing stored"):
        store.delete("L1")


def test_store_missing_profile_is_an_error(tmp_path) -> None:
    with pytest.raises(ProfileStoreError, match="no hearing profile"):
        ProfileStore(tmp_path).load("nobody")


def test_store_defaults_to_the_gitignored_private_directory() -> None:
    """The safe location must be the one you get by doing nothing."""
    from audire.config.paths import private_dir

    assert ProfileStore().root == (private_dir() / "profiles").resolve()
