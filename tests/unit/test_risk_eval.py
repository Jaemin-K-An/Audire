"""G3 — features, models, calibration, metrics, listener-level splits and leakage."""

from __future__ import annotations

import numpy as np
import pytest

from audire.confusion import CalibrationTrial, ConfusionProfile
from audire.eval import (
    LeakageError,
    LeakySplitter,
    assert_no_listener_leakage,
    bootstrap_metric,
    cohort_matrix,
    compute_metrics,
    contrast,
    evaluate_arm,
    expected_calibration_error,
    leave_one_listener_out,
    listener_folds,
    metric_statistic,
    paired_bootstrap_difference,
    prevalence_baseline_metrics,
    reliability_curve,
)
from audire.hangul.inventory import Position
from audire.profile.schema import (
    Audiogram,
    AudiogramPoint,
    Ear,
    EarProfile,
    HearingProfile,
    ProfileSource,
    SpeechScores,
)
from audire.risk import (
    ABLATION_ARMS,
    CalibratedRiskModel,
    FeatureSpec,
    LogisticRiskModel,
    PhonemeIndependenceRisk,
    WordContext,
    WordScorer,
    build_matrix,
    clinical_features,
    confusion_features,
    make_model,
    phoneme_independence_risk,
    phoneme_risks,
    pta_features,
    word_features,
)
from audire.sim import SimulationConfig, build_cohort

TINY = SimulationConfig(
    name="risk-tiny", n_listeners=25, n_calibration_trials=60, n_word_trials=60, seeds=[13]
)


@pytest.fixture(scope="module")
def cohort():
    return build_cohort(TINY, 13)


def _profile(db: float = 40.0, *, with_speech: bool = True) -> HearingProfile:
    ear = EarProfile(
        ear=Ear.RIGHT,
        audiogram=Audiogram(
            ear=Ear.RIGHT,
            thresholds={f: AudiogramPoint(db_hl=db) for f in (500, 1000, 2000, 4000)},
        ),
        speech=SpeechScores(
            ear=Ear.RIGHT,
            srt_db_hl=db if with_speech else None,
            wrs_percent=70.0 if with_speech else None,
            wrs_presentation_level_db_hl=70.0 if with_speech else None,
        ),
    )
    return HearingProfile(
        listener_id="L1", source=ProfileSource.MANUAL, is_synthetic=False, right=ear
    )


def _confusion(pairs: list[tuple[str, str]]) -> ConfusionProfile:
    return ConfusionProfile.from_trials(
        "L1",
        [
            CalibrationTrial(stimulus_id=f"s{i}", target=t, response=r)
            for i, (t, r) in enumerate(pairs)
        ],
        is_synthetic=False,
    )


# =========================================================== word features


def test_word_features_capture_length_and_structure() -> None:
    f = word_features("가족")
    assert f["w_n_syllables"] == 2.0
    assert f["w_n_phonemes"] == 6.0  # three positions per syllable, no-coda included
    assert f["w_n_codas"] == 1.0
    assert f["w_coda_ratio"] == 0.5
    assert f["w_struct_CV"] == 0.5 and f["w_struct_CVC"] == 0.5


def test_word_features_handle_a_non_hangul_token() -> None:
    f = word_features("OK123")
    assert f["w_n_syllables"] == 0.0
    assert set(word_features("가")) == set(f)  # identical column set


def test_cluster_coda_is_flagged() -> None:
    assert word_features("값")["w_has_cluster_coda"] == 1.0
    assert word_features("간")["w_has_cluster_coda"] == 0.0


# =========================================================== clinical features


def test_missing_clinical_values_are_nan_plus_an_indicator() -> None:
    f = clinical_features(_profile(with_speech=False))
    assert np.isnan(f["h_wrs"])
    assert f["h_wrs_missing"] == 1.0
    assert not np.isnan(f["h_pta_better"])
    assert f["h_pta_better_missing"] == 0.0


def test_clinical_block_is_a_strict_superset_of_the_pta_block() -> None:
    """RQ1 needs the arms to be nested, otherwise the contrast is not interpretable."""
    p = _profile()
    assert set(pta_features(p)) < set(clinical_features(p))
    assert "h_wrs" in clinical_features(p)
    assert "h_wrs" not in pta_features(p)


def test_srt_minus_pta_is_present_when_both_are() -> None:
    f = clinical_features(_profile(db=40.0))
    assert f["h_srt_minus_pta"] == pytest.approx(0.0)
    assert f["h_srt_minus_pta_missing"] == 0.0


# =========================================================== confusion features


def test_r_phon_is_the_documented_independence_product() -> None:
    prof = _confusion([("각", "각")] * 20 + [("닥", "닥")] * 20)
    risks = phoneme_risks("각", prof)
    expected = 1.0 - float(np.prod([r.p_correct for r in risks]))
    assert phoneme_independence_risk("각", prof) == pytest.approx(expected)
    assert confusion_features("각", prof)["x_r_phon"] == pytest.approx(expected)


def test_r_phon_rises_when_the_listener_confuses_a_phoneme_in_the_word() -> None:
    good = _confusion([("각", "각")] * 30)
    bad = _confusion([("각", "닥")] * 30)
    assert phoneme_independence_risk("각", bad) > phoneme_independence_risk("각", good)


def test_r_phon_is_zero_for_a_word_with_no_hangul() -> None:
    assert phoneme_independence_risk("abc", _confusion([("각", "각")])) == 0.0


def test_longer_words_are_riskier_under_independence() -> None:
    prof = _confusion([("각", "각")] * 20)
    assert phoneme_independence_risk("각각각", prof) > phoneme_independence_risk("각", prof)


def test_evidence_features_expose_thin_calibration() -> None:
    """A confident-looking estimate resting on no data must be visible to the model."""
    prof = _confusion([("각", "각")])
    f = confusion_features("힣", prof)  # none of ㅎ/ㅣ/ㅎ was ever presented
    assert f["x_min_evidence"] == 0.0
    assert f["x_frac_unobserved"] == 1.0
    rich = confusion_features("각", prof)
    assert rich["x_min_evidence"] >= 1.0
    assert rich["x_frac_unobserved"] == 0.0


def test_cluster_coda_backs_off_to_its_neutralised_surface() -> None:
    """ㅄ surfaces as ㅂ, so evidence about ㅂ is evidence about ㅄ."""
    prof = _confusion([("갑", "갑")] * 20)  # coda ㅂ observed, ㅄ never
    risks = {(r.position, r.target): r for r in phoneme_risks("값", prof)}
    coda = risks[(Position.CODA, "ㅄ")]
    assert coda.backed_off is True
    assert coda.n_observations == 20
    assert coda.p_correct == pytest.approx(prof.p_correct(Position.CODA, "ㅂ"))


def test_unobserved_onset_does_not_back_off_and_says_so() -> None:
    prof = _confusion([("각", "각")] * 20)
    risks = {(r.position, r.target): r for r in phoneme_risks("학", prof)}
    onset = risks[(Position.ONSET, "ㅎ")]
    assert onset.backed_off is False
    assert onset.n_observations == 0


def test_confusion_features_on_a_non_hangul_token_are_neutral() -> None:
    f = confusion_features("xyz", _confusion([("각", "각")]))
    assert f["x_r_phon"] == 0.0
    assert f["x_mean_p_correct"] == 1.0


# =========================================================== feature specs


def test_every_arm_shares_the_word_and_context_blocks() -> None:
    """Arms must differ only in the listener representation, or RQ1 is unanswerable."""
    for name, blocks in ABLATION_ARMS.items():
        assert "word" in blocks and "context" in blocks, name


def test_arm_requiring_a_profile_refuses_to_run_without_one() -> None:
    spec = FeatureSpec.arm("clinical")
    with pytest.raises(ValueError, match="needs a HearingProfile"):
        spec.row("가", WordContext(), None, None)
    spec2 = FeatureSpec.arm("confusion_only")
    with pytest.raises(ValueError, match="needs a ConfusionProfile"):
        spec2.row("가", WordContext(), _profile(), None)


def test_unknown_arm_is_rejected() -> None:
    with pytest.raises(KeyError, match="unknown arm"):
        FeatureSpec.arm("wishful_thinking")


def test_build_matrix_enforces_a_stable_column_set() -> None:
    spec = FeatureSpec.arm("word_context_only")
    rows = [("L1", "가", WordContext(), None, None), ("L1", "가족", WordContext(), None, None)]
    m = build_matrix(spec, rows, [0, 1])
    assert m.X.shape == (2, len(m.feature_names))
    assert m.groups.tolist() == ["L1", "L1"]
    assert m.y is not None and m.y.tolist() == [0, 1]

    with pytest.raises(ValueError, match="zero rows"):
        build_matrix(spec, [], [])
    with pytest.raises(ValueError, match="but 1 labels"):
        build_matrix(spec, rows, [0])


# =========================================================== metrics


def test_perfect_and_inverted_predictions() -> None:
    y = np.array([0, 0, 1, 1])
    perfect = compute_metrics(y, np.array([0.01, 0.02, 0.98, 0.99]))
    assert perfect.roc_auc == 1.0
    assert perfect.pr_auc == 1.0
    assert perfect.recall == 1.0 and perfect.specificity == 1.0

    inverted = compute_metrics(y, np.array([0.99, 0.98, 0.02, 0.01]))
    assert inverted.roc_auc == 0.0
    assert inverted.brier > perfect.brier


def test_single_class_sample_yields_nan_discrimination_not_a_crash() -> None:
    """A degenerate evaluation fold must be visible, not silently scored."""
    m = compute_metrics(np.array([1, 1, 1]), np.array([0.6, 0.7, 0.8]))
    assert np.isnan(m.pr_auc) and np.isnan(m.roc_auc)
    assert np.isfinite(m.brier)
    assert m.base_rate == 1.0


def test_metric_input_validation() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        compute_metrics(np.array([0, 1]), np.array([0.5]))
    with pytest.raises(ValueError, match="empty sample"):
        compute_metrics(np.array([], dtype=np.int64), np.array([]))
    with pytest.raises(ValueError, match="only 0 and 1"):
        compute_metrics(np.array([0, 2]), np.array([0.5, 0.5]))
    with pytest.raises(ValueError, match="must lie in"):
        compute_metrics(np.array([0, 1]), np.array([-0.1, 0.5]))
    with pytest.raises(ValueError, match="non-finite"):
        compute_metrics(np.array([0, 1]), np.array([np.nan, 0.5]))


def test_ece_is_zero_for_a_perfectly_calibrated_predictor() -> None:
    rng = np.random.default_rng(0)
    p = np.full(20000, 0.3)
    y = (rng.random(20000) < 0.3).astype(np.int64)
    ece, mce = expected_calibration_error(y, p, n_bins=10)
    assert ece < 0.02 and mce < 0.02


def test_ece_is_large_for_a_confidently_wrong_predictor() -> None:
    y = np.zeros(1000, dtype=np.int64)
    ece, mce = expected_calibration_error(y, np.full(1000, 0.95))
    assert ece == pytest.approx(0.95, abs=1e-9)
    assert mce == pytest.approx(0.95, abs=1e-9)


def test_ece_requires_at_least_two_bins() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        expected_calibration_error(np.array([0, 1]), np.array([0.2, 0.8]), n_bins=1)


def test_probability_one_lands_in_the_last_bin() -> None:
    curve = reliability_curve(np.array([1, 1]), np.array([1.0, 1.0]), n_bins=10)
    assert curve["bin_centre"] == [0.95]
    assert curve["count"] == [2.0]


def test_reliability_curve_omits_empty_bins() -> None:
    y = np.array([0, 1, 0, 1])
    curve = reliability_curve(y, np.array([0.05, 0.95, 0.05, 0.95]), n_bins=10)
    assert len(curve["bin_centre"]) == 2


def test_prevalence_floor_has_chance_discrimination() -> None:
    floor = prevalence_baseline_metrics(np.array([0, 0, 0, 1]))
    assert floor.pr_auc == pytest.approx(0.25, abs=0.3)
    assert floor.base_rate == 0.25


# =========================================================== splits & leakage


def test_listener_folds_never_leak() -> None:
    groups = np.array([f"L{i // 20:02d}" for i in range(400)], dtype=np.str_)
    y = np.array([i % 3 == 0 for i in range(400)], dtype=np.int64)
    folds = listener_folds(groups, y, n_splits=5, seed=0)
    assert len(folds) == 5
    for f in folds:
        assert not set(f.train_listeners) & set(f.test_listeners)
        assert f.n_train + f.n_test == 400


def test_every_row_is_tested_exactly_once() -> None:
    groups = np.array([f"L{i // 10:02d}" for i in range(200)], dtype=np.str_)
    y = np.array([i % 4 == 0 for i in range(200)], dtype=np.int64)
    tested = np.concatenate([f.test_idx for f in listener_folds(groups, y, n_splits=4)])
    assert sorted(tested.tolist()) == list(range(200))


def test_leakage_guard_fires_on_a_deliberately_leaky_split() -> None:
    """The guard must be shown to work, not merely assumed to."""
    groups = np.array([f"L{i // 20:02d}" for i in range(400)], dtype=np.str_)
    leaked = 0
    for train, test in LeakySplitter(n_splits=5, seed=0).split(groups):
        with pytest.raises(LeakageError, match="appear in both train and test"):
            assert_no_listener_leakage(groups, train, test)
        leaked += 1
    assert leaked == 5


def test_too_few_listeners_is_an_error_not_a_silent_fold_reduction() -> None:
    groups = np.array(["A"] * 10 + ["B"] * 10, dtype=np.str_)
    with pytest.raises(ValueError, match="cannot build 5 listener-level folds"):
        listener_folds(groups, np.zeros(20, dtype=np.int64), n_splits=5)


def test_stratified_folds_need_labels() -> None:
    groups = np.array([f"L{i}" for i in range(10)], dtype=np.str_)
    with pytest.raises(ValueError, match="need labels"):
        listener_folds(groups, None, n_splits=2, stratify=True)
    assert len(listener_folds(groups, None, n_splits=2, stratify=False)) == 2


def test_leave_one_listener_out() -> None:
    groups = np.array(["A", "A", "B", "B", "C"], dtype=np.str_)
    folds = leave_one_listener_out(groups)
    assert [f.test_listeners for f in folds] == [("A",), ("B",), ("C",)]
    assert all(len(f.test_listeners) == 1 for f in folds)


# =========================================================== bootstrap


def test_bootstrap_interval_brackets_the_point_estimate() -> None:
    rng = np.random.default_rng(2)
    groups = np.array([f"L{i // 50:02d}" for i in range(1000)], dtype=np.str_)
    y = (rng.random(1000) < 0.4).astype(np.int64)
    p = np.clip(y * 0.4 + rng.random(1000) * 0.5, 0, 1)
    from sklearn.metrics import average_precision_score

    iv = bootstrap_metric(
        groups, metric_statistic(y, p, average_precision_score), n_resamples=200, seed=1
    )
    assert iv.lo <= iv.point <= iv.hi
    assert iv.n_valid > 150


def test_paired_bootstrap_of_a_model_against_itself_straddles_zero() -> None:
    rng = np.random.default_rng(3)
    groups = np.array([f"L{i // 40:02d}" for i in range(800)], dtype=np.str_)
    y = (rng.random(800) < 0.35).astype(np.int64)
    p = np.clip(y * 0.3 + rng.random(800) * 0.6, 0, 1)
    from sklearn.metrics import average_precision_score

    stat = metric_statistic(y, p, average_precision_score)
    iv = paired_bootstrap_difference(groups, stat, stat, n_resamples=200, seed=1)
    assert iv.point == pytest.approx(0.0)
    assert not iv.excludes_zero


def test_bootstrap_reports_zero_valid_resamples_rather_than_nan_silently() -> None:
    groups = np.array(["A"] * 5, dtype=np.str_)
    iv = bootstrap_metric(groups, lambda _idx: float("nan"), n_resamples=10, seed=0)
    assert iv.n_valid == 0
    assert np.isnan(iv.lo)


# =========================================================== models


def test_phoneme_independence_needs_the_confusion_block() -> None:
    spec = FeatureSpec.arm("clinical")
    m = build_matrix(spec, [("L1", "가", WordContext(), _profile(), None)], [0])
    with pytest.raises(ValueError, match="needs the 'confusion' feature block"):
        PhonemeIndependenceRisk().fit(m)


def test_phoneme_independence_returns_r_phon_unchanged() -> None:
    prof = _confusion([("각", "각")] * 10 + [("닥", "던")] * 10)
    spec = FeatureSpec.arm("confusion_only")
    words = ["각", "닥"]
    m = build_matrix(spec, [("L1", w, WordContext(), None, prof) for w in words], [0, 1])
    model = PhonemeIndependenceRisk().fit(m)
    assert np.allclose(model.predict_proba(m), [phoneme_independence_risk(w, prof) for w in words])
    assert model.describe()["family"] == "deterministic"


def test_logistic_refuses_a_single_class_training_set(cohort) -> None:
    spec = FeatureSpec.arm("clinical")
    m = cohort_matrix(cohort, spec)
    from audire.eval.ablation import _subset

    zeros = np.flatnonzero(m.y == 0)[:50].astype(np.int64)
    with pytest.raises(ValueError, match="single class"):
        LogisticRiskModel().fit(_subset(m, zeros))


def test_model_rejects_a_changed_column_set(cohort) -> None:
    m1 = cohort_matrix(cohort, FeatureSpec.arm("clinical"))
    m2 = cohort_matrix(cohort, FeatureSpec.arm("pta_only"))
    model = LogisticRiskModel().fit(m1)
    with pytest.raises(ValueError, match="feature columns changed"):
        model.predict_proba(m2)


def test_unfitted_model_refuses_to_predict(cohort) -> None:
    m = cohort_matrix(cohort, FeatureSpec.arm("clinical"))
    with pytest.raises(ValueError, match="not fitted"):
        LogisticRiskModel().predict_proba(m)


def test_logistic_coefficients_are_named_and_ordered(cohort) -> None:
    m = cohort_matrix(cohort, FeatureSpec.arm("clinical_plus_confusion"))
    model = LogisticRiskModel().fit(m)
    coefs = model.coefficients()
    assert set(coefs) == set(m.feature_names)
    values = [abs(v) for v in coefs.values()]
    assert values == sorted(values, reverse=True)
    assert "top_coefficients" in model.describe()


def test_model_registry() -> None:
    assert make_model("logistic").name == "logistic"
    assert make_model("gradient_boosting").name == "gradient_boosting"
    with pytest.raises(KeyError, match="unknown model"):
        make_model("neural_magic")


def test_all_model_predictions_are_probabilities(cohort) -> None:
    m = cohort_matrix(cohort, FeatureSpec.arm("clinical_plus_confusion"))
    for name in ("logistic", "gradient_boosting", "phoneme_independence"):
        p = make_model(name).fit(m).predict_proba(m)
        assert p.shape == (len(m),)
        assert float(p.min()) >= 0.0 and float(p.max()) <= 1.0


# =========================================================== calibration


def test_calibration_improves_a_deliberately_miscalibrated_model(cohort) -> None:
    """R_phon ranks reasonably but its values are not word-mishearing probabilities."""
    m = cohort_matrix(cohort, FeatureSpec.arm("confusion_only"))
    raw = PhonemeIndependenceRisk().fit(m).predict_proba(m)
    calibrated = CalibratedRiskModel(base=PhonemeIndependenceRisk(), method="isotonic", seed=0).fit(
        m
    )
    cal = calibrated.predict_proba(m)
    assert m.y is not None
    assert compute_metrics(m.y, cal).ece < compute_metrics(m.y, raw).ece
    assert compute_metrics(m.y, cal).brier < compute_metrics(m.y, raw).brier


def test_calibration_holds_out_whole_listeners(cohort) -> None:
    m = cohort_matrix(cohort, FeatureSpec.arm("clinical"))
    model = CalibratedRiskModel(base=LogisticRiskModel(), method="platt", holdout_fraction=0.25)
    model.fit(m)
    assert model.n_calibration_listeners >= 1
    assert model.n_calibration_listeners < int(np.unique(m.groups).size)
    described = model.describe()
    # 요청한 방법과 실제 수행된 방법을 모두 기록해야 폴백이 보입니다.
    assert described["requested_method"] == "platt"
    assert described["effective_method"] == "platt"
    assert described["fell_back"] is False
    assert described["fallback_reason"] is None


def test_calibration_none_passes_probabilities_through(cohort) -> None:
    m = cohort_matrix(cohort, FeatureSpec.arm("confusion_only"))
    plain = PhonemeIndependenceRisk().fit(m).predict_proba(m)
    passthrough = CalibratedRiskModel(base=PhonemeIndependenceRisk(), method="none").fit(m)
    assert np.allclose(passthrough.predict_proba(m), plain)


def test_calibration_needs_more_than_one_listener() -> None:
    prof = _confusion([("각", "각")] * 5)
    spec = FeatureSpec.arm("confusion_only")
    m = build_matrix(
        spec,
        [("L1", w, WordContext(), None, prof) for w in ("각", "닥", "간", "곡")],
        [0, 1, 0, 1],
    )
    with pytest.raises(ValueError, match="at least two listeners"):
        CalibratedRiskModel(base=PhonemeIndependenceRisk(), method="platt").fit(m)


def test_calibration_rejects_a_degenerate_holdout_fraction() -> None:
    with pytest.raises(ValueError, match=r"must be in \(0, 1\)"):
        CalibratedRiskModel(base=LogisticRiskModel(), holdout_fraction=0.0)


# =========================================================== end-to-end arm evaluation


def test_evaluate_arm_produces_out_of_fold_predictions_for_every_row(cohort) -> None:
    r = evaluate_arm(cohort, "clinical", "logistic", seed=1, n_splits=5, n_bootstrap=25)
    assert r.n_listeners == len(cohort)
    assert r.y_prob.size == r.n_trials
    assert np.all(np.isfinite(r.y_prob))
    assert 0.0 <= r.metrics.pr_auc <= 1.0
    assert r.metrics.pr_auc > r.prevalence_floor.pr_auc
    assert sum(t for _, t in r.fold_sizes) == r.n_trials


def test_personalized_arms_beat_the_non_personalized_floor(cohort) -> None:
    """If personalization never helped at all, the whole premise would be unsupported."""
    personal = evaluate_arm(cohort, "clinical_plus_confusion", "logistic", seed=1, n_bootstrap=25)
    floor = evaluate_arm(cohort, "word_context_only", "logistic", seed=1, n_bootstrap=25)
    assert personal.metrics.pr_auc > floor.metrics.pr_auc


def test_contrast_requires_identically_ordered_evaluation_rows(cohort) -> None:
    a = evaluate_arm(cohort, "clinical", "logistic", seed=1, n_bootstrap=25)
    b = evaluate_arm(cohort, "pta_only", "logistic", seed=1, n_bootstrap=25)
    c = contrast(a, b, metric="pr_auc", n_bootstrap=25)
    assert c.arm == "clinical" and c.reference == "pta_only"

    import dataclasses

    shuffled = dataclasses.replace(b, groups=b.groups[::-1])
    with pytest.raises(ValueError, match="identical rows in identical order"):
        contrast(a, shuffled, n_bootstrap=10)


def test_arm_result_serialises_without_the_raw_predictions(cohort) -> None:
    d = evaluate_arm(cohort, "clinical", "logistic", seed=1, n_bootstrap=10).to_dict()
    assert "y_prob" not in d and "y_true" not in d
    assert d["metrics"]["pr_auc"] >= 0.0
    assert d["prevalence_floor"]["base_rate"] > 0.0
    assert d["reliability"]["bin_centre"]


# =========================================================== scorer


def test_word_scorer_uses_the_same_feature_path_as_research(cohort) -> None:
    spec = FeatureSpec.arm("clinical_plus_confusion")
    m = cohort_matrix(cohort, spec)
    model = LogisticRiskModel().fit(m)
    record = cohort.records[0]
    scorer = WordScorer(model=model, spec=spec)

    words = ["가족", "학교", "값"]
    scores = scorer.score(
        record.listener_id,
        words,
        [WordContext(snr_db=20.0, speaker="male")] * 3,
        record.hearing,
        record.estimated_confusion,
    )
    assert scores.shape == (3,)
    assert np.all((scores >= 0) & (scores <= 1))
    assert scorer.describe()["arm"] == "clinical_plus_confusion"


def test_word_scorer_validates_input_lengths(cohort) -> None:
    spec = FeatureSpec.arm("confusion_only")
    model = LogisticRiskModel().fit(cohort_matrix(cohort, spec))
    scorer = WordScorer(model=model, spec=spec)
    r = cohort.records[0]
    with pytest.raises(ValueError, match="same length"):
        scorer.score(r.listener_id, ["가"], [], r.hearing, r.estimated_confusion)
    assert scorer.score(r.listener_id, [], [], r.hearing, r.estimated_confusion).size == 0
