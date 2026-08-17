"""G5 — CPU-only end-to-end pipeline smoke test.

Runs the complete production path — transcript → personalized word risk → caption policy
→ SRT/ASS/JSON — without downloading model weights, by replaying a transcript that was
recorded from a real backend. The replay backend performs no inference and invents
nothing; it is the ASR-cache equivalent, and it preserves the original backend's identity
in provenance so a replayed result can never claim to be a fresh recognition.

The real ``faster-whisper`` backend is exercised separately by tests marked ``asr``, which
require the weights and are excluded from the default run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from audire.asr import (
    ASRUnavailable,
    FasterWhisperBackend,
    IncompleteProfile,
    ReplayBackend,
    Token,
    Transcript,
    caption_media,
    check_ready,
    save_transcript,
    score_transcript,
)
from audire.caption import (
    BudgetPolicy,
    CaptionDecision,
    FullCaptionPolicy,
    ThresholdPolicy,
    to_ass,
    to_json,
    to_srt,
)
from audire.eval.ablation import cohort_matrix
from audire.risk import FeatureSpec, LogisticRiskModel, WordScorer
from audire.sim import SimulationConfig, build_cohort

CFG = SimulationConfig(
    name="pipeline-smoke", n_listeners=16, n_calibration_trials=60, n_word_trials=60, seeds=[31]
)

#: A short Korean utterance with realistic word timings and per-word confidences.
RECORDED = {
    "backend": "faster-whisper",
    "model_id": "small",
    "language": "ko",
    "language_probability": 0.99,
    "duration_s": 4.2,
    "provenance": {"media": "sample_ko.wav", "recorded_for": "AUDIRE integration fixture"},
    "tokens": [
        {"text": "오늘", "start_s": 0.10, "end_s": 0.52, "confidence": 0.97},
        {"text": "날씨가", "start_s": 0.58, "end_s": 1.14, "confidence": 0.93},
        {"text": "정말", "start_s": 1.22, "end_s": 1.60, "confidence": 0.41},
        {"text": "좋아서", "start_s": 1.66, "end_s": 2.24, "confidence": 0.95},
        {"text": "산책을", "start_s": 2.40, "end_s": 2.98, "confidence": 0.88},
        {"text": "했습니다", "start_s": 3.04, "end_s": 3.70, "confidence": 0.96},
        {"text": "2024", "start_s": 3.80, "end_s": 4.10, "confidence": 0.60},
    ],
}


@pytest.fixture(scope="module")
def fitted():
    """A cohort, a fitted scorer and one listener's profiles."""
    cohort = build_cohort(CFG, 31)
    spec = FeatureSpec.arm("clinical_plus_confusion", speakers=("male", "female", "unknown"))
    model = LogisticRiskModel().fit(cohort_matrix(cohort, spec))
    record = cohort.records[0]
    return WordScorer(model=model, spec=spec), record


@pytest.fixture
def media(tmp_path: Path) -> Path:
    p = tmp_path / "sample_ko.wav"
    p.write_bytes(b"RIFF")  # the replay backend never reads the bytes
    return p


@pytest.fixture
def backend(tmp_path: Path) -> ReplayBackend:
    path = tmp_path / "recorded.json"
    path.write_text(json.dumps(RECORDED, ensure_ascii=False), encoding="utf-8")
    return ReplayBackend(path)


# =========================================================== end to end


def test_full_pipeline_produces_captions_and_all_three_exports(fitted, media, backend) -> None:
    scorer, record = fitted
    result = caption_media(
        media,
        backend,
        scorer,
        listener_id=record.listener_id,
        hearing=record.hearing,
        confusion=record.estimated_confusion,
        policy=BudgetPolicy(budget=0.4),
    )

    assert len(result.words) == len(RECORDED["tokens"])
    assert 0.0 < result.caption_ratio <= 1.0
    assert result.n_shown == round(0.4 * len(result.words))

    srt = to_srt(result.words)
    ass = to_ass(result.words)
    payload = json.loads(
        to_json(
            result.words,
            listener_id=result.listener_id,
            policy=result.policy,
            provenance=result.provenance,
        )
    )
    assert srt.strip() and "-->" in srt
    assert "[Script Info]" in ass and "Dialogue:" in ass
    assert payload["n_words"] == len(result.words)
    assert payload["provenance"]["asr"]["backend"] == "replay"


def test_every_word_carries_risk_and_asr_confidence_separately(fitted, media, backend) -> None:
    scorer, record = fitted
    result = caption_media(
        media,
        backend,
        scorer,
        listener_id=record.listener_id,
        hearing=record.hearing,
        confusion=record.estimated_confusion,
    )
    by_text = {w.text: w for w in result.words}
    low_conf = by_text["정말"]
    assert low_conf.asr_confidence == pytest.approx(0.41)
    # The recogniser's uncertainty must not have leaked into the listener risk.
    assert low_conf.listener_risk != low_conf.asr_confidence
    assert 0.0 <= low_conf.listener_risk <= 1.0
    assert low_conf.explanation()["asr_confidence"] == pytest.approx(0.41)


def test_non_hangul_token_is_kept_but_flagged_unscoreable(fitted, media, backend) -> None:
    """A Korean confusion profile cannot score '2024'; dropping it silently would be worse."""
    scorer, record = fitted
    result = caption_media(
        media,
        backend,
        scorer,
        listener_id=record.listener_id,
        hearing=record.hearing,
        confusion=record.estimated_confusion,
    )
    numeral = next(w for w in result.words if w.text == "2024")
    assert numeral.meta["scoreable"] is False
    assert numeral.meta["reason_not_scored"] == "no Hangul syllables"
    assert numeral.listener_risk == 0.0
    assert numeral.contributions == ()


def test_explanations_name_the_phonemes_and_their_evidence(fitted, media, backend) -> None:
    scorer, record = fitted
    result = caption_media(
        media,
        backend,
        scorer,
        listener_id=record.listener_id,
        hearing=record.hearing,
        confusion=record.estimated_confusion,
    )
    word = next(w for w in result.words if w.text == "산책을")
    exp = word.explanation()
    assert exp["weakest_phonemes"]
    first = exp["weakest_phonemes"][0]
    assert first["position"] in {"onset", "nucleus", "coda"}
    assert isinstance(first["n_calibration_observations"], int)
    assert exp["model_arm"] == "clinical_plus_confusion"


def test_the_three_caption_modes_show_different_amounts(fitted, media, backend) -> None:
    scorer, record = fitted
    kwargs = {
        "listener_id": record.listener_id,
        "hearing": record.hearing,
        "confusion": record.estimated_confusion,
    }
    full = caption_media(media, backend, scorer, policy=FullCaptionPolicy(), **kwargs)
    budget = caption_media(media, backend, scorer, policy=BudgetPolicy(budget=0.3), **kwargs)
    threshold = caption_media(media, backend, scorer, policy=ThresholdPolicy(tau=0.99), **kwargs)

    assert full.caption_ratio == 1.0
    assert budget.caption_ratio < full.caption_ratio
    assert threshold.caption_ratio <= budget.caption_ratio
    assert full.policy["policy"] == "full"


def test_asr_confidence_floor_surfaces_the_uncertain_token_distinctly(
    fitted, media, backend
) -> None:
    scorer, record = fitted
    result = caption_media(
        media,
        backend,
        scorer,
        listener_id=record.listener_id,
        hearing=record.hearing,
        confusion=record.estimated_confusion,
        policy=ThresholdPolicy(tau=0.995, asr_confidence_floor=0.5),
    )
    shown = [w for w in result.words if w.is_shown]
    assert shown, "the low-confidence tokens should have been surfaced"
    assert all(w.decision is CaptionDecision.SHOWN_LOW_ASR_CONFIDENCE for w in shown)
    # Those must be distinguishable from personalization hits when the study is scored.
    assert not any(w.decision is CaptionDecision.SHOWN_HIGH_RISK for w in shown)


def test_pipeline_is_deterministic(fitted, media, backend) -> None:
    scorer, record = fitted
    kwargs = {
        "listener_id": record.listener_id,
        "hearing": record.hearing,
        "confusion": record.estimated_confusion,
        "policy": BudgetPolicy(budget=0.4),
    }
    a = caption_media(media, backend, scorer, **kwargs)
    b = caption_media(media, backend, scorer, **kwargs)
    assert [w.listener_risk for w in a.words] == [w.listener_risk for w in b.words]
    assert to_srt(a.words) == to_srt(b.words)


def test_result_summary_reports_asr_and_listener_provenance(fitted, media, backend) -> None:
    scorer, record = fitted
    summary = caption_media(
        media,
        backend,
        scorer,
        listener_id=record.listener_id,
        hearing=record.hearing,
        confusion=record.estimated_confusion,
    ).summary()
    assert summary["asr"]["backend"] == "faster-whisper"  # the ORIGINAL recogniser
    assert summary["asr"]["n_scoreable_tokens"] == 6  # '2024' is not scoreable
    assert summary["provenance"]["listener"]["calibration_trials"] == CFG.n_calibration_trials
    assert summary["provenance"]["listener"]["is_synthetic"] is True


# =========================================================== failure paths


def test_missing_media_file_is_reported(fitted, backend) -> None:
    scorer, record = fitted
    with pytest.raises(ValueError, match="recorded transcript was produced from"):
        caption_media(
            Path("/nonexistent/other.wav"),
            backend,
            scorer,
            listener_id=record.listener_id,
            hearing=record.hearing,
            confusion=record.estimated_confusion,
        )


def test_missing_confusion_profile_fails_loudly(fitted, media, backend) -> None:
    scorer, record = fitted
    with pytest.raises(IncompleteProfile, match="run a calibration first"):
        caption_media(
            media,
            backend,
            scorer,
            listener_id=record.listener_id,
            hearing=record.hearing,
            confusion=None,
        )


def test_too_little_calibration_fails_loudly(fitted, media, backend) -> None:
    """A plausible-looking score from three trials would be worse than an error."""
    from audire.confusion import CalibrationTrial, ConfusionProfile

    scorer, record = fitted
    thin = ConfusionProfile.from_trials(
        "thin",
        [CalibrationTrial(stimulus_id=f"s{i}", target="각", response="각") for i in range(3)],
        is_synthetic=False,
    )
    problems = check_ready(scorer, record.hearing, thin)
    assert any("only 3 calibration trials" in p for p in problems)
    with pytest.raises(IncompleteProfile, match="only 3 calibration trials"):
        caption_media(
            media,
            backend,
            scorer,
            listener_id="thin",
            hearing=record.hearing,
            confusion=thin,
        )


def test_unfitted_model_is_refused(fitted, media, backend) -> None:
    _, record = fitted
    spec = FeatureSpec.arm("clinical_plus_confusion")
    unfitted = WordScorer(model=LogisticRiskModel(), spec=spec)
    problems = check_ready(unfitted, record.hearing, record.estimated_confusion)
    assert "the risk model is not fitted" in problems


def test_replay_refuses_a_mismatched_media_file(tmp_path: Path, backend) -> None:
    other = tmp_path / "different.wav"
    other.write_bytes(b"RIFF")
    with pytest.raises(ValueError, match="Pass allow_media_mismatch=True"):
        backend.transcribe(other)

    permissive = ReplayBackend(backend.transcript_path, allow_media_mismatch=True)
    assert len(permissive.transcribe(other)) == len(RECORDED["tokens"])


def test_replay_preserves_the_original_backend_identity(backend, media) -> None:
    t = backend.transcribe(media)
    assert t.backend == "faster-whisper"
    assert t.provenance["replayed"] is True
    assert "replayed_from" in t.provenance
    assert backend.describe()["note"].startswith("replays a recorded transcript")


def test_missing_recorded_transcript_is_reported(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="recorded transcript not found"):
        ReplayBackend(tmp_path / "nope.json")


# =========================================================== transcript contract


def test_transcript_detects_overlapping_and_out_of_order_tokens() -> None:
    bad = Transcript(
        tokens=(
            Token("가", 0.0, 1.0, 0.9),
            Token("나", 0.5, 1.5, 0.9),  # overlaps
            Token("다", 0.2, 0.8, 0.9),  # goes backwards
        ),
        language="ko",
        language_probability=0.9,
        duration_s=2.0,
        backend="test",
        model_id="test",
    )
    problems = bad.timing_problems()
    assert any("overlapping token" in p for p in problems)
    assert any("before token" in p for p in problems)
    # 진단 문자열은 로그로 나가므로 전사 내용을 담아서는 안 됩니다.
    assert not any(t.text in p for p in problems for t in bad.tokens), problems


def test_token_validates_timing_and_confidence() -> None:
    with pytest.raises(ValueError, match=r"ends .* before it starts"):
        Token("가", 1.0, 0.5)
    with pytest.raises(ValueError, match="negative start time"):
        Token("가", -0.1, 0.5)
    with pytest.raises(ValueError, match="confidence must be in"):
        Token("가", 0.0, 0.5, 1.4)


def test_token_hangul_extraction() -> None:
    assert Token("좋아서,", 0.0, 0.5).hangul_text == "좋아서"
    assert Token("2024", 0.0, 0.5).has_hangul is False
    assert Token("A가B", 0.0, 0.5).hangul_text == "가"


def test_transcript_roundtrips_through_disk(tmp_path: Path, backend, media) -> None:
    original = backend.transcribe(media)
    path = save_transcript(original, tmp_path / "out.json")
    restored = ReplayBackend(path, allow_media_mismatch=True).transcribe(media)
    assert [t.text for t in restored.tokens] == [t.text for t in original.tokens]
    assert [t.confidence for t in restored.tokens] == [t.confidence for t in original.tokens]


def test_score_transcript_without_a_policy_leaves_decisions_unapplied(
    fitted, backend, media
) -> None:
    scorer, record = fitted
    words = score_transcript(
        backend.transcribe(media),
        scorer,
        listener_id=record.listener_id,
        hearing=record.hearing,
        confusion=record.estimated_confusion,
    )
    assert all(w.policy == "unapplied" for w in words)
    assert all(w.decision is CaptionDecision.HIDDEN for w in words)


# =========================================================== real backend


def test_default_asr_model_revision_is_pinned_and_forwarded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A moving Hugging Face branch must never determine production weights."""
    import sys
    from types import SimpleNamespace

    from audire.asr.whisper_backend import DEFAULT_MODEL_REVISION

    received: dict[str, object] = {}

    def fake_model(model_id: str, **kwargs: object) -> object:
        received.update(model_id=model_id, **kwargs)
        return object()

    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=fake_model))
    backend = FasterWhisperBackend(download_root=tmp_path)

    backend._load()

    assert len(DEFAULT_MODEL_REVISION) == 40
    assert received["revision"] == DEFAULT_MODEL_REVISION
    assert backend.describe()["model_revision"] == DEFAULT_MODEL_REVISION


def _require_real_asr() -> None:
    """Keep model downloads and external-data tests explicit in the default suite."""
    import os

    if os.environ.get("AUDIRE_RUN_REAL_ASR") != "1":
        pytest.skip("set AUDIRE_RUN_REAL_ASR=1 to run real model inference")


@pytest.fixture(scope="module")
def real_korean_wav(tmp_path_factory: pytest.TempPathFactory) -> Path:
    _require_real_asr()
    sf = pytest.importorskip("soundfile")
    from audire.data.zeroth import load_zeroth_utterances

    try:
        utterance = load_zeroth_utterances(limit=1, with_audio=True)[0]
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
    assert utterance.audio is not None
    path = tmp_path_factory.mktemp("real-asr") / "zeroth_ko.wav"
    sf.write(path, utterance.audio, utterance.sample_rate)
    return path


@pytest.fixture(scope="module")
def real_backend() -> FasterWhisperBackend:
    _require_real_asr()
    return FasterWhisperBackend()


@pytest.mark.asr
@pytest.mark.slow
def test_faster_whisper_backend_reports_availability_honestly(media: Path) -> None:
    """Without the extra installed, the backend must say so with an actionable message."""
    b = FasterWhisperBackend()
    # A missing media file is reported before any model work is attempted.
    with pytest.raises(FileNotFoundError, match="media file not found"):
        b.transcribe(Path("/nonexistent/anything.wav"))
    if not b.is_available():
        with pytest.raises(ASRUnavailable, match="make bootstrap-asr"):
            b.transcribe(media)
        pytest.skip("faster-whisper is not installed; the availability path was verified")
    assert b.describe()["backend"] == "faster-whisper"
    assert b.describe()["decode_options"]["temperature"] == 0.0


@pytest.mark.asr
@pytest.mark.data
@pytest.mark.slow
def test_faster_whisper_transcribes_pinned_korean_audio_with_word_timestamps(
    real_backend: FasterWhisperBackend, real_korean_wav: Path
) -> None:
    """Exercise installed code, downloaded weights and pinned Korean audio together."""
    from audire.asr.whisper_backend import DEFAULT_MODEL_REVISION

    transcript = real_backend.transcribe(real_korean_wav)

    assert transcript.backend == "faster-whisper"
    assert transcript.model_id == "small"
    assert transcript.language == "ko"
    assert transcript.duration_s > 0
    assert transcript.tokens
    assert transcript.hangul_tokens
    assert transcript.timing_problems() == []
    assert all(token.end_s >= token.start_s >= 0 for token in transcript.tokens)
    assert all(
        token.confidence is None or 0 <= token.confidence <= 1 for token in transcript.tokens
    )
    assert transcript.provenance["library_version"] == "1.2.1"
    assert transcript.provenance["model_revision"] == DEFAULT_MODEL_REVISION


@pytest.mark.asr
@pytest.mark.data
@pytest.mark.slow
@pytest.mark.parametrize(("suffix", "codec"), [("mp3", "libmp3lame"), ("mp4", "aac")])
def test_faster_whisper_accepts_compressed_media(
    suffix: str,
    codec: str,
    real_backend: FasterWhisperBackend,
    real_korean_wav: Path,
    tmp_path: Path,
) -> None:
    """The production backend must decode both advertised upload formats."""
    import shutil
    import subprocess

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg is required for compressed-media validation")
    encoded = tmp_path / f"zeroth_ko.{suffix}"
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(real_korean_wav),
            "-codec:a",
            codec,
            str(encoded),
        ],
        check=True,
    )

    transcript = real_backend.transcribe(encoded)

    assert transcript.tokens
    assert transcript.hangul_tokens
    assert transcript.provenance["media"] == encoded.name


def test_non_finite_timings_are_rejected_at_construction() -> None:
    """회귀 테스트.

    나머지 검사는 전부 부등식이고 NaN 과의 부등식은 항상 False 이므로, NaN 타임스탬프가
    모든 검사를 조용히 통과했습니다. 그 값은 `timing_problems`(역시 부등식이라 아무 일도
    하지 않음)를 지나 자막 타임라인과 SRT/VTT 내보내기까지 흘러가 깨진 타임코드가 됩니다.
    """
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="non-finite"):
            Token("가", bad, 1.0)
        with pytest.raises(ValueError, match="non-finite"):
            Token("가", 0.0, bad)


def test_non_finite_transcript_duration_is_rejected() -> None:
    """NaN 길이는 '미디어 길이를 넘었는가' 검사를 통째로 무력화합니다."""
    with pytest.raises(ValueError, match="finite and non-negative"):
        Transcript(
            tokens=(),
            language="ko",
            language_probability=0.9,
            duration_s=float("nan"),
            backend="test",
            model_id="test",
        )


def test_non_finite_language_probability_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"finite value in \[0, 1\]"):
        Transcript(
            tokens=(),
            language="ko",
            language_probability=float("nan"),
            duration_s=1.0,
            backend="test",
            model_id="test",
        )
