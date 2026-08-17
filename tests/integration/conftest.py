"""라이브 API 시험이 공유하는 픽스처.

`test_live_api.py`(TestClient 로 계약을 고정)와 `test_live_bridge.py`(진짜 소켓 위에서
확장의 실제 JS 클라이언트를 태움)가 같은 앱을 세워야 합니다. 둘이 따로 조립하면 한쪽만
고쳐지고, 그러면 "계약은 맞는데 실물은 안 되는" 상태가 생깁니다.

`_private` 를 autouse 로 두지 않았습니다. 이 디렉터리의 다른 시험들까지 격리 환경으로
끌고 들어가는 것은 이 변경의 범위가 아닙니다. 대신 `store` 와 `live_scorer` 가 이것을
의존하므로, 라이브 경로를 쓰는 시험은 자동으로 격리됩니다.
"""

from __future__ import annotations

import pytest

from audire.confusion import CalibrationTrial, ConfusionProfile
from audire.live.service import LiveScorer
from audire.profile import ProfileStore
from audire.profile.schema import (
    Audiogram,
    AudiogramPoint,
    Ear,
    EarProfile,
    HearingProfile,
    ProfileSource,
    SpeechScores,
)

LISTENER = "L001"


@pytest.fixture
def _private(tmp_path, monkeypatch):
    monkeypatch.setenv("AUDIRE_PRIVATE_DIR", str(tmp_path / "private"))
    yield


@pytest.fixture
def store(tmp_path, _private) -> ProfileStore:
    store = ProfileStore(tmp_path / "profiles")
    ear = EarProfile(
        ear=Ear.RIGHT,
        audiogram=Audiogram(
            ear=Ear.RIGHT,
            thresholds={f: AudiogramPoint(db_hl=55.0) for f in (500, 1000, 2000, 4000)},
        ),
        speech=SpeechScores(
            ear=Ear.RIGHT, srt_db_hl=55.0, wrs_percent=62.0, wrs_presentation_level_db_hl=85.0
        ),
    )
    store.save_hearing(
        HearingProfile(
            listener_id=LISTENER, source=ProfileSource.MANUAL, is_synthetic=False, right=ear
        )
    )
    store.save_confusion(
        ConfusionProfile.from_trials(
            LISTENER,
            [
                CalibrationTrial(stimulus_id=f"s{i}", target="각", response="각" if i % 3 else "닥")
                for i in range(60)
            ],
            is_synthetic=False,
        )
    )
    return store


@pytest.fixture
def live_scorer(_private):
    """작은 라이브 아티팩트를 실제로 적합해 계약 경로를 그대로 통과시킵니다."""
    import yaml

    from audire.config.paths import private_dir
    from audire.risk.artifact import fit_live_artifact

    config = private_dir() / "live.yaml"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(
        yaml.safe_dump(
            {
                "name": "live_api_test",
                "simulation": {
                    "name": "c",
                    "seeds": [1],
                    "n_listeners": 10,
                    "n_calibration_trials": 20,
                    "n_word_trials": 20,
                },
                "arms": ["live_word_context_clinical_confusion"],
                "models": ["logistic"],
                "n_splits": 3,
                "n_bootstrap": 0,
                "contrasts": [],
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    artifact = fit_live_artifact(config)
    metadata = {**artifact.metadata, "artifact_sha256": "test-digest"}
    return LiveScorer(scorer=artifact.scorer, artifact_metadata=metadata)
