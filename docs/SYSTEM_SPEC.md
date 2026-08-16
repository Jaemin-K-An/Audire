# AUDIRE 시스템 명세

아키텍처와 데이터 계약. 인터페이스가 정의되었으나 아직 구현되지 않은 곳은 명시합니다 —
게이트 상태는 `docs/TASKS.md` 참조.

---

## 1. 파이프라인

```
                      ┌─────────────────────────────────────────┐
   오디오 / 비디오 ──► │ ASR 어댑터                              │
                      │  → 토큰 + 단어 타임스탬프 + 신뢰도       │
                      └───────────────┬─────────────────────────┘
                                      │ 텍스트 토큰
                                      ▼
  ┌────────────────┐          ┌───────────────────┐
  │ HearingProfile │────┐     │ audire.hangul     │
  │ PTA/SRT/WRS/…  │    │     │ 초성/중성/종성    │
  └────────────────┘    │     └─────────┬─────────┘
                        ▼               │
  ┌────────────────┐  ┌─────────────────▼─────────┐   ┌──────────────────┐
  │ 교정 응답      │─►│ audire.risk.features      │──►│ RiskModel        │
  │                │  │  word · context · pta ·   │   │ 로지스틱 / R_phon│
  └───────┬────────┘  │  clinical · confusion     │   │ / 부스팅         │
          ▼           └───────────────────────────┘   └────────┬─────────┘
  ┌────────────────┐                                            │ P(오청)
  │ ConfusionProfile│                                           ▼
  │ 위치별 C_u      │                              ┌────────────────────────┐
  └────────────────┘                              │ WordRisk               │
                                                   │  listener_risk         │
                                                   │  asr_confidence (별개) │
                                                   └───────────┬────────────┘
                                                               ▼
                                            ┌──────────────────────────────┐
                                            │ CaptionPolicy                │
                                            │  전체 / 임계값 / 예산        │
                                            └──────────┬───────────────────┘
                                                       ▼
                                              SRT · ASS · JSON
```

두 개인화 입력은 **의도적으로 별개 객체**입니다. `HearingProfile`은 전역 어음인지 상을,
`ConfusionProfile`은 국소 오류 구조를 담습니다. RQ1이 곧 "후자가 전자를 넘어 정보를
더하는가"이므로, 둘을 합치면 질문 자체가 성립하지 않습니다(ADR-0008).

---

## 2. 모듈 계약

| 모듈 | 책임 | 핵심 불변식 |
|---|---|---|
| `audire.hangul` | 자모 분해/조합, 자모 목록 | U+AC00–U+D7A3 전역에서 전사(total)·정확. "받침 없음"은 명시적 범주 |
| `audire.confusion` | 응답 파싱, 오류 분류, `C_u` | 빈도와 확률을 항상 함께 보유. 어떤 관측도 버리지 않음 |
| `audire.profile` | 임상 스키마, 파생 지표, 비공개 저장 | 결측은 명시적. 모든 파생값이 계산 방법을 명시 |
| `audire.risk` | 특징, 모델, 보정 | arm은 청취자 표현만 다름. 대치는 파이프라인 내부 |
| `audire.caption` | 정책, `WordRisk`, 내보내기 | ASR 신뢰도는 절대 `listener_risk`에 들어가지 않음 |
| `audire.sim` | 합성 청취자·시행 | 전 구간 `is_synthetic=True`. 생성기 ≠ 채점 모델 |
| `audire.eval` | 지표, 분할, 부트스트랩, 연구 | 청취자 수준 분할만. 폴드마다 누출 단언 |
| `audire.data` | 출처 레지스트리, 취득, 매니페스트, 자극 | 등록되지 않은 것은 내려받지 않음. 파일별 SHA-256 |
| `audire.experiments` | 설정, 출처 레지스트리, 그림 | 보고된 모든 수치가 실행 기록으로 추적됨 |
| `audire.api` | 로컬 프로필·교정·미디어 처리 API | 모델 부재/손상 시 실패; 업로드는 처리 후 삭제 |

---

## 3. 핵심 데이터 계약

### 3.1 `HearingProfile`

필수: `listener_id`(불투명 식별자, 절대 이름 아님), `source` ∈ {manual, clinical_export,
synthetic}, `is_synthetic`, 최소 한쪽 귀.

귀별: 주파수별 기도 역치(각각 `db_hl | None`, `no_response`, `masked`를 갖는
`AudiogramPoint`), `SpeechScores`(SRT, WRS + 제시 강도 및 어표), 선택적 `PIFunction`과
`LoudnessLevels`.

검증기가 강제하는 불변식:

* 제시 강도 없는 WRS는 거부됩니다 — 해석 불가능하기 때문입니다.
* `no_response=True`는 무반응이 확인된 강도(청력계 출력 한계)를 요구하며, 그 점은
  평균에 포함되지 않고 **제외**됩니다.
* 필수 주파수가 하나라도 없으면 PTA는 부분 평균이 아니라 `None`을 반환합니다.
* `source=synthetic`과 `is_synthetic`은 반드시 일치해야 합니다.
* UCL은 MCL보다 낮을 수 없습니다.

### 3.2 `ConfusionProfile`

`ConfusionMatrix` 3개(초성·중성·종성). 각각 직사각형입니다. 행은 목표 범주, 열은
목표 범주 + `NO_RESPONSE`.

* `counts`(정수)와 `probabilities()`(평활)는 분리되며, 어떤 확률 옆에도
  `n_observations`를 항상 조회할 수 있습니다.
* 평활은 명시적 사전분포(`SmoothingSpec`)를 갖는 디리클레 사후평균입니다.
* 미관측 행은 정확히 사전분포와 같으며 `unobserved_targets`가 이름으로 보고합니다.
* `coverage`가 각 자모 목록 중 근거가 있는 비율을 보고합니다.
* `pool_profiles`는 합성과 비합성 청취자의 혼합을 거부합니다.

### 3.3 `WordRisk`

`text`, `start_s`, `end_s`, `listener_risk`, `asr_confidence`(`None` 가능),
`model_version`, `model_arm`, `decision`, `policy`, `contributions`, `meta`.

`decision`은 `shown_high_risk`와 `shown_low_asr_confidence`를 구별합니다.
따라서 인식기가 확신하지 못해 표시된 단어가 **개인화 적중으로 계산될 수 없습니다.**

---

## 4. 프라이버시 계약

* 실제 프로파일과 원시 교정 응답은 `private/` 아래에만 기록되며, git 무시 대상이고
  CI의 `scripts/check_repo_hygiene.py`가 강제합니다.
* 청취자 id는 `[A-Za-z0-9._-]`로 제한됩니다 — 검증기가 이름을 거부합니다.
* `ProfileStore.export()`는 한 청취자에 대해 저장된 전부를 반환합니다.
  `delete()`는 되돌릴 수 없게 제거하고 삭제할 것이 없으면 에러를 냅니다 —
  실패한 삭제 요청이 보이도록 하기 위함입니다.
* 저장소에는 스키마와 합성 예시만 존재합니다.

---

## 5. 재현성 계약

모든 실행이 `experiments/registry.yaml`에 추가합니다: `run_id`, git SHA,
**git 더티 플래그**, 의존성 락 해시, Python 버전, 플랫폼, 시드 목록, 전체 설정,
데이터 매니페스트 콘텐츠 다이제스트, 아티팩트 경로, 지표, 상태.
**실패한 실행도 기록**됩니다.

`audire figures`는 살아 있는 모델이 아니라 `summary.json`만으로 모든 표와 그림을
재생성합니다. 따라서 그림이 그것을 만든 실행과 어긋날 수 없습니다.

배포 scorer는 `make model`이 사전등록 설정의 **모든 5개 합성 시드**(독립 청취자 실현
400개, 단어 시행 100,000개)에 최종 선택 로지스틱 계열을 다시 적합해 `private/models/`에
생성합니다. 모델과 JSON sidecar는 설정 해시, 시드, git SHA/더티 상태, 락 해시, 데이터
매니페스트, 합성 학습 경고와 모델 체크섬을 보존합니다. 서버는 체크섬·스키마·모델 버전·적합
상태를 통과한 로컬 아티팩트만 자동 로드합니다.

---

## 6. 사용자 시스템과 남은 검증

`apps/api`와 `apps/web`은 프로필 생성, 내장 자극 교정, WAV/MP3/MP4/M4A/FLAC/OGG 업로드,
전체/임계값/예산 자막 정책, 단어별 위험·ASR 신뢰도 분리 표시, SRT/ASS/JSON 다운로드를
제공합니다. API 계약과 실제 Chromium 정상·실패 경로가 같은 생산 파이프라인을 실행합니다.

`audire.asr`의 `FasterWhisperBackend`는 고정된 `small` 모델 리비전으로 실제 Zeroth-Korean
WAV·MP3·MP4를 CPU int8에서 전사했습니다(ADR-0013). 기본 CI 종단 스모크는 대형 가중치를
매번 내려받지 않도록 녹화된 전사를 재생하는 리플레이 백엔드로 계속 동작합니다(추론 없음,
원 인식기 정체성 보존). 실제 WER·CER·타임스탬프의 10화자 고정 회귀는 §7 범위로
실행됐으며 457개 전체 평가는 아직 미실행입니다.

10화자 고정 ASR 회귀는 실제 가중치에서 WER/CER·타임스탬프·RTF를 기록했습니다. 아직
미실행인 검증은 457개 전체 또는 더 큰 사전 정의 평가집, E5 전체 교정 전략 스윕,
SNR·하위군 분석, 새 환경 인수, 깨끗한 SHA 재현입니다. 사람 효능과 전문가 검토는 외부
차단 요인이며 G6 소프트웨어 완료가 이를 대체하지 않습니다.
