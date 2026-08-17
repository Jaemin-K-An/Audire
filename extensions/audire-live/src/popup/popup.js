/**
 * 팝업. 서비스 워커에 메시지를 보내고 결과를 보여줍니다.
 *
 * 여기서 지켜지는 것
 * ------------------
 * - **`textContent` 만 씁니다.** 서버 응답에도 프로파일 별칭 같은 문자열이 들어 있고,
 *   그것을 `innerHTML` 로 넣으면 마크업이 실행됩니다.
 * - **토큰을 다루지 않습니다.** 팝업은 "연결됨/안 됨"만 압니다.
 * - **실패 사유를 구분해 보여줍니다.** "안 됨" 하나로 뭉치면 사용자가 서버를 켜야 할지,
 *   페어링을 해야 할지, 교정을 해야 할지 알 수 없습니다.
 */

import { LiveState } from '../bridge/audireClient.js';
import { DISABLED, MessageType } from '../messages.js';

/** 상태 코드를 사용자가 할 일이 있는 문장으로 옮깁니다. */
const STATE_TEXT = {
  [LiveState.OK]: '로컬 AUDIRE 에 연결되었습니다.',
  [LiveState.SERVER_OFFLINE]: '로컬 서버가 응답하지 않습니다. `make serve` 로 켜십시오.',
  [LiveState.NOT_PAIRED]: '아직 페어링되지 않았습니다. 아래에서 페어링하십시오.',
  [LiveState.UNKNOWN_PROFILE]: '선택된 프로파일을 찾을 수 없습니다.',
  [LiveState.PROFILE_NOT_READY]: '이 프로파일은 교정이 아직 없습니다.',
  [LiveState.MODEL_UNAVAILABLE]: '라이브 모델 아티팩트가 없습니다. `make live-model` 이 필요합니다.',
  [LiveState.CONTRACT_MISMATCH]: '모델과 실행 환경의 특징 계약이 다릅니다. 아티팩트를 다시 만드십시오.',
  [LiveState.INVALID_CUE]: '자막을 채점할 수 없는 형태입니다.',
  [LiveState.ORIGIN_NOT_ALLOWED]:
    '서버가 이 출처를 거절했습니다. 라이브 API 는 AUDIRE 확장에서만 부를 수 있습니다.',
  [DISABLED]: '꺼져 있습니다.',
};

/** @param {string} state */
function describe(state) {
  return STATE_TEXT[state] ?? '알 수 없는 오류입니다.';
}

/** @param {{type: string, payload?: any}} message */
function send(message) {
  return chrome.runtime.sendMessage(message);
}

/** @param {string} id */
const $ = (id) => /** @type {HTMLElement} */ (document.getElementById(id));

/** @param {string} id @param {string} text */
function setText(id, text) {
  $(id).textContent = text;
}

async function refresh() {
  const state = await send({ type: MessageType.GET_STATE });
  setText('server-state', describe(state.state));
  setText(
    'pairing-state',
    state.hasToken ? '이 기기는 페어링되어 있습니다.' : '페어링이 없습니다.',
  );

  const settings = state.settings ?? {};
  /** @type {HTMLInputElement} */ ($('enabled')).checked = Boolean(settings.enabled);
  const percent = Math.round((settings.targetCaptionRate ?? 0.2) * 100);
  /** @type {HTMLInputElement} */ ($('rate')).value = String(percent);
  setText('rate-value', `${percent}%`);

  const model = state.server?.model;
  setText(
    'model-state',
    model
      ? `모델 ${model.family} · 계약 ${model.input_contract} · 임계값 ${model.threshold_policy}`
      : '모델 정보 없음',
  );

  await refreshProfiles(settings.profileId ?? null);
}

/** @param {string|null} selected */
async function refreshProfiles(selected) {
  const select = /** @type {HTMLSelectElement} */ ($('profile-select'));
  select.replaceChildren();

  const result = await send({ type: MessageType.LIST_PROFILES });
  if (!result.ok) {
    setText('profile-state', describe(result.state));
    return;
  }
  const profiles = result.profiles ?? [];
  if (profiles.length === 0) {
    setText('profile-state', '프로파일이 없습니다. 먼저 교정을 진행하십시오.');
    return;
  }

  for (const profile of profiles) {
    const option = document.createElement('option');
    option.value = profile.id;
    // 별칭은 서버가 준 문자열입니다. 텍스트로만 넣습니다.
    option.textContent = profile.ready
      ? `${profile.alias} (교정 ${profile.calibration_trials}회)`
      : `${profile.alias} — 교정 필요`;
    option.disabled = !profile.ready;
    if (profile.id === selected) option.selected = true;
    select.append(option);
  }
  const ready = profiles.filter((p) => p.ready).length;
  setText('profile-state', `${profiles.length}개 중 ${ready}개가 채점 가능합니다.`);
}

/** @param {Partial<{profileId: string|null, targetCaptionRate: number, enabled: boolean}>} patch */
async function updateSettings(patch) {
  const result = await send({ type: MessageType.UPDATE_SETTINGS, payload: patch });
  if (!result.ok) setText('server-state', result.error ?? '설정을 저장하지 못했습니다.');
  return result;
}

function wire() {
  $('pair').addEventListener('click', async () => {
    setText('pairing-state', '페어링 중…');
    const result = await send({ type: MessageType.PAIR });
    setText(
      'pairing-state',
      result.ok ? '페어링되었습니다.' : `페어링 실패: ${describe(result.state)}`,
    );
    await refresh();
  });

  $('unpair').addEventListener('click', async () => {
    const result = await send({ type: MessageType.UNPAIR });
    await refresh();
    if (!result.revokedOnServer) {
      // 로컬 토큰은 지워졌지만 서버에는 페어링이 남아 있을 수 있습니다. 끊겼다고
      // 잘못 믿게 두지 않습니다.
      setText(
        'pairing-state',
        `이 브라우저의 토큰은 지웠습니다. 다만 서버에 닿지 못해(${describe(
          result.serverState,
        )}) 서버 쪽 페어링은 남아 있을 수 있습니다. 서버를 켠 뒤 다시 눌러 주십시오.`,
      );
    }
  });

  $('profile-select').addEventListener('change', (event) => {
    updateSettings({ profileId: /** @type {HTMLSelectElement} */ (event.target).value });
  });

  $('rate').addEventListener('input', (event) => {
    const percent = Number(/** @type {HTMLInputElement} */ (event.target).value);
    setText('rate-value', `${percent}%`);
  });
  $('rate').addEventListener('change', (event) => {
    const percent = Number(/** @type {HTMLInputElement} */ (event.target).value);
    updateSettings({ targetCaptionRate: percent / 100 });
  });

  $('enabled').addEventListener('change', (event) => {
    updateSettings({ enabled: /** @type {HTMLInputElement} */ (event.target).checked });
  });
}

if (typeof document !== 'undefined') {
  wire();
  refresh().catch((error) => setText('server-state', String(error)));
}

export { describe, STATE_TEXT };
