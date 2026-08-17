/**
 * 확장 내부 메시지 어휘.
 *
 * 이것이 `background.js` 와 분리된 이유는 실제 결함을 막기 위해서입니다. 팝업이
 * `background.js` 를 import 하면 그 파일 맨 아래의 `chrome.runtime.onMessage` 등록이
 * **팝업 안에서도** 실행되어, 확장 안에 메시지 처리기가 둘 생깁니다. 어휘는 부작용이
 * 없는 이 파일에 두고, 처리기는 서비스 워커에만 둡니다.
 */

export const MessageType = Object.freeze({
  GET_STATE: 'audire/get-state',
  PAIR: 'audire/pair',
  UNPAIR: 'audire/unpair',
  LIST_PROFILES: 'audire/list-profiles',
  UPDATE_SETTINGS: 'audire/update-settings',
  SCORE_CUE: 'audire/score-cue',
  /**
   * 서비스 워커 → 콘텐츠 스크립트 방송. 설정이 바뀌었으니 다시 물어보라는 뜻입니다.
   *
   * 콘텐츠 스크립트가 `chrome.storage.onChanged` 를 직접 듣지 않는 이유가 여기 있습니다.
   * 그 이벤트는 **바뀐 모든 열쇠의 새 값**을 실어 오고, 거기에는 페어링 토큰도 들어
   * 있습니다. 임의의 페이지 위에서 도는 코드에 토큰을 내려보내게 됩니다.
   */
  SETTINGS_CHANGED: 'audire/settings-changed',
});

/** 확장이 켜져 있지 않을 때의 상태. 서버 오류와 구분되어야 합니다. */
export const DISABLED = 'disabled';
