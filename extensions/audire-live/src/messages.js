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
});

/** 확장이 켜져 있지 않을 때의 상태. 서버 오류와 구분되어야 합니다. */
export const DISABLED = 'disabled';
