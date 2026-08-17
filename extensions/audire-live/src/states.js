/**
 * 페이지 쪽 상태.
 *
 * 서버 쪽 상태(`bridge/audireClient.js` 의 `LiveState`)와 일부러 분리했습니다. 사용자가
 * 해야 할 일이 전혀 다르기 때문입니다 — 서버가 꺼진 것은 서버를 켜면 되고, 어댑터가 없는
 * 것은 이 사이트를 지원하지 않는다는 뜻이며, 선택자가 낡은 것은 확장을 고쳐야 한다는
 * 뜻입니다.
 *
 * 이것들을 "실패" 하나로 뭉치면 UI 가 그 셋을 구분할 수 없습니다. 특히 마지막 둘의 차이가
 * 중요합니다.
 *
 * - `NO_MATCHING_ADAPTER` — 이 사이트는 애초에 지원 대상이 아닙니다. 사용자가 할 일이
 *   없습니다.
 * - `SUBTITLE_ROOT_NOT_FOUND` — 어댑터는 이 사이트를 안다고 했는데 자막 뿌리가 없습니다.
 *   자막이 꺼져 있거나, **선택자가 낡았습니다.** 후자라면 유지보수 신호입니다.
 */

export const PageState = Object.freeze({
  /** 어댑터를 찾았고 자막을 읽을 수 있습니다. */
  OK: 'ok',
  /** 이 위치에 맞는 어댑터가 없습니다. 지원하지 않는 사이트입니다. */
  NO_MATCHING_ADAPTER: 'no_matching_adapter',
  /** 어댑터는 맞았지만 자막 뿌리가 없습니다. 자막이 꺼졌거나 선택자가 낡았습니다. */
  SUBTITLE_ROOT_NOT_FOUND: 'subtitle_root_not_found',
  /** 어댑터가 계약을 어기는 큐를 냈습니다. 고쳐야 할 결함입니다. */
  INVALID_CUE: 'invalid_cue',
  /** 관찰하던 자막 뿌리가 문서에서 사라졌습니다. 호출자가 다시 붙여야 합니다. */
  ROOT_STALE: 'root_stale',
});
