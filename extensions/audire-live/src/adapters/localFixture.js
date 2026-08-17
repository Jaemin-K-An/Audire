/**
 * 픽스처 어댑터 — 어댑터 계약의 참조 구현.
 *
 * 이것은 제품 기능이 아닙니다. `fixtures/ott-page.html` 하나만 읽으며, 그 페이지는 이
 * 저장소 안에 있습니다. 실제 사이트 선택자를 쓰기 전에 계약 전체가 결정적으로 도는지
 * 증명하는 것이 목적입니다.
 *
 * 여기서 고정되는 규칙
 * --------------------
 * - **뿌리를 못 찾으면 그것으로 끝입니다.** 비슷한 요소를 찾아 나서지 않습니다. 픽스처
 *   페이지에는 제목·설명·댓글에 자막처럼 생긴 한국어가 일부러 깔려 있고, 그 규칙이
 *   깨지면 사용자는 엉뚱한 텍스트를 자막으로 보게 됩니다.
 * - **뿌리 탐색은 한 가지 방법뿐입니다.** 대안 선택자를 나열해 두면 첫 번째가 낡았을 때
 *   두 번째가 조용히 받아내고, 선택자가 낡았다는 사실이 영영 드러나지 않습니다.
 * - **자기 페이지에만 붙습니다.** 개발용 어댑터가 사용자의 실제 시청 페이지에 붙으면
 *   안 됩니다.
 */

import { assertSubtitleCue, readVisibleText, snapshotRect } from './types.js';

/** 픽스처 페이지의 자막 뿌리 id. `fixtures/ott-page.html` 과 같아야 합니다. */
export const FIXTURE_SUBTITLE_ID = 'audire-fixture-subtitles';

/** 이 어댑터가 붙는 유일한 경로. */
export const FIXTURE_PATH_SUFFIX = '/fixtures/ott-page.html';

/** 이 어댑터의 선택자를 마지막으로 실제 확인한 날짜. */
export const FIXTURE_LAST_INSPECTED = '2026-08-17';

/**
 * 이 위치가 저장소 안의 픽스처 페이지인지 판정합니다.
 *
 * 경로만 보면 원격 사이트가 같은 경로를 흉내 내 어댑터를 끌어올 수 있습니다. 그래서
 * `file:` 이거나 루프백 http 여야 합니다.
 *
 * @param {Location | {protocol?: string, hostname?: string, pathname?: string} | null} location
 */
export function matchesFixtureLocation(location) {
  if (!location || typeof location !== 'object') return false;
  if (!String(location.pathname ?? '').endsWith(FIXTURE_PATH_SUFFIX)) return false;

  const protocol = String(location.protocol ?? '');
  if (protocol === 'file:') return true;
  const hostname = String(location.hostname ?? '');
  return protocol === 'http:' && (hostname === '127.0.0.1' || hostname === 'localhost');
}

/**
 * @param {{document?: Document, now?: () => number}} [deps]
 * @returns {import('./types.js').SubtitleAdapter}
 */
export function createLocalFixtureAdapter(deps = {}) {
  const doc = deps.document ?? globalThis.document;
  const now = deps.now ?? (() => globalThis.performance?.now() ?? 0);

  function locateRoot() {
    // 방법은 이 한 줄뿐입니다. 실패하면 실패한 채로 둡니다.
    return doc?.querySelector(`#${FIXTURE_SUBTITLE_ID}`) ?? null;
  }

  return {
    id: 'local-fixture',
    lastInspected: FIXTURE_LAST_INSPECTED,
    matches: matchesFixtureLocation,
    locateRoot,

    readCue() {
      const root = locateRoot();
      if (!root) return null;

      const text = readVisibleText(root);
      if (!text) return null;

      const boundingRect = snapshotRect(root);
      const cue = {
        source: 'local-fixture',
        text,
        observedAtMs: now(),
        // 잴 수 없으면 넣지 않습니다. 0 으로 채우면 화면 맨 위에 있다는 뜻이 됩니다.
        ...(boundingRect === null ? {} : { boundingRect }),
      };
      // 어댑터가 자기 출력을 스스로 검사합니다. 잘못된 큐가 관찰자까지 가기 전에
      // 여기서 터지는 편이 낫습니다.
      return assertSubtitleCue(cue);
    },
  };
}
