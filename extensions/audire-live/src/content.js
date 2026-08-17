/**
 * 페이지 위에서 도는 실제 배선.
 *
 * 하는 일은 제어기를 만들고 브라우저 쪽 물건(진짜 `location`, 진짜
 * `MutationObserver`, 진짜 `chrome.runtime`)을 꽂는 것뿐입니다. 판단은 전부
 * `pageController.js` 와 `subtitleObserver.js` 에 있고, 그쪽은 브라우저 없이 시험됩니다.
 *
 * 이 파일이 얇아야 하는 이유가 그것입니다 — 여기 로직이 쌓이면 시험할 수 없는 곳에
 * 쌓입니다.
 */

import { createLocalFixtureAdapter } from './adapters/localFixture.js';
import { register, reset, selectAdapter } from './adapters/registry.js';
import { createPageController } from './observer/pageController.js';
import { createSubtitleObserver } from './observer/subtitleObserver.js';

/**
 * 이 탭의 화면 식별자.
 *
 * 순서 판정이 화면별로 이루어져야 합니다. 서비스 워커는 클라이언트를 하나만 두므로,
 * 이 값이 없으면 탭 A 의 응답이 탭 B 보다 늦게 왔다는 이유로 버려집니다.
 */
function makeStreamId() {
  const random = globalThis.crypto?.randomUUID?.() ?? String(Math.random()).slice(2);
  return `tab-${random}`;
}

export async function bootstrap() {
  // 등록소는 모듈 수준 상태입니다. 같은 프레임에서 두 번 부트스트랩되면 중복 등록으로
  // 던지므로 먼저 비웁니다.
  reset();
  register(createLocalFixtureAdapter());

  const controller = createPageController({
    location: globalThis.location,
    resolveAdapter: selectAdapter,
    createObserver: (options) => createSubtitleObserver(options),
    sendMessage: (message) => chrome.runtime.sendMessage(message),
    streamId: makeStreamId(),
    // Phase 10 의 렌더러가 여기에 붙습니다. 지금은 결과를 받아 두기만 하고 그리지
    // 않습니다 — 그리는 것은 별도 결정이고, 배선이 먼저 맞아야 합니다.
    onResult: () => {},
    onClear: () => {},
  });

  chrome.runtime.onMessage.addListener((message) => {
    void controller.handleMessage(message);
    // 응답을 돌려주지 않습니다. 방송을 받는 쪽입니다.
    return false;
  });

  await controller.start();
  return controller;
}
