/**
 * 자막 관찰자.
 *
 * 하는 일은 하나입니다: **논리적 자막 하나를 한 번만 내보내는 것.**
 *
 * MutationObserver 는 보이는 자막 하나에 대해 수십 번 발화합니다. 자막 플랫폼이 한
 * 문장을 span 여러 개로 쪼개고 그 쪼개는 방식을 계속 바꾸기 때문입니다. 그것을 그대로
 * 흘려보내면 문장 하나가 수십 번의 추론 요청이 됩니다.
 *
 * 여기 없는 것
 * ------------
 * 서버 요청도, 화면 그리기도 없습니다. 그 둘이 섞이면 중복 판정이 네트워크 상태와
 * 렌더링 타이밍에 얽혀 시험할 수 없게 됩니다.
 *
 * 디바운스도 없습니다
 * -------------------
 * MutationObserver 는 이미 브라우저가 마이크로태스크에서 묶어 전달합니다. 한 다발이
 * 한 번의 읽기가 되고, 그 위에 지문 비교가 얹힙니다. 여기에 타이머를 더하면 짧게
 * 스쳐가는 자막이 **한 번도 관측되지 않고** 사라집니다. 대사가 빠른 구간이 통째로
 * 비는 쪽이, 요청 몇 번 더 나가는 쪽보다 나쁩니다.
 */

import { assertSubtitleCue, cueFingerprint } from '../adapters/types.js';
import { PageState } from '../states.js';

/**
 * @param {{
 *   adapter: import('../adapters/types.js').SubtitleAdapter,
 *   onCue: (cue: import('../adapters/types.js').SubtitleCue) => void,
 *   onClear?: () => void,
 *   onError?: (state: string, detail: {adapterId: string, cueChars?: number}) => void,
 *   MutationObserverImpl?: typeof MutationObserver,
 * }} deps
 */
export function createSubtitleObserver(deps) {
  const adapter = deps.adapter;
  const onCue = deps.onCue;
  const onClear = deps.onClear ?? (() => {});
  const onError = deps.onError ?? (() => {});
  const MutationObserverImpl = deps.MutationObserverImpl ?? globalThis.MutationObserver;

  /** @type {any} */ let root = null;
  /** @type {any} */ let observer = null;
  /**
   * 마지막으로 **내보낸** 자막의 지문. 본 적 있는 문장을 전부 기억하지는 않습니다 —
   * 그러면 반복되는 대사가 두 번째부터 영영 나오지 않습니다.
   */
  let lastFingerprint = null;
  let cuesEmitted = 0;
  let clearsEmitted = 0;
  let cueChars = 0;

  function fail(state) {
    // 세부에 자막 내용을 넣지 않습니다. 길이·어댑터·상태만 남깁니다.
    onError(state, { adapterId: adapter.id, cueChars });
    return { state };
  }

  function read() {
    // 뿌리가 문서에서 떨어졌는지 먼저 봅니다. SPA 는 자막 컨테이너를 통째로 다시
    // 만들고, 그때 우리가 들고 있는 노드는 더는 화면의 그것이 아닙니다.
    if (root && root.isConnected === false) {
      stop();
      lastFingerprint = null;
      return fail(PageState.ROOT_STALE);
    }

    /** @type {import('../adapters/types.js').SubtitleCue | null} */
    let cue = null;
    try {
      cue = adapter.readCue();
      if (cue !== null) assertSubtitleCue(cue);
    } catch {
      // 어댑터의 결함이 관찰자를 죽이면 그 뒤의 자막이 전부 사라집니다. 이 한 번을
      // 실패로 보고하고 다음 변경에서 다시 시도합니다.
      return fail(PageState.INVALID_CUE);
    }

    if (cue === null) {
      // 사라짐은 빈 큐가 아니라 **상태 변화**입니다. 빈 문자열을 채점하러 보내면
      // 서버는 그것을 잘못된 입력으로 거절하고, 화면은 지워지지 않습니다.
      if (lastFingerprint !== null) {
        lastFingerprint = null;
        cueChars = 0;
        clearsEmitted += 1;
        onClear();
      }
      return { state: PageState.OK };
    }

    const fingerprint = cueFingerprint(cue);
    if (fingerprint === lastFingerprint) return { state: PageState.OK };

    lastFingerprint = fingerprint;
    cueChars = cue.text.length;
    cuesEmitted += 1;
    onCue(cue);
    return { state: PageState.OK };
  }

  function stop() {
    observer?.disconnect();
    observer = null;
    root = null;
  }

  function start() {
    stop();

    root = adapter.locateRoot();
    if (!root) {
      // 어댑터가 지목한 노드가 없습니다. 자막이 꺼졌거나 선택자가 낡았습니다. 어느
      // 쪽이든 **다른 노드를 찾아 나서지 않습니다.**
      return fail(PageState.SUBTITLE_ROOT_NOT_FOUND);
    }

    // 뿌리의 **부모**를 관찰합니다. 뿌리 자신만 보면 뿌리가 제거될 때의 childList
    // 변경은 부모에서 일어나므로 영영 발화하지 않고, 관찰자는 조용히 죽은 채로
    // 남습니다. 부모까지 올라가는 것은 뿌리 수명을 보기 위한 최소 범위이며,
    // subtree 로 안쪽 변경도 같은 관찰자가 받습니다.
    const target = root.parentElement ?? root;
    observer = new MutationObserverImpl(() => read());
    observer.observe(target, { childList: true, subtree: true, characterData: true });

    // 사용자가 재생 중간에 켤 수 있습니다. 다음 변경까지 기다리면 그 문장을 놓칩니다.
    return read();
  }

  return {
    start,
    stop,
    isActive: () => observer !== null,
    /** 자막 내용은 들어 있지 않습니다. 길이와 개수만 남습니다. */
    stats: () => ({ adapterId: adapter.id, cuesEmitted, clearsEmitted, cueChars }),
  };
}
