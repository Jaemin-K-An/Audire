/**
 * 페이지 제어기 — 어댑터 선택 · 관찰 · 메시지를 잇는 자리.
 *
 * 흐름
 * ----
 *   페이지 → 어댑터 선택 → 관찰자 → 서비스 워커 → 로컬 서버
 *
 * 여기서 지켜지는 것
 * ------------------
 * - **꺼져 있으면 관찰자가 아예 시작되지 않습니다.** 서비스 워커도 꺼짐을 막지만, 그
 *   지점에서는 자막 텍스트가 이미 확장 안으로 들어온 뒤입니다. 읽지 않는 것과 읽고 나서
 *   버리는 것은 다릅니다.
 * - **토큰을 만지지 않습니다.** `chrome.storage` 를 직접 읽지 않고, 설정 변경도
 *   `chrome.storage.onChanged` 가 아니라 서비스 워커의 방송으로 받습니다. 그 이벤트는
 *   바뀐 모든 열쇠의 새 값을 싣고 오며 거기에는 토큰이 들어 있습니다.
 * - **뿌리 재획득은 유한합니다.** 관찰자는 `ROOT_STALE` 을 보고하고 멈추고(설계 A),
 *   다시 붙이는 것은 이쪽 몫입니다. 정해진 횟수만 시도하고 포기합니다. `document` 를
 *   영원히 뒤지지 않습니다.
 * - **자막 텍스트를 보관하지 않습니다.** 상태에는 길이와 개수만 남습니다.
 */

import { DISABLED, MessageType } from '../messages.js';
import { PageState } from '../states.js';

/** 뿌리가 사라진 뒤 다시 붙여 볼 횟수. 이후에는 포기하고 상태로 알립니다. */
export const DEFAULT_REATTACH_ATTEMPTS = 5;
/** 재시도 간격(ms). SPA 가 컨테이너를 다시 그리는 데 걸리는 시간을 넘겨주는 정도입니다. */
export const REATTACH_DELAY_MS = 250;

/**
 * @param {{
 *   location: Location | object,
 *   resolveAdapter: (location: any) => {state: string, adapter: any},
 *   createObserver: (deps: any) => any,
 *   sendMessage: (message: any) => Promise<any>,
 *   streamId: string,
 *   onResult?: (result: any) => void,
 *   onClear?: () => void,
 *   onStateChange?: (state: string) => void,
 *   scheduler?: {setTimeout: Function, clearTimeout: Function},
 *   reattachAttempts?: number,
 * }} deps
 */
export function createPageController(deps) {
  const {
    location,
    resolveAdapter,
    createObserver,
    sendMessage,
    streamId,
    onResult = () => {},
    onClear = () => {},
    onStateChange = () => {},
  } = deps;
  const scheduler = deps.scheduler ?? { setTimeout, clearTimeout };
  const maxReattach = deps.reattachAttempts ?? DEFAULT_REATTACH_ATTEMPTS;

  /** @type {any} */ let observer = null;
  /** @type {any} */ let adapter = null;
  let state = DISABLED;
  let reattachTimer = null;
  let reattachesLeft = maxReattach;
  /** 이 화면에서 보낸 큐 수. 큐 id 를 만들 때 씁니다 — **자막 내용은 넣지 않습니다.** */
  let cueCounter = 0;
  let lastCueChars = 0;
  let lastSelected = 0;
  let inFlight = 0;

  function setState(next) {
    if (state === next) return;
    state = next;
    onStateChange(next);
  }

  function cancelReattach() {
    if (reattachTimer !== null) {
      scheduler.clearTimeout(reattachTimer);
      reattachTimer = null;
    }
  }

  async function scoreCue(cue) {
    inFlight += 1;
    try {
      const response = await sendMessage({
        type: MessageType.SCORE_CUE,
        payload: {
          // 큐 id 에 자막을 넣지 않습니다. id 는 로그와 응답 대조에 쓰이는 값입니다.
          cueId: `${streamId}:${(cueCounter += 1)}`,
          text: cue.text,
          source: cue.source,
          streamId,
        },
      });

      if (response?.stale) return; // 같은 화면의 더 새로운 자막이 이미 반영되었습니다.
      if (!response?.ok) {
        setState(response?.state ?? PageState.INVALID_CUE);
        return;
      }

      setState(PageState.OK);
      lastCueChars = cue.text.length;
      lastSelected = (response.result?.words ?? []).filter((w) => w.selected).length;
      onResult(response.result);
    } finally {
      inFlight -= 1;
    }
  }

  function attach() {
    const started = observer.start();
    if (started.state === PageState.OK) {
      reattachesLeft = maxReattach;
      setState(PageState.OK);
      return;
    }
    // 붙지 못했습니다. 뿌리가 아직 안 생겼을 수 있으니 정해진 횟수만 다시 시도합니다.
    if (reattachesLeft > 0) {
      reattachesLeft -= 1;
      cancelReattach();
      reattachTimer = scheduler.setTimeout(() => {
        reattachTimer = null;
        attach();
      }, REATTACH_DELAY_MS);
      return;
    }
    setState(started.state);
  }

  function handleObserverError(observerState) {
    if (observerState === PageState.ROOT_STALE) {
      // 관찰자는 멈췄습니다. 다시 붙이는 것은 이쪽 몫입니다(설계 A).
      setState(PageState.ROOT_STALE);
      attach();
      return;
    }
    setState(observerState);
  }

  /**
   * 확장이 켜져 있고 이 페이지에 어댑터가 있으면 관찰을 시작합니다.
   *
   * @returns {Promise<{state: string}>}
   */
  async function start() {
    stop();

    const status = await sendMessage({ type: MessageType.GET_STATE });
    if (!status?.settings?.enabled) {
      // **페이지를 읽지 않습니다.** 어댑터도 고르지 않습니다.
      setState(DISABLED);
      return { state: DISABLED };
    }

    const resolved = resolveAdapter(location);
    if (resolved.state !== PageState.OK) {
      setState(resolved.state);
      return { state: resolved.state };
    }
    adapter = resolved.adapter;

    observer = createObserver({
      adapter,
      onCue: (cue) => {
        void scoreCue(cue);
      },
      onClear: () => {
        lastCueChars = 0;
        lastSelected = 0;
        onClear();
      },
      onError: handleObserverError,
    });

    reattachesLeft = maxReattach;
    attach();
    return { state };
  }

  function stop() {
    cancelReattach();
    observer?.stop();
    observer = null;
    adapter = null;
    setState(DISABLED);
  }

  /**
   * 서비스 워커가 보낸 메시지. 설정이 바뀌면 켜짐/꺼짐이 즉시 반영되어야 합니다 —
   * 팝업에서 끄고도 탭을 새로 고쳐야 멈춘다면 그것은 꺼진 것이 아닙니다.
   */
  function handleMessage(message) {
    if (message?.type === MessageType.SETTINGS_CHANGED) return start();
    return undefined;
  }

  return {
    start,
    stop,
    handleMessage,
    /** 자막 내용은 들어 있지 않습니다. */
    getState: () => ({
      state,
      adapterId: adapter?.id ?? null,
      streamId,
      lastCueChars,
      lastSelected,
      inFlight,
      reattachesLeft,
    }),
  };
}
