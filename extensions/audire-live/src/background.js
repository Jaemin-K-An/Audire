/**
 * 서비스 워커. 확장의 유일한 네트워크 출구입니다.
 *
 * 왜 콘텐츠 스크립트가 직접 서버를 부르지 않는가
 * ---------------------------------------------
 * 1. **토큰.** 콘텐츠 스크립트는 임의의 페이지 위에서 돕니다. 토큰이 거기로 내려가면
 *    로컬 서버의 출처 구분이 무너집니다. 토큰은 이 파일 밖으로 나가지 않습니다.
 * 2. **CORS.** MV3 에서 콘텐츠 스크립트의 fetch 는 페이지 출처를 따릅니다. youtube.com
 *    에서 127.0.0.1 로 직접 요청하면 막힙니다. 서비스 워커는 host_permissions 로
 *    확장 출처에서 요청합니다.
 * 3. **순서.** 오래된 큐 응답이 새 자막을 덮어쓰지 않도록 하는 판정이 한 곳에 있어야
 *    합니다. 탭마다 클라이언트를 두면 그 판정이 흩어집니다.
 *
 * 여기서 자막 텍스트는 **저장되지 않습니다.** 지나가고 끝입니다.
 */

import { AudireClient, LiveState } from './bridge/audireClient.js';
import { DISABLED, MessageType } from './messages.js';
import { SettingsStore } from './settings.js';

/**
 * 메시지 처리기. `chrome` 없이도 동작하도록 의존성을 받습니다 — 그래야 이 규칙들을
 * 실제로 시험할 수 있습니다.
 *
 * @param {{settings: SettingsStore, makeClient?: (baseUrl: string, token: string|null) => AudireClient}} deps
 */
export function createRouter(deps) {
  const settings = deps.settings;
  const makeClient =
    deps.makeClient ?? ((baseUrl, token) => new AudireClient({ baseUrl, token }));

  /** 같은 기준 주소/토큰이면 클라이언트를 재사용해야 큐 순서 판정이 이어집니다. */
  let cached = { key: null, client: /** @type {AudireClient|null} */ (null) };

  function invalidateClient() {
    cached = { key: null, client: null };
  }

  async function client() {
    const { baseUrl } = await settings.load();
    const token = await settings.readToken();
    const key = `${baseUrl} ${token ?? ''}`;
    if (cached.key !== key) {
      cached = { key, client: makeClient(baseUrl, token) };
    }
    return /** @type {AudireClient} */ (cached.client);
  }

  /**
   * @param {{type: string, payload?: any}} message
   * @returns {Promise<any>}
   */
  return async function handleMessage(message) {
    const type = message?.type;
    if (!Object.values(MessageType).includes(type)) {
      // 모르는 메시지를 통과시키지 않습니다. 조용히 무시하면 오타 난 호출이 "응답 없음"
      // 으로 보이고 원인을 찾기 어려워집니다.
      return { ok: false, state: LiveState.UNKNOWN_ERROR, error: `unknown message: ${type}` };
    }

    switch (type) {
      case MessageType.GET_STATE: {
        const current = await settings.load();
        const status = await (await client()).status();
        return {
          ok: status.state === LiveState.OK,
          state: status.state,
          // 토큰 값이 아니라 **있는지 여부**만 나갑니다.
          hasToken: (await settings.readToken()) !== null,
          settings: current,
          server: status.body ?? null,
        };
      }

      case MessageType.PAIR: {
        // 페어링은 토큰이 없는 상태에서 부르는 유일한 요청입니다.
        const { baseUrl } = await settings.load();
        const result = await makeClient(baseUrl, null).pair();
        if (result.state !== LiveState.OK || !result.body?.token) {
          return { ok: false, state: result.state, error: 'pairing failed' };
        }
        await settings.writeToken(result.body.token);
        invalidateClient();
        // 토큰은 저장만 하고 돌려주지 않습니다.
        return { ok: true, state: LiveState.OK, pairedAt: result.body.created_at_utc ?? null };
      }

      case MessageType.UNPAIR: {
        await settings.clearToken();
        invalidateClient();
        return { ok: true, state: LiveState.OK };
      }

      case MessageType.LIST_PROFILES: {
        const result = await (await client()).profiles();
        return {
          ok: result.state === LiveState.OK,
          state: result.state,
          profiles: result.body?.profiles ?? [],
        };
      }

      case MessageType.UPDATE_SETTINGS: {
        try {
          const next = await settings.update(message.payload ?? {});
          invalidateClient();
          return { ok: true, state: LiveState.OK, settings: next };
        } catch (error) {
          const detail = String(error?.message ?? error);
          return { ok: false, state: LiveState.UNKNOWN_ERROR, error: detail };
        }
      }

      case MessageType.SCORE_CUE: {
        const current = await settings.load();
        if (!current.enabled) {
          return { ok: false, state: DISABLED };
        }
        if (!current.profileId) {
          // 프로파일 없이 채점하지 않습니다. 임계값이 청취자에게서 나오기 때문입니다.
          return { ok: false, state: LiveState.UNKNOWN_PROFILE };
        }
        const result = await (await client()).scoreCue({
          profileId: current.profileId,
          cueId: message.payload?.cueId ?? '',
          text: message.payload?.text ?? '',
          source: message.payload?.source ?? 'unknown',
          targetCaptionRate: current.targetCaptionRate,
        });
        return {
          ok: result.state === LiveState.OK && !result.stale,
          state: result.state,
          stale: result.stale,
          result: result.body ?? null,
        };
      }

      default:
        return { ok: false, state: LiveState.UNKNOWN_ERROR };
    }
  };
}

// --- 확장 실행 환경에서만 연결됩니다. 테스트에서는 이 아래가 실행되지 않습니다. ---
if (typeof chrome !== 'undefined' && chrome.runtime?.onMessage) {
  const router = createRouter({ settings: new SettingsStore() });
  chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
    router(message).then(sendResponse, (error) =>
      sendResponse({ ok: false, state: LiveState.UNKNOWN_ERROR, error: String(error) }),
    );
    return true; // 비동기 응답을 쓰겠다는 뜻입니다.
  });
}
