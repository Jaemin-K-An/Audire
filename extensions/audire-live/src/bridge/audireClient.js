/**
 * 로컬 AUDIRE 서버로 가는 다리.
 *
 * 여기서 지켜지는 것
 * ------------------
 * - **오래된 응답이 새 자막을 덮어쓰지 않습니다.** 큐 A 요청이 큐 B 보다 늦게 돌아와도
 *   화면에는 B 가 남아야 합니다. 라이브 자막에서 이것은 흔히 일어나고, 처리하지 않으면
 *   화면이 과거로 되돌아갑니다.
 * - **실패 사유가 구분됩니다.** 서버 미가동, 미페어링, 프로파일 미준비, 모델 없음이 각각
 *   다른 상태로 전달되어야 사용자에게 무엇을 하라고 말할 수 있습니다.
 * - 토큰은 헤더로만 나가고 URL 에 실리지 않습니다.
 */

export const DEFAULT_BASE_URL = 'http://127.0.0.1:8000';

/** 확장이 사용자에게 보여줄 수 있는 상태. 각각 다른 안내가 필요합니다. */
export const LiveState = Object.freeze({
  OK: 'ok',
  SERVER_OFFLINE: 'server_offline',
  NOT_PAIRED: 'not_paired',
  UNKNOWN_PROFILE: 'unknown_profile',
  PROFILE_NOT_READY: 'profile_not_ready',
  MODEL_UNAVAILABLE: 'model_unavailable',
  CONTRACT_MISMATCH: 'contract_mismatch',
  INVALID_CUE: 'invalid_cue',
  UNKNOWN_ERROR: 'unknown_error',
});

export class AudireClient {
  /**
   * @param {{baseUrl?: string, token?: string|null, fetchImpl?: typeof fetch}} [options]
   */
  constructor(options = {}) {
    this.baseUrl = options.baseUrl ?? DEFAULT_BASE_URL;
    this.token = options.token ?? null;
    this._fetch = options.fetchImpl ?? globalThis.fetch?.bind(globalThis);
    // 마지막으로 **보낸** 큐의 순번. 응답이 도착했을 때 이것과 비교해 오래된 것을 버립니다.
    this._sequence = 0;
    this._latestApplied = 0;
  }

  /** @param {string|null} token */
  setToken(token) {
    this.token = token;
  }

  _headers() {
    const headers = { 'content-type': 'application/json' };
    if (this.token) headers['x-audire-token'] = this.token;
    return headers;
  }

  async _request(path, init = {}) {
    if (!this._fetch) {
      return { state: LiveState.SERVER_OFFLINE, error: 'fetch is unavailable' };
    }
    let response;
    try {
      response = await this._fetch(`${this.baseUrl}${path}`, {
        ...init,
        headers: this._headers(),
      });
    } catch (cause) {
      // 로컬 서버가 꺼져 있는 것과 요청이 거절된 것은 사용자에게 다른 뜻입니다.
      return { state: LiveState.SERVER_OFFLINE, error: String(cause) };
    }

    let body = null;
    try {
      body = await response.json();
    } catch {
      body = null;
    }
    if (response.ok) return { state: LiveState.OK, body };

    const reason = body?.detail?.reason;
    const known = Object.values(LiveState).includes(reason) ? reason : LiveState.UNKNOWN_ERROR;
    return { state: known, status: response.status, body };
  }

  status() {
    return this._request('/api/live/status', { method: 'GET' });
  }

  profiles() {
    return this._request('/api/live/profiles', { method: 'GET' });
  }

  /**
   * 새 페어링 토큰을 받습니다.
   *
   * 응답의 토큰은 **호출자가 저장만 하고 어디에도 전달하지 않아야 합니다.** 이 메서드가
   * 토큰이 확장 안으로 들어오는 유일한 지점입니다.
   */
  pair(label = 'browser-extension') {
    return this._request('/api/live/pair', {
      method: 'POST',
      body: JSON.stringify({ label }),
    });
  }

  /**
   * 큐 하나를 채점합니다.
   *
   * 반환값의 `stale` 이 true 이면 **이 응답을 화면에 반영하면 안 됩니다.** 더 새로운 큐가
   * 이미 적용되었다는 뜻입니다.
   *
   * @param {{profileId: string, cueId: string, text: string, source: string, targetCaptionRate?: number}} cue
   */
  async scoreCue(cue) {
    const sequence = ++this._sequence;
    const result = await this._request('/api/live/score-cue', {
      method: 'POST',
      body: JSON.stringify({
        profile_id: cue.profileId,
        cue_id: cue.cueId,
        text: cue.text,
        source: cue.source,
        ...(cue.targetCaptionRate === undefined
          ? {}
          : { target_caption_rate: cue.targetCaptionRate }),
      }),
    });

    if (sequence < this._latestApplied) {
      return { ...result, stale: true, sequence };
    }
    this._latestApplied = sequence;
    return { ...result, stale: false, sequence };
  }
}
