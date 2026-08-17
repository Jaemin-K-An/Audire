/**
 * 확장 설정과 페어링 토큰의 보관소.
 *
 * 두 가지가 여기서 갈립니다.
 *
 * - **설정**(기준 주소, 프로파일, 목표 자막률, 켜짐 여부)은 팝업이 읽고 씁니다.
 * - **토큰**은 서비스 워커만 읽습니다. 팝업에도, 콘텐츠 스크립트에도 값이 나가지
 *   않습니다. 콘텐츠 스크립트는 임의의 페이지 위에서 도는 코드이고, 거기로 토큰이
 *   내려가는 순간 로컬 서버의 출처 구분이 무의미해집니다.
 *
 * 기본값은 **꺼짐**입니다. 설치만으로 어떤 페이지도 읽지 않습니다.
 */

export const SETTINGS_KEY = 'audire.settings';
const TOKEN_KEY = 'audire.token';

export const DEFAULT_SETTINGS = Object.freeze({
  baseUrl: 'http://127.0.0.1:8000',
  /** 선택된 청취자 프로파일. 없으면 채점하지 않습니다. */
  profileId: null,
  /** 목표 자막률. 임계값은 서버가 이 값으로 청취자별로 정합니다. */
  targetCaptionRate: 0.2,
  /** 기본은 꺼짐. 사용자가 켜기 전에는 자막을 읽지 않습니다. */
  enabled: false,
});

/** 로컬 루프백만 허용합니다. 원격 주소를 넣으면 자막이 기기 밖으로 나갑니다. */
const LOOPBACK = /^http:\/\/(127\.0\.0\.1|localhost)(:\d+)?$/;

/** @param {string} value */
export function isLoopbackBaseUrl(value) {
  return typeof value === 'string' && LOOPBACK.test(value.replace(/\/+$/, ''));
}

export class SettingsStore {
  /** @param {{local: {get: Function, set: Function, remove: Function}}} [storage] */
  constructor(storage) {
    this._area = storage ?? globalThis.chrome?.storage?.local;
    if (!this._area) throw new Error('chrome.storage.local is unavailable');
  }

  async load() {
    const stored = await this._area.get(SETTINGS_KEY);
    return { ...DEFAULT_SETTINGS, ...(stored?.[SETTINGS_KEY] ?? {}) };
  }

  /** @param {Partial<typeof DEFAULT_SETTINGS>} patch */
  async update(patch) {
    const next = { ...(await this.load()), ...patch };
    if (!isLoopbackBaseUrl(next.baseUrl)) {
      // 조용히 기본값으로 되돌리지 않습니다. 사용자가 원격 주소를 넣었다면 그 의도를
      // 알아야 하고, 그 요청은 거절되어야 합니다.
      throw new Error(`base URL must be loopback, got ${next.baseUrl}`);
    }
    if (!(next.targetCaptionRate > 0 && next.targetCaptionRate < 1)) {
      throw new Error(`target caption rate must be in (0, 1), got ${next.targetCaptionRate}`);
    }
    await this._area.set({ [SETTINGS_KEY]: next });
    return next;
  }

  /**
   * 토큰을 읽습니다. **서비스 워커 전용입니다.** 메시지 응답에 실으면 안 됩니다.
   * @returns {Promise<string|null>}
   */
  async readToken() {
    const stored = await this._area.get(TOKEN_KEY);
    const token = stored?.[TOKEN_KEY];
    return typeof token === 'string' && token ? token : null;
  }

  /** @param {string} token */
  async writeToken(token) {
    if (typeof token !== 'string' || !token) throw new Error('token must be a non-empty string');
    await this._area.set({ [TOKEN_KEY]: token });
  }

  async clearToken() {
    await this._area.remove(TOKEN_KEY);
  }
}
