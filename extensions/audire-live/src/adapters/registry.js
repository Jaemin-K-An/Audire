/**
 * 어댑터 등록소.
 *
 * 현재 페이지에 맞는 어댑터를 고르고, 없으면 **명시적으로 "지원하지 않음"** 을 돌려줍니다.
 * 비슷해 보이는 어댑터로 넘어가지 않습니다 — 그러면 자막이 아닌 요소를 읽고도 성공한
 * 것처럼 보입니다.
 */

import { PageState } from '../states.js';
import { assertAdapterContract } from './types.js';

/** @type {import('./types.js').SubtitleAdapter[]} */
const registered = [];

/**
 * 어댑터를 등록합니다. 계약 위반은 등록 시점에 드러납니다.
 * @param {Partial<import('./types.js').SubtitleAdapter>} adapter
 */
export function register(adapter) {
  const checked = assertAdapterContract(adapter);
  if (registered.some((existing) => existing.id === checked.id)) {
    throw new Error(`subtitle adapter "${checked.id}" is already registered`);
  }
  registered.push(checked);
  return checked;
}

/** 테스트가 상태를 격리하기 위해 씁니다. */
export function reset() {
  registered.length = 0;
}

/** @returns {import('./types.js').SubtitleAdapter[]} */
export function listAdapters() {
  return [...registered];
}

/**
 * 이 위치에 맞는 어댑터를 고릅니다.
 *
 * 결과는 어댑터가 아니라 **상태**입니다. `null` 을 돌려주면 호출자가 그것을 "아직 안
 * 읽었음"·"자막 없음"·"지원 안 함" 중 무엇으로든 해석할 수 있고, 그 셋은 사용자에게
 * 전혀 다른 뜻입니다.
 *
 * 등록 순서가 곧 우선순위입니다. 먼저 등록된 어댑터가 먼저 평가됩니다.
 *
 * @param {Location | object | null} location
 * @returns {{state: string, adapter: import('./types.js').SubtitleAdapter | null}}
 */
export function selectAdapter(location) {
  for (const adapter of registered) {
    let matched = false;
    try {
      matched = adapter.matches(location);
    } catch {
      // 어댑터 하나가 던져도 나머지가 계속 평가되어야 합니다. 다만 던진 어댑터를
      // 조용히 "맞음" 으로 취급하지는 않습니다.
      matched = false;
    }
    if (matched) return { state: PageState.OK, adapter };
  }
  return { state: PageState.NO_MATCHING_ADAPTER, adapter: null };
}
