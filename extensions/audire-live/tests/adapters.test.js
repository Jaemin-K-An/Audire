/**
 * 어댑터 계약과 텍스트 읽기 규칙.
 *
 * jsdom 을 쓰지 않습니다. `readVisibleText` 가 `textContent` 하나만 만지도록 설계된
 * 덕분에 평범한 객체로 시험할 수 있고, 그 제약 자체가 여기서 확인됩니다 — 어느 날
 * 누군가 `innerHTML` 을 쓰기 시작하면 이 시험이 먼저 깨집니다.
 */

import assert from 'node:assert/strict';
import test from 'node:test';

import { createLocalFixtureAdapter } from '../src/adapters/localFixture.js';
import { listAdapters, register, reset, selectAdapter } from '../src/adapters/registry.js';
import { PageState } from '../src/states.js';
import {
  assertAdapterContract,
  cueFingerprint,
  normaliseCueText,
  readVisibleText,
} from '../src/adapters/types.js';

/** @param {Partial<import('../src/adapters/types.js').SubtitleAdapter>} overrides */
function fakeAdapter(overrides = {}) {
  return {
    id: 'fixture',
    matches: () => true,
    locateRoot: () => null,
    readCue: () => null,
    lastInspected: '2026-08-17',
    ...overrides,
  };
}

test.beforeEach(() => reset());

test('중첩 span 을 합칠 때 공백을 넣지 않는다', () => {
  // 자막 플랫폼은 한 단어를 여러 span 으로 쪼갭니다. 공백을 넣으면 "안녕"이 "안 녕"이
  // 되고, 그 순간 채점 대상이 실재하지 않는 단어가 됩니다.
  const root = { textContent: '안녕하세요' };
  assert.equal(readVisibleText(root), '안녕하세요');
});

test('빈 뿌리는 빈 문자열이다', () => {
  assert.equal(readVisibleText(null), '');
  assert.equal(readVisibleText({ textContent: '   ' }), '');
});

test('조합형과 완성형 한글이 같은 큐가 된다', () => {
  const composed = '학교'; // NFC
  const decomposed = composed.normalize('NFD');
  assert.notEqual(composed, decomposed, 'NFD 표기가 실제로 달라야 시험이 의미가 있다');
  assert.equal(normaliseCueText(decomposed), composed);
});

test('공백만 다른 자막은 같은 지문이다', () => {
  const a = { source: 'fixture', text: normaliseCueText('오늘  날씨는   맑음') };
  const b = { source: 'fixture', text: normaliseCueText('오늘 날씨는 맑음\n') };
  assert.equal(cueFingerprint(a), cueFingerprint(b));
});

test('출처가 다르면 같은 문장도 다른 지문이다', () => {
  const a = { source: 'youtube', text: '같은 문장' };
  const b = { source: 'user-selected', text: '같은 문장' };
  assert.notEqual(cueFingerprint(a), cueFingerprint(b));
});

test('필수 항목이 빠진 어댑터는 등록에서 거절된다', () => {
  for (const missing of ['id', 'matches', 'locateRoot', 'readCue', 'lastInspected']) {
    const adapter = fakeAdapter();
    delete adapter[missing];
    assert.throws(() => assertAdapterContract(adapter), new RegExp(missing));
  }
});

test('마지막 확인 날짜가 없으면 등록되지 않는다', () => {
  // 사이트 마크업은 예고 없이 바뀝니다. 이 날짜가 선택자 유지보수의 유일한 단서입니다.
  assert.throws(() => assertAdapterContract(fakeAdapter({ lastInspected: '최근' })), /ISO date/);
});

test('같은 id 를 두 번 등록할 수 없다', () => {
  register(fakeAdapter());
  assert.throws(() => register(fakeAdapter()), /already registered/);
  assert.equal(listAdapters().length, 1);
});

test('맞는 어댑터가 없으면 명시적인 미지원 상태다', () => {
  // null 이면 호출자가 "아직 안 읽었음"·"자막 없음"·"지원 안 함" 중 무엇으로든 읽을 수
  // 있고, 그 셋은 사용자에게 전혀 다른 뜻입니다.
  register(fakeAdapter({ id: 'never', matches: () => false }));
  const result = selectAdapter({ hostname: 'example.com' });
  assert.equal(result.state, PageState.NO_MATCHING_ADAPTER);
  assert.equal(result.adapter, null);
});

test('등록 순서가 우선순위이며 결정적이다', () => {
  register(fakeAdapter({ id: 'first' }));
  register(fakeAdapter({ id: 'second' }));
  for (let i = 0; i < 5; i += 1) {
    assert.equal(selectAdapter({ hostname: 'example.com' }).adapter.id, 'first');
  }
});

test('어댑터 하나가 던져도 나머지가 평가된다', () => {
  register(
    fakeAdapter({
      id: 'broken',
      matches: () => {
        throw new Error('boom');
      },
    }),
  );
  register(fakeAdapter({ id: 'good', matches: (loc) => loc.hostname === 'example.com' }));
  const result = selectAdapter({ hostname: 'example.com' });
  assert.equal(result.state, PageState.OK);
  assert.equal(result.adapter.id, 'good');
});

test('던진 어댑터가 일치로 취급되지 않는다', () => {
  register(
    fakeAdapter({
      id: 'broken',
      matches: () => {
        throw new Error('boom');
      },
    }),
  );
  assert.equal(selectAdapter({ hostname: 'example.com' }).state, PageState.NO_MATCHING_ADAPTER);
});

test('픽스처 어댑터는 등록 후에도 실제 사이트를 끌어오지 않는다', () => {
  // 등록소를 거친 선택 경로에서도 같은 성질이 유지되어야 합니다.
  register(createLocalFixtureAdapter({ document: null }));
  const youtube = { protocol: 'https:', hostname: 'www.youtube.com', pathname: '/watch' };
  assert.equal(selectAdapter(youtube).state, PageState.NO_MATCHING_ADAPTER);

  const fixture = { protocol: 'file:', hostname: '', pathname: '/repo/fixtures/ott-page.html' };
  assert.equal(selectAdapter(fixture).adapter.id, 'local-fixture');
});
