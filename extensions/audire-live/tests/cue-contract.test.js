/**
 * SubtitleCue 는 DOM 객체가 아니라 **전선 위의 데이터**입니다.
 *
 * 왜 이것이 계약이어야 하는가
 * ---------------------------
 * 큐는 콘텐츠 스크립트에서 서비스 워커로, 거기서 로컬 서버로 건너갑니다. 그 경계를
 * 넘는 값에 DOM 객체가 섞이면 다음이 일어납니다.
 *
 * - `{...rect}` 가 `{}` 가 됩니다. DOMRect 의 8개 좌표는 프로토타입의 접근자이지 자기
 *   속성이 아니기 때문입니다. 메시지를 조립하다 전개 한 번이면 좌표가 통째로 사라지고,
 *   아무 오류도 나지 않습니다.
 * - 값이 페이지의 배치 상태에 묶여 있습니다. 큐를 잠시 들고 있는 동안 페이지가 스크롤되면
 *   "관측 시점의 위치" 가 아닌 것을 들고 있게 됩니다.
 * - Element 가 새면 확장이 페이지 노드에 대한 참조를 붙들게 됩니다.
 *
 * 그래서 검증은 "값이 숫자인가" 가 아니라 **"자기 데이터 속성인가"** 를 봅니다. 그것이
 * 스냅숏과 DOM 객체를 실제로 가르는 성질입니다.
 */

import assert from 'node:assert/strict';
import test from 'node:test';

import {
  CAPTION_RECT_KEYS,
  assertCaptionRect,
  assertSubtitleCue,
  normaliseCueText,
  snapshotRect,
} from '../src/adapters/types.js';
import { FakeDOMRect, element } from './helpers/dom.js';

function validRect() {
  return { x: 10, y: 20, width: 300, height: 40, top: 20, right: 310, bottom: 60, left: 10 };
}

function validCue(overrides = {}) {
  return {
    source: 'local-fixture',
    text: '오늘 병원에 갑니다',
    observedAtMs: 1234.5,
    ...overrides,
  };
}

// --- CaptionRect ------------------------------------------------------------

test('올바른 사각형은 JSON 왕복을 견딘다', () => {
  const rect = assertCaptionRect(validRect());
  assert.deepEqual(JSON.parse(JSON.stringify(rect)), rect);
});

test('DOMRect 는 계약을 통과하지 못한다', () => {
  // DOMRect 도 toJSON 이 있어 JSON.stringify 는 통과합니다. 그래서 직렬화만으로는
  // 걸러지지 않고, 자기 속성 여부로 봐야 합니다.
  const domRect = new FakeDOMRect(10, 20, 300, 40);
  assert.equal(Object.keys(domRect).length, 0, 'DOMRect 의 좌표는 자기 속성이 아니다');
  assert.deepEqual({ ...domRect }, {}, '전개하면 좌표가 사라진다 — 이것이 막으려는 사고다');
  assert.throws(() => assertCaptionRect(domRect), /own data properties/);
});

test('접근자로 위장한 사각형도 거절된다', () => {
  const sneaky = {};
  for (const key of CAPTION_RECT_KEYS) {
    Object.defineProperty(sneaky, key, { get: () => 1, enumerable: true });
  }
  assert.throws(() => assertCaptionRect(sneaky), /own data properties/);
});

test('유한하지 않은 좌표는 거절된다', () => {
  for (const bad of [Number.NaN, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY]) {
    assert.throws(() => assertCaptionRect({ ...validRect(), top: bad }), /finite/);
  }
});

test('좌표가 숫자가 아니면 거절된다', () => {
  assert.throws(() => assertCaptionRect({ ...validRect(), x: '10' }), /finite/);
  assert.throws(() => assertCaptionRect({ ...validRect(), x: null }), /finite/);
});

test('빠진 좌표는 거절된다', () => {
  const missing = validRect();
  delete missing.bottom;
  assert.throws(() => assertCaptionRect(missing), /bottom/);
});

test('모르는 열쇠는 거절된다', () => {
  // 여기가 느슨하면 사각형 안에 Element 를 실어 보낼 수 있습니다.
  assert.throws(
    () => assertCaptionRect({ ...validRect(), node: element('div') }),
    /unexpected/,
  );
});

test('요소에서 뜬 스냅숏은 순수 데이터다', () => {
  const node = element('div');
  node.setRect(new FakeDOMRect(5, 15, 200, 30));
  const snapshot = snapshotRect(node);

  assert.doesNotThrow(() => assertCaptionRect(snapshot));
  assert.deepEqual(Object.keys(snapshot).sort(), [...CAPTION_RECT_KEYS].sort());
  assert.equal(snapshot.right, 205);
});

test('스냅숏은 원본과 끊어져 있다', () => {
  // 페이지가 스크롤되어도 큐가 들고 있는 것은 관측 시점의 위치여야 합니다.
  const node = element('div');
  node.setRect(new FakeDOMRect(0, 0, 100, 20));
  const snapshot = snapshotRect(node);
  node.setRect(new FakeDOMRect(0, 500, 100, 20));

  assert.equal(snapshot.y, 0, 'snapshot must not track later layout changes');
});

test('위치를 알 수 없으면 null 이다', () => {
  // 좌표를 지어내지 않습니다. 없으면 없다고 합니다.
  assert.equal(snapshotRect(null), null);
  assert.equal(snapshotRect({}), null);
  const broken = element('div');
  broken.setRect(new FakeDOMRect(Number.NaN, 0, 0, 0));
  assert.equal(snapshotRect(broken), null);
});

// --- SubtitleCue ------------------------------------------------------------

test('올바른 큐는 JSON 왕복을 견딘다', () => {
  const cue = assertSubtitleCue({ ...validCue(), boundingRect: validRect(), language: 'ko' });
  assert.deepEqual(JSON.parse(JSON.stringify(cue)), cue);
});

test('출처가 비어 있으면 거절된다', () => {
  assert.throws(() => assertSubtitleCue(validCue({ source: '' })), /source/);
  assert.throws(() => assertSubtitleCue(validCue({ source: 42 })), /source/);
});

test('정규화되지 않은 텍스트는 거절된다', () => {
  // 어댑터가 정규화를 잊으면 같은 자막이 매번 새 큐가 되어 중복 요청이 나갑니다.
  // 조용히 고쳐주지 않고 어댑터의 결함으로 드러냅니다.
  assert.throws(() => assertSubtitleCue(validCue({ text: '오늘  병원에' })), /normalised/);
  assert.throws(() => assertSubtitleCue(validCue({ text: ' 오늘' })), /normalised/);
  assert.throws(
    () => assertSubtitleCue(validCue({ text: '학교'.normalize('NFD') })),
    /normalised/,
  );
});

test('정규화된 텍스트는 그대로 통과한다', () => {
  const text = normaliseCueText('  오늘\n병원에   갑니다 ');
  assert.equal(text, '오늘 병원에 갑니다');
  assert.doesNotThrow(() => assertSubtitleCue(validCue({ text })));
});

test('구두점과 이모지는 그대로 남는다', () => {
  // HTML 을 해석하지 않는 것과 마찬가지로, 텍스트를 마음대로 걸러내지도 않습니다.
  const text = normaliseCueText('정말요? 🙂 (네)');
  assert.equal(text, '정말요? 🙂 (네)');
  assert.doesNotThrow(() => assertSubtitleCue(validCue({ text })));
});

test('관측 시각은 유한한 음이 아닌 수여야 한다', () => {
  for (const bad of [Number.NaN, Number.POSITIVE_INFINITY, -1, '0', null]) {
    assert.throws(() => assertSubtitleCue(validCue({ observedAtMs: bad })), /observedAtMs/);
  }
  assert.doesNotThrow(() => assertSubtitleCue(validCue({ observedAtMs: 0 })));
});

test('언어는 선택이지만 비어 있을 수는 없다', () => {
  assert.doesNotThrow(() => assertSubtitleCue(validCue()));
  assert.doesNotThrow(() => assertSubtitleCue(validCue({ language: 'ko' })));
  assert.throws(() => assertSubtitleCue(validCue({ language: '' })), /language/);
});

test('사각형은 선택이지만 DOM 객체일 수는 없다', () => {
  assert.doesNotThrow(() => assertSubtitleCue(validCue({ boundingRect: validRect() })));
  assert.throws(
    () => assertSubtitleCue(validCue({ boundingRect: new FakeDOMRect(0, 0, 1, 1) })),
    /own data properties/,
  );
});

test('Element 는 어떤 경로로도 큐에 실릴 수 없다', () => {
  // 모르는 열쇠를 거절하는 것이 이 성질을 지탱합니다.
  assert.throws(() => assertSubtitleCue({ ...validCue(), node: element('div') }), /unexpected/);
  assert.throws(() => assertSubtitleCue({ ...validCue(), root: element('div') }), /unexpected/);
});

test('큐 전체가 자기 데이터 속성이어야 한다', () => {
  const cue = validCue();
  Object.defineProperty(cue, 'text', { get: () => '오늘', enumerable: true });
  assert.throws(() => assertSubtitleCue(cue), /own data properties/);
});

test('큐가 아닌 것은 거절된다', () => {
  for (const bad of [null, undefined, 'cue', 42, []]) {
    assert.throws(() => assertSubtitleCue(bad));
  }
});
