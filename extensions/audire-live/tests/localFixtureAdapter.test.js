/**
 * 픽스처 어댑터.
 *
 * 이 어댑터는 제품 기능이 아니라 **계약의 참조 구현**입니다. 실제 사이트 선택자를 쓰기
 * 전에, 어댑터 계약 전체가 결정적으로 동작하는지 여기서 증명합니다.
 *
 * 가장 중요한 규칙은 하나입니다: **자막 뿌리를 못 찾으면 그것으로 끝난다.** 비슷한 것을
 * 찾아 나서지 않습니다. 픽스처 페이지에는 제목·설명·댓글에 자막처럼 생긴 한국어 문장이
 * 일부러 깔려 있어서, 그 규칙이 깨지면 여기서 드러납니다.
 */

import assert from 'node:assert/strict';
import test from 'node:test';

import { FIXTURE_SUBTITLE_ID, createLocalFixtureAdapter } from '../src/adapters/localFixture.js';
import { assertAdapterContract, assertSubtitleCue } from '../src/adapters/types.js';
import { FakeDOMRect, createDocument, element, resetDom, text } from './helpers/dom.js';

/** 픽스처 페이지를 닮은 문서. 자막이 아닌 한국어가 곳곳에 있습니다. */
function fixtureDocument({ withRoot = true } = {}) {
  const stage = element('div', { id: 'fixture-stage' });
  if (withRoot) {
    const root = element('div', { id: FIXTURE_SUBTITLE_ID });
    root.setRect(new FakeDOMRect(40, 300, 520, 32));
    stage.append(root);
  }

  const body = element('body', {
    children: [
      element('h1', { children: [text('지어낸 다큐멘터리 — 병원 가는 길')] }),
      stage,
      element('p', { children: [text('어제 학교에서 만난 사람과 오늘 병원에 갑니다.')] }),
      element('ul', {
        children: [element('li', { children: [text('이 장면에서 나온 말이 잘 안 들렸습니다')] })],
      }),
    ],
  });
  return createDocument(body);
}

function adapterFor(document, now = () => 1000) {
  return createLocalFixtureAdapter({ document, now });
}

test.beforeEach(() => resetDom());

// --- 계약 -------------------------------------------------------------------

test('픽스처 어댑터는 어댑터 계약을 지킨다', () => {
  assert.doesNotThrow(() => assertAdapterContract(adapterFor(fixtureDocument())));
});

// --- matches ----------------------------------------------------------------

test('픽스처 경로에서만 맞는다', () => {
  const adapter = adapterFor(fixtureDocument());
  assert.equal(adapter.matches({ protocol: 'file:', hostname: '', pathname: '/x/fixtures/ott-page.html' }), true);
  assert.equal(
    adapter.matches({ protocol: 'http:', hostname: '127.0.0.1', pathname: '/fixtures/ott-page.html' }),
    true,
  );
  assert.equal(
    adapter.matches({ protocol: 'http:', hostname: 'localhost', pathname: '/fixtures/ott-page.html' }),
    true,
  );
});

test('픽스처 어댑터가 실제 사이트에 붙지 않는다', () => {
  // 이것이 뚫리면 개발용 어댑터가 사용자의 실제 시청 페이지를 읽습니다.
  const adapter = adapterFor(fixtureDocument());
  const foreign = [
    { protocol: 'https:', hostname: 'www.youtube.com', pathname: '/watch' },
    { protocol: 'https:', hostname: 'www.netflix.com', pathname: '/watch/80000' },
    // 경로만 흉내 낸 원격 호스트도 맞으면 안 됩니다.
    { protocol: 'https:', hostname: 'evil.example.com', pathname: '/fixtures/ott-page.html' },
    { protocol: 'http:', hostname: '127.0.0.1.evil.com', pathname: '/fixtures/ott-page.html' },
    // 루프백이지만 다른 페이지입니다.
    { protocol: 'http:', hostname: '127.0.0.1', pathname: '/index.html' },
  ];
  for (const location of foreign) {
    assert.equal(adapter.matches(location), false, `${location.hostname}${location.pathname}`);
  }
});

test('location 이 없어도 던지지 않는다', () => {
  assert.equal(adapterFor(fixtureDocument()).matches(null), false);
});

// --- locateRoot -------------------------------------------------------------

test('자막 뿌리를 찾는다', () => {
  const adapter = adapterFor(fixtureDocument());
  assert.equal(adapter.locateRoot()?.id, FIXTURE_SUBTITLE_ID);
});

test('뿌리가 없으면 null 이다', () => {
  assert.equal(adapterFor(fixtureDocument({ withRoot: false })).locateRoot(), null);
});

// --- readCue ----------------------------------------------------------------

test('뿌리가 비어 있으면 큐가 없다', () => {
  assert.equal(adapterFor(fixtureDocument()).readCue(), null);
});

test('공백만 있는 뿌리도 큐가 없다', () => {
  const document = fixtureDocument();
  document.querySelector(`#${FIXTURE_SUBTITLE_ID}`).append(text('   \n  '));
  assert.equal(adapterFor(document).readCue(), null);
});

test('뿌리가 없으면 페이지의 다른 한국어를 읽지 않는다', () => {
  // 이 문서에는 제목·설명·댓글에 자막처럼 생긴 문장이 있습니다. 뿌리가 사라졌을 때
  // "비슷한 것" 으로 흘러가면 사용자는 엉뚱한 텍스트를 자막으로 보게 됩니다.
  const document = fixtureDocument({ withRoot: false });
  assert.equal(adapterFor(document).readCue(), null);
});

test('중첩 span 한 단어가 붙어서 나온다', () => {
  const document = fixtureDocument();
  document
    .querySelector(`#${FIXTURE_SUBTITLE_ID}`)
    .append(element('span', { children: [text('안')] }), element('span', { children: [text('녕')] }));

  assert.equal(adapterFor(document).readCue().text, '안녕');
});

test('실제 공백은 단어 경계로 남는다', () => {
  // 중첩 span 을 붙이는 규칙이 진짜 공백까지 지워버리면 안 됩니다.
  const document = fixtureDocument();
  document
    .querySelector(`#${FIXTURE_SUBTITLE_ID}`)
    .append(
      element('span', { children: [text('오늘')] }),
      text(' '),
      element('span', { children: [text('병원에')] }),
    );

  assert.equal(adapterFor(document).readCue().text, '오늘 병원에');
});

test('줄바꿈과 연속 공백은 하나로 접힌다', () => {
  const document = fixtureDocument();
  document.querySelector(`#${FIXTURE_SUBTITLE_ID}`).append(text('오늘\n병원에   갑니다'));
  assert.equal(adapterFor(document).readCue().text, '오늘 병원에 갑니다');
});

test('조합형으로 들어온 자막이 완성형으로 나온다', () => {
  const document = fixtureDocument();
  document.querySelector(`#${FIXTURE_SUBTITLE_ID}`).append(text('학교'.normalize('NFD')));
  const cue = adapterFor(document).readCue();
  assert.equal(cue.text, '학교');
  assert.equal(cue.text.normalize('NFC'), cue.text);
});

test('구두점과 이모지는 텍스트로 남는다', () => {
  const document = fixtureDocument();
  document.querySelector(`#${FIXTURE_SUBTITLE_ID}`).append(text('정말요? 🙂 (네)'));
  assert.equal(adapterFor(document).readCue().text, '정말요? 🙂 (네)');
});

test('HTML 을 해석하지 않는다', () => {
  // 페이지가 넣은 문자열은 끝까지 텍스트입니다.
  const document = fixtureDocument();
  document.querySelector(`#${FIXTURE_SUBTITLE_ID}`).append(text('<b>강조</b>'));
  assert.equal(adapterFor(document).readCue().text, '<b>강조</b>');
});

// --- 큐의 모양 ---------------------------------------------------------------

test('큐가 전선 계약을 지킨다', () => {
  const document = fixtureDocument();
  document.querySelector(`#${FIXTURE_SUBTITLE_ID}`).append(text('오늘 병원에 갑니다'));
  const cue = adapterFor(document, () => 2500).readCue();

  assert.doesNotThrow(() => assertSubtitleCue(cue));
  assert.equal(cue.source, 'local-fixture');
  assert.equal(cue.observedAtMs, 2500);
  assert.deepEqual(JSON.parse(JSON.stringify(cue)), cue);
});

test('큐의 위치는 DOMRect 가 아니라 스냅숏이다', () => {
  const document = fixtureDocument();
  const root = document.querySelector(`#${FIXTURE_SUBTITLE_ID}`);
  root.append(text('오늘'));
  const cue = adapterFor(document).readCue();

  assert.equal(cue.boundingRect.top, 300);
  assert.equal(cue.boundingRect.right, 560);
  assert.equal(Object.getPrototypeOf(cue.boundingRect), Object.prototype);

  // 스냅숏은 이후의 배치 변화를 따라가지 않습니다.
  root.setRect(new FakeDOMRect(0, 900, 10, 10));
  assert.equal(cue.boundingRect.top, 300);
});

test('위치를 알 수 없으면 boundingRect 없이 나간다', () => {
  const document = fixtureDocument();
  const root = document.querySelector(`#${FIXTURE_SUBTITLE_ID}`);
  root.setRect(new FakeDOMRect(Number.NaN, 0, 0, 0));
  root.append(text('오늘'));

  const cue = adapterFor(document).readCue();
  assert.ok(!('boundingRect' in cue), 'a rect that cannot be measured must be omitted, not faked');
  assert.doesNotThrow(() => assertSubtitleCue(cue));
});
