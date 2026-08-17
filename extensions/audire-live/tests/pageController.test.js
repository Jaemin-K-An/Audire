/**
 * 페이지 제어기.
 *
 * 어댑터 선택 · 관찰 · 메시지가 여기서 만납니다. 그래서 각각은 맞는데 **이어 붙였을 때**
 * 틀리는 결함이 나올 수 있는 자리입니다.
 *
 * 가장 중요한 것: 꺼져 있으면 페이지를 **읽지 않습니다.** 서비스 워커도 꺼짐을 막지만,
 * 그 지점에서는 자막이 이미 확장 안으로 들어온 뒤입니다.
 */

import assert from 'node:assert/strict';
import test from 'node:test';

import { FIXTURE_SUBTITLE_ID, createLocalFixtureAdapter } from '../src/adapters/localFixture.js';
import { DISABLED, MessageType } from '../src/messages.js';
import { createPageController } from '../src/observer/pageController.js';
import { createSubtitleObserver } from '../src/observer/subtitleObserver.js';
import { PageState } from '../src/states.js';
import { MutationObserverStub, createDocument, element, resetDom, text, tick } from './helpers/dom.js';

/** 즉시 실행하는 예약기. 재시도 지연을 기다리지 않고 결정적으로 돌립니다. */
function immediateScheduler() {
  const pending = new Map();
  let next = 1;
  return {
    setTimeout: (fn) => {
      const id = next++;
      pending.set(id, fn);
      return id;
    },
    clearTimeout: (id) => pending.delete(id),
    /** 대기 중인 콜백을 한 바퀴 실행합니다. */
    run() {
      const callbacks = [...pending.values()];
      pending.clear();
      return callbacks.map((fn) => fn());
    },
    get size() {
      return pending.size;
    },
  };
}

function harness({ enabled = true, adapterState = PageState.OK, scoreResponse } = {}) {
  const stage = element('div', { id: 'fixture-stage' });
  stage.append(element('div', { id: FIXTURE_SUBTITLE_ID }));
  const body = element('body', {
    children: [stage, element('p', { children: [text('이 문장은 자막이 아닙니다')] })],
  });
  const document = createDocument(body);
  const adapter = createLocalFixtureAdapter({ document, now: () => 1 });

  const sent = [];
  const sendMessage = async (message) => {
    sent.push(message);
    if (message.type === MessageType.GET_STATE) {
      return { ok: true, state: 'ok', settings: { enabled, profileId: 'L01' } };
    }
    if (message.type === MessageType.SCORE_CUE) {
      return (
        scoreResponse?.(message) ?? {
          ok: true,
          state: 'ok',
          stale: false,
          result: {
            words: [
              { text: '오늘', risk: 0.1, selected: false },
              { text: '병원에', risk: 0.9, selected: true },
            ],
          },
        }
      );
    }
    return { ok: true };
  };

  const scheduler = immediateScheduler();
  const results = [];
  const clears = [];
  const states = [];

  const controller = createPageController({
    location: { protocol: 'file:', hostname: '', pathname: '/repo/fixtures/ott-page.html' },
    resolveAdapter: () =>
      adapterState === PageState.OK
        ? { state: PageState.OK, adapter }
        : { state: adapterState, adapter: null },
    createObserver: (options) =>
      createSubtitleObserver({ ...options, MutationObserverImpl: MutationObserverStub }),
    sendMessage,
    streamId: 'tab-test',
    onResult: (r) => results.push(r),
    onClear: () => clears.push(true),
    onStateChange: (s) => states.push(s),
    scheduler,
  });

  const root = () => document.querySelector(`#${FIXTURE_SUBTITLE_ID}`);
  return { controller, sent, results, clears, states, root, stage, scheduler, document };
}

const scoreCalls = (sent) => sent.filter((m) => m.type === MessageType.SCORE_CUE);

test.beforeEach(() => resetDom());

// --- 꺼짐 --------------------------------------------------------------------

test('꺼져 있으면 페이지를 읽지 않는다', async () => {
  const h = harness({ enabled: false });
  h.root().append(text('민감할 수 있는 자막입니다'));

  const result = await h.controller.start();

  assert.equal(result.state, DISABLED);
  assert.deepEqual(scoreCalls(h.sent), [], 'no cue may leave the page while disabled');
  assert.equal(h.controller.getState().adapterId, null, '어댑터조차 고르지 않는다');
});

test('꺼진 상태에서 DOM 이 바뀌어도 아무것도 나가지 않는다', async () => {
  const h = harness({ enabled: false });
  await h.controller.start();
  for (let i = 0; i < 5; i += 1) {
    h.root().replaceChildren(text(`자막 ${i}`));
    await tick();
  }
  assert.deepEqual(scoreCalls(h.sent), []);
});

// --- 지원하지 않는 페이지 ------------------------------------------------------

test('어댑터가 없으면 관찰하지 않고 상태로 알린다', async () => {
  const h = harness({ adapterState: PageState.NO_MATCHING_ADAPTER });
  const result = await h.controller.start();
  assert.equal(result.state, PageState.NO_MATCHING_ADAPTER);
  assert.deepEqual(scoreCalls(h.sent), []);
});

// --- 정상 흐름 ---------------------------------------------------------------

test('자막이 서비스 워커까지 이어진다', async () => {
  const h = harness();
  await h.controller.start();
  h.root().replaceChildren(text('오늘 병원에 갑니다'));
  await tick();
  await tick();

  const calls = scoreCalls(h.sent);
  assert.equal(calls.length, 1);
  assert.equal(calls[0].payload.text, '오늘 병원에 갑니다');
  assert.equal(calls[0].payload.streamId, 'tab-test');
  assert.equal(h.results.length, 1);
  assert.equal(h.controller.getState().lastSelected, 1);
});

test('큐 id 에 자막 내용이 들어가지 않는다', async () => {
  // 큐 id 는 로그와 응답 대조에 쓰입니다. 거기에 자막을 실으면 기록하지 않기로 한 것이
  // 식별자를 타고 남습니다.
  const h = harness();
  await h.controller.start();
  h.root().replaceChildren(text('민감할 수 있는 자막입니다'));
  await tick();
  await tick();

  const { cueId } = scoreCalls(h.sent)[0].payload;
  assert.ok(!cueId.includes('민감'), `cue id leaked the subtitle: ${cueId}`);
  assert.match(cueId, /^tab-test:\d+$/);
});

test('같은 문장의 DOM 변경이 요청 하나로 남는다', async () => {
  const h = harness();
  await h.controller.start();
  for (let i = 0; i < 20; i += 1) {
    h.root().replaceChildren(
      ...Array.from({ length: (i % 3) + 1 }, (_, k) => element('span', { children: [text(k === 0 ? '오늘 병원에 갑니다' : '')] })),
    );
    await tick();
  }
  await tick();
  assert.equal(scoreCalls(h.sent).length, 1);
});

test('사라짐은 채점 요청이 아니라 clear 다', async () => {
  const h = harness();
  await h.controller.start();
  h.root().replaceChildren(text('오늘 갑니다'));
  await tick();
  await tick();
  h.root().replaceChildren();
  await tick();

  assert.equal(h.clears.length, 1);
  assert.equal(scoreCalls(h.sent).length, 1, 'clearing must not send an empty cue for scoring');
  assert.equal(h.controller.getState().lastCueChars, 0);
});

test('오래된 응답은 화면에 반영되지 않는다', async () => {
  // 응답이 `ok: true` 여도 stale 이면 그리지 않습니다. "지난 자막을 화면에 올리지
  // 않는다" 는 화면 쪽 불변식이고, 제어기가 화면 쪽입니다. 서비스 워커가 stale 을
  // ok:false 로도 표시한다는 사실에 기대면, 그 결합이 끊어지는 날 조용히 무너집니다.
  const h = harness({
    scoreResponse: () => ({ ok: true, stale: true, state: 'ok', result: { words: [] } }),
  });
  await h.controller.start();
  h.root().replaceChildren(text('지난 자막'));
  await tick();
  await tick();

  assert.deepEqual(h.results, [], 'a stale response must never reach the screen');
});

test('오래된 응답이 상태를 흔들지 않는다', async () => {
  const h = harness({
    scoreResponse: () => ({ ok: false, stale: true, state: 'server_offline' }),
  });
  await h.controller.start();
  h.root().replaceChildren(text('지난 자막'));
  await tick();
  await tick();

  assert.equal(h.controller.getState().state, PageState.OK, 'stale is not a failure');
});

test('서버 실패 사유가 상태로 드러난다', async () => {
  const h = harness({
    scoreResponse: () => ({ ok: false, stale: false, state: 'profile_not_ready' }),
  });
  await h.controller.start();
  h.root().replaceChildren(text('오늘 갑니다'));
  await tick();
  await tick();

  assert.equal(h.controller.getState().state, 'profile_not_ready');
  assert.deepEqual(h.results, []);
});

// --- 뿌리 수명 ---------------------------------------------------------------

test('뿌리가 다시 생기면 제어기가 붙인다', async () => {
  // 관찰자는 ROOT_STALE 을 보고하고 멈춥니다(설계 A). 다시 붙이는 것은 제어기 몫입니다.
  const h = harness();
  await h.controller.start();
  h.root().replaceChildren(text('먼저'));
  await tick();
  await tick();

  h.root().remove();
  h.stage.append(element('div', { id: FIXTURE_SUBTITLE_ID, children: [text('나중')] }));
  await tick();
  await tick();

  const texts = scoreCalls(h.sent).map((m) => m.payload.text);
  assert.deepEqual(texts, ['먼저', '나중']);
});

test('뿌리가 영영 안 돌아오면 유한 번 시도하고 포기한다', async () => {
  // document 를 영원히 뒤지지 않습니다.
  const h = harness();
  await h.controller.start();
  h.root().remove();
  await tick();

  for (let i = 0; i < 20; i += 1) h.scheduler.run();

  assert.equal(h.scheduler.size, 0, 'the retry loop must terminate');
  assert.equal(h.controller.getState().state, PageState.SUBTITLE_ROOT_NOT_FOUND);
});

test('다시 붙으면 재시도 예산이 회복된다', async () => {
  const h = harness();
  await h.controller.start();
  const before = h.controller.getState().reattachesLeft;

  h.root().remove();
  h.stage.append(element('div', { id: FIXTURE_SUBTITLE_ID }));
  await tick();

  assert.equal(h.controller.getState().reattachesLeft, before);
});

// --- 설정 변경 ---------------------------------------------------------------

test('설정이 켜지면 새로고침 없이 관찰이 시작된다', async () => {
  let enabled = false;
  const stage = element('div', { id: 'fixture-stage' });
  stage.append(element('div', { id: FIXTURE_SUBTITLE_ID, children: [text('이미 떠 있는 자막')] }));
  const document = createDocument(element('body', { children: [stage] }));
  const sent = [];
  const controller = createPageController({
    location: { protocol: 'file:', hostname: '', pathname: '/x/fixtures/ott-page.html' },
    resolveAdapter: () => ({
      state: PageState.OK,
      adapter: createLocalFixtureAdapter({ document, now: () => 1 }),
    }),
    createObserver: (o) => createSubtitleObserver({ ...o, MutationObserverImpl: MutationObserverStub }),
    sendMessage: async (message) => {
      sent.push(message);
      if (message.type === MessageType.GET_STATE) return { settings: { enabled } };
      return { ok: true, state: 'ok', stale: false, result: { words: [] } };
    },
    streamId: 'tab-toggle',
    scheduler: immediateScheduler(),
  });

  await controller.start();
  assert.deepEqual(scoreCalls(sent), [], '꺼진 채로 시작하면 아무것도 읽지 않는다');

  enabled = true;
  await controller.handleMessage({ type: MessageType.SETTINGS_CHANGED });
  await tick();

  assert.equal(scoreCalls(sent).length, 1, '토글이 새로고침 없이 반영되어야 한다');
});

test('설정이 꺼지면 즉시 멈춘다', async () => {
  // 팝업에서 껐는데 탭을 새로 고쳐야 멈춘다면 그것은 꺼진 것이 아닙니다.
  let enabled = true;
  const stage = element('div', { id: 'fixture-stage' });
  stage.append(element('div', { id: FIXTURE_SUBTITLE_ID }));
  const document = createDocument(element('body', { children: [stage] }));
  const sent = [];
  const controller = createPageController({
    location: { protocol: 'file:', hostname: '', pathname: '/x/fixtures/ott-page.html' },
    resolveAdapter: () => ({
      state: PageState.OK,
      adapter: createLocalFixtureAdapter({ document, now: () => 1 }),
    }),
    createObserver: (o) => createSubtitleObserver({ ...o, MutationObserverImpl: MutationObserverStub }),
    sendMessage: async (message) => {
      sent.push(message);
      if (message.type === MessageType.GET_STATE) return { settings: { enabled } };
      return { ok: true, state: 'ok', stale: false, result: { words: [] } };
    },
    streamId: 'tab-toggle',
    scheduler: immediateScheduler(),
  });

  await controller.start();
  enabled = false;
  await controller.handleMessage({ type: MessageType.SETTINGS_CHANGED });

  const before = scoreCalls(sent).length;
  document.querySelector(`#${FIXTURE_SUBTITLE_ID}`).replaceChildren(text('꺼진 뒤의 자막'));
  await tick();
  assert.equal(scoreCalls(sent).length, before);
});

// --- 정지와 개인정보 ----------------------------------------------------------

test('정지 후에는 자막이 나가지 않는다', async () => {
  const h = harness();
  await h.controller.start();
  h.controller.stop();
  h.root().replaceChildren(text('정지 후'));
  await tick();
  assert.deepEqual(scoreCalls(h.sent), []);
});

test('제어기 상태에 자막 내용이 없다', async () => {
  const h = harness();
  await h.controller.start();
  h.root().replaceChildren(text('민감할 수 있는 자막입니다'));
  await tick();
  await tick();

  const serialised = JSON.stringify(h.controller.getState());
  assert.ok(!serialised.includes('민감'), 'controller state must not retain cue text');
  assert.equal(h.controller.getState().lastCueChars, '민감할 수 있는 자막입니다'.length);
});

test('콘텐츠 스크립트 모듈이 적재된다', async () => {
  // content.js 는 브라우저에서만 도는 얇은 배선이라 시험이 실행하지 않습니다. 그래서
  // import 오타가 조용히 남을 수 있고, 그것은 확장이 페이지에서 통째로 죽는 결함입니다.
  // 적어도 모듈 그래프가 성립하는지는 확인합니다.
  const module = await import('../src/content.js');
  assert.equal(typeof module.bootstrap, 'function');
});

test('chrome.storage 를 직접 만지지 않는다', async () => {
  // 콘텐츠 스크립트가 storage 를 읽으면 페어링 토큰이 임의의 페이지 위에서 도는 코드에
  // 닿습니다. 설정 변경도 onChanged 가 아니라 서비스 워커의 방송으로 받습니다.
  const trap = new Proxy(
    {},
    {
      get() {
        throw new Error('the content script must not touch chrome.storage');
      },
    },
  );
  const original = globalThis.chrome;
  globalThis.chrome = { storage: trap, runtime: trap };
  try {
    const h = harness();
    await h.controller.start();
    h.root().replaceChildren(text('오늘 갑니다'));
    await tick();
    await tick();
    assert.equal(scoreCalls(h.sent).length, 1);
  } finally {
    if (original === undefined) delete globalThis.chrome;
    else globalThis.chrome = original;
  }
});
