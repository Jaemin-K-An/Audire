/**
 * 자막 관찰자.
 *
 * MutationObserver 는 **보이는 자막 하나에 대해 수십 번 발화합니다.** 자막 플랫폼이
 * 한 문장을 span 여러 개로 쪼개고, 그 쪼개는 방식을 프레임마다 바꾸기 때문입니다.
 * 그것을 그대로 흘려보내면 문장 하나가 수십 번의 추론 요청이 됩니다.
 *
 * 그래서 관찰자의 일은 "변경을 전달하는 것" 이 아니라 **"논리적 자막 하나를 한 번만
 * 내보내는 것"** 입니다.
 *
 * 여기에는 서버도 렌더링도 없습니다. 그 둘이 섞이면 중복 판정이 네트워크 상태에 얽혀
 * 시험할 수 없게 됩니다.
 */

import assert from 'node:assert/strict';
import test from 'node:test';

import { FIXTURE_SUBTITLE_ID, createLocalFixtureAdapter } from '../src/adapters/localFixture.js';
import { createSubtitleObserver } from '../src/observer/subtitleObserver.js';
import { PageState } from '../src/states.js';
import {
  MutationObserverStub,
  createDocument,
  element,
  liveObserverCount,
  resetDom,
  text,
  tick,
} from './helpers/dom.js';

function scene({ withRoot = true } = {}) {
  const stage = element('div', { id: 'fixture-stage' });
  if (withRoot) stage.append(element('div', { id: FIXTURE_SUBTITLE_ID }));

  const body = element('body', {
    children: [
      stage,
      element('p', { children: [text('어제 학교에서 만난 사람과 오늘 병원에 갑니다.')] }),
    ],
  });
  const document = createDocument(body);

  let clock = 0;
  const adapter = createLocalFixtureAdapter({ document, now: () => (clock += 1) });

  const cues = [];
  const clears = [];
  const errors = [];
  const observer = createSubtitleObserver({
    adapter,
    onCue: (cue) => cues.push(cue),
    onClear: () => clears.push(true),
    onError: (state, detail) => errors.push({ state, detail }),
    MutationObserverImpl: MutationObserverStub,
  });

  const root = () => document.querySelector(`#${FIXTURE_SUBTITLE_ID}`);

  /** 보이는 문장은 그대로 두고 DOM 구조만 바꿉니다. */
  const respan = (value, groupSize) => {
    const spans = [];
    for (let i = 0; i < value.length; i += groupSize) {
      spans.push(element('span', { children: [text(value.slice(i, i + groupSize))] }));
    }
    root().replaceChildren(...spans);
  };

  return { document, stage, adapter, observer, cues, clears, errors, root, respan };
}

test.beforeEach(() => resetDom());

// --- 시작 --------------------------------------------------------------------

test('자막이 없으면 큐도 없다', async () => {
  const s = scene();
  s.observer.start();
  await tick();
  assert.deepEqual(s.cues, []);
  assert.deepEqual(s.clears, []);
});

test('붙일 때 이미 있던 자막은 즉시 나온다', () => {
  // 사용자가 재생 중간에 켤 수 있습니다. 다음 변경까지 기다리면 그 문장을 놓칩니다.
  const s = scene();
  s.root().append(text('오늘 병원에 갑니다'));
  s.observer.start();
  assert.equal(s.cues.length, 1);
  assert.equal(s.cues[0].text, '오늘 병원에 갑니다');
});

test('뿌리가 없으면 명시적 상태로 실패한다', () => {
  const s = scene({ withRoot: false });
  const result = s.observer.start();
  assert.equal(result.state, PageState.SUBTITLE_ROOT_NOT_FOUND);
  assert.equal(s.observer.isActive(), false);
  assert.deepEqual(s.errors.map((e) => e.state), [PageState.SUBTITLE_ROOT_NOT_FOUND]);
});

test('뿌리가 없을 때 페이지의 다른 텍스트를 읽지 않는다', () => {
  const s = scene({ withRoot: false });
  s.observer.start();
  assert.deepEqual(s.cues, []);
});

// --- 중복 억제 ---------------------------------------------------------------

test('첫 자막이 정확히 한 번 나온다', async () => {
  const s = scene();
  s.observer.start();
  s.root().append(text('오늘 병원에 갑니다'));
  await tick();
  assert.equal(s.cues.length, 1);
  assert.equal(s.cues[0].text, '오늘 병원에 갑니다');
});

test('같은 문장에 대한 DOM 변경 20번이 큐 하나가 된다', async () => {
  // 발화를 **따로따로** 도착시킵니다. 한 다발로 몰면 관찰자가 한 번만 읽어서, 중복
  // 억제가 없어도 통과해 버립니다.
  const s = scene();
  s.observer.start();
  for (let i = 0; i < 20; i += 1) {
    s.respan('오늘 병원에 갑니다', (i % 3) + 1);
    await tick();
  }
  assert.equal(s.cues.length, 1, `expected exactly one logical cue, got ${s.cues.length}`);
});

test('span 구조만 달라지고 보이는 문장이 같으면 큐가 늘지 않는다', async () => {
  const s = scene();
  s.observer.start();
  s.respan('안녕하세요', 1);
  await tick();
  assert.equal(s.cues.length, 1);
  assert.equal(s.cues[0].text, '안녕하세요');

  s.respan('안녕하세요', 5);
  await tick();
  assert.equal(s.cues.length, 1, 'the visible sentence did not change');
});

test('중복 판정이 관측 시각에 의존하지 않는다', async () => {
  // 시각이 지문에 섞이면 모든 읽기가 새 큐가 되어 억제가 통째로 무력해집니다.
  const s = scene();
  s.observer.start();
  s.root().append(text('오늘'));
  await tick();
  const first = s.cues[0].observedAtMs;

  s.respan('오늘', 1);
  await tick();
  assert.equal(s.cues.length, 1);
  assert.ok(first >= 0);
});

// --- 문장 교체 ---------------------------------------------------------------

test('문장이 바뀌면 두 번째 큐가 나온다', async () => {
  const s = scene();
  s.observer.start();
  s.root().replaceChildren(text('오늘 갑니다'));
  await tick();
  s.root().replaceChildren(text('내일 갑니다'));
  await tick();

  assert.deepEqual(s.cues.map((c) => c.text), ['오늘 갑니다', '내일 갑니다']);
});

test('짧게 스쳐가는 자막도 놓치지 않는다', async () => {
  // 디바운스를 넣었다면 여기서 사라집니다. 관찰자에는 타이머가 없습니다.
  const s = scene();
  s.observer.start();
  for (const line of ['하나', '둘', '셋', '넷']) {
    s.root().replaceChildren(text(line));
    await tick();
  }
  assert.deepEqual(s.cues.map((c) => c.text), ['하나', '둘', '셋', '넷']);
});

test('이전 문장으로 되돌아가면 다시 나온다', async () => {
  // 지문은 "마지막에 내보낸 것" 과만 비교합니다. 본 적 있는 문장을 전부 기억하면
  // 반복되는 대사가 영영 안 나옵니다.
  const s = scene();
  s.observer.start();
  for (const line of ['가', '나', '가']) {
    s.root().replaceChildren(text(line));
    await tick();
  }
  assert.deepEqual(s.cues.map((c) => c.text), ['가', '나', '가']);
});

// --- 사라짐 ------------------------------------------------------------------

test('자막이 사라지면 clear 가 정확히 한 번 나온다', async () => {
  const s = scene();
  s.observer.start();
  s.root().replaceChildren(text('오늘 갑니다'));
  await tick();
  s.root().replaceChildren();
  await tick();

  assert.equal(s.clears.length, 1);
  assert.equal(s.cues.length, 1, 'disappearance must not be delivered as an empty cue');
});

test('빈 상태에서 계속 변경이 와도 clear 가 반복되지 않는다', async () => {
  const s = scene();
  s.observer.start();
  s.root().replaceChildren(text('오늘 갑니다'));
  await tick();
  for (let i = 0; i < 10; i += 1) {
    s.root().replaceChildren();
    await tick();
  }
  assert.equal(s.clears.length, 1);
});

test('시작부터 비어 있으면 clear 를 내지 않는다', async () => {
  // 아무것도 없던 상태에서 "없어졌다" 고 말할 것이 없습니다.
  const s = scene();
  s.observer.start();
  for (let i = 0; i < 3; i += 1) {
    s.root().replaceChildren();
    await tick();
  }
  assert.deepEqual(s.clears, []);
});

test('clear 이후 새 자막이 다시 나온다', async () => {
  const s = scene();
  s.observer.start();
  s.root().replaceChildren(text('오늘 갑니다'));
  await tick();
  s.root().replaceChildren();
  await tick();
  s.root().replaceChildren(text('오늘 갑니다'));
  await tick();

  assert.equal(s.cues.length, 2);
  assert.equal(s.clears.length, 1);
});

// --- 뿌리 수명 ---------------------------------------------------------------

test('뿌리가 사라지면 ROOT_STALE 을 내고 멈춘다', async () => {
  // SPA 는 자막 컨테이너를 통째로 다시 만듭니다. 조용히 죽으면 사용자는 자막이 없는
  // 장면인지 확장이 멈춘 것인지 알 수 없습니다.
  const s = scene();
  s.observer.start();
  s.root().replaceChildren(text('오늘 갑니다'));
  await tick();

  s.root().remove();
  await tick();

  assert.deepEqual(s.errors.map((e) => e.state), [PageState.ROOT_STALE]);
  assert.equal(s.observer.isActive(), false);
});

test('뿌리가 다시 생기면 재시작으로 이어진다', async () => {
  const s = scene();
  s.observer.start();
  s.root().remove();
  await tick();
  assert.equal(s.observer.isActive(), false);

  // 호출자가 다시 붙입니다. 관찰자가 스스로 document 를 뒤지지 않습니다.
  s.stage.append(element('div', { id: FIXTURE_SUBTITLE_ID, children: [text('새 자막입니다')] }));
  const result = s.observer.start();

  assert.equal(result.state, PageState.OK);
  assert.equal(s.cues.at(-1).text, '새 자막입니다');
});

test('ROOT_STALE 은 한 번만 나온다', async () => {
  const s = scene();
  s.observer.start();
  const detached = s.root();
  detached.remove();
  await tick();
  for (let i = 0; i < 5; i += 1) {
    s.stage.append(element('span'));
    await tick();
  }
  assert.equal(s.errors.filter((e) => e.state === PageState.ROOT_STALE).length, 1);
});

// --- 정지 --------------------------------------------------------------------

test('정지 후에는 어떤 변경도 나오지 않는다', async () => {
  const s = scene();
  s.observer.start();
  s.root().replaceChildren(text('오늘 갑니다'));
  await tick();
  s.observer.stop();

  s.root().replaceChildren(text('내일 갑니다'));
  await tick();
  s.root().replaceChildren();
  await tick();

  assert.equal(s.cues.length, 1);
  assert.deepEqual(s.clears, []);
});

test('정지가 관찰 등록을 실제로 놓아준다', async () => {
  const s = scene();
  s.observer.start();
  assert.ok(liveObserverCount() > 0);
  s.observer.stop();
  assert.equal(liveObserverCount(), 0, 'disconnect must release the registration, not just a flag');
});

test('다시 시작해도 관찰자가 쌓이지 않는다', () => {
  const s = scene();
  for (let i = 0; i < 5; i += 1) s.observer.start();
  assert.equal(liveObserverCount(), 1);
  s.observer.stop();
  assert.equal(liveObserverCount(), 0);
});

// --- 결함 있는 어댑터 ---------------------------------------------------------

test('어댑터가 잘못된 큐를 내면 INVALID_CUE 로 드러난다', async () => {
  const s = scene();
  const broken = {
    ...s.adapter,
    readCue: () => ({ source: 'broken', text: '  띄어쓰기가  안  접힌  ', observedAtMs: 1 }),
  };
  const cues = [];
  const errors = [];
  const observer = createSubtitleObserver({
    adapter: broken,
    onCue: (cue) => cues.push(cue),
    onClear: () => {},
    onError: (state, detail) => errors.push({ state, detail }),
    MutationObserverImpl: MutationObserverStub,
  });

  observer.start();
  assert.deepEqual(cues, []);
  assert.equal(errors[0].state, PageState.INVALID_CUE);
});

test('어댑터가 던져도 관찰자가 죽지 않는다', async () => {
  const s = scene();
  let fail = true;
  const flaky = {
    ...s.adapter,
    readCue: () => {
      if (fail) throw new Error('selector blew up');
      return { source: 'flaky', text: '회복했습니다', observedAtMs: 1 };
    },
  };
  const cues = [];
  const errors = [];
  const observer = createSubtitleObserver({
    adapter: flaky,
    onCue: (cue) => cues.push(cue),
    onClear: () => {},
    onError: (state) => errors.push(state),
    MutationObserverImpl: MutationObserverStub,
  });

  observer.start();
  assert.deepEqual(errors, [PageState.INVALID_CUE]);

  fail = false;
  s.root().append(text('무엇이든'));
  await tick();
  assert.equal(cues.at(-1).text, '회복했습니다');
});

// --- 개인정보 ----------------------------------------------------------------

test('오류 세부에는 자막 내용이 실리지 않는다', async () => {
  const s = scene();
  s.observer.start();
  s.root().replaceChildren(text('민감할 수 있는 자막입니다'));
  await tick();
  s.root().remove();
  await tick();

  const serialised = JSON.stringify(s.errors);
  assert.ok(!serialised.includes('민감할'), 'error detail must not carry cue text');
  assert.equal(s.errors.at(-1).detail.adapterId, 'local-fixture');
});

test('통계에는 길이만 남는다', async () => {
  const s = scene();
  s.observer.start();
  s.root().replaceChildren(text('오늘 병원에 갑니다'));
  await tick();

  const stats = s.observer.stats();
  assert.equal(stats.cueChars, 10);
  assert.equal(stats.cuesEmitted, 1);
  assert.ok(!JSON.stringify(stats).includes('병원'));
});

// --- 전선 계약 ---------------------------------------------------------------

// --- Phase 7 완료 조건 --------------------------------------------------------

test('Phase 7 완료 순서가 통째로 성립한다', async () => {
  // 개별 규칙이 각각 맞는 것과, 그것들이 **한 줄기로 이어졌을 때도** 맞는 것은 다른
  // 문제입니다. 순서가 섞이면 상태 기계의 결함이 드러납니다.
  const s = scene();
  const log = [];
  const observer = createSubtitleObserver({
    adapter: s.adapter,
    onCue: (cue) => log.push(`cue:${cue.text}`),
    onClear: () => log.push('clear'),
    onError: (state) => log.push(`error:${state}`),
    MutationObserverImpl: MutationObserverStub,
  });

  // 자막 없음 → 큐 없음
  observer.start();
  await tick();

  // 자막 A → 큐 하나
  s.root().replaceChildren(text('오늘 병원에 갑니다'));
  await tick();

  // A 의 DOM 이 반복해서 바뀜 → 중복 없음
  for (let i = 0; i < 20; i += 1) {
    s.respan('오늘 병원에 갑니다', (i % 3) + 1);
    await tick();
  }

  // 자막 B → 큐 하나 더
  s.root().replaceChildren(text('내일 학교에서 만납시다'));
  await tick();

  // 자막 제거 → clear 하나
  s.root().replaceChildren();
  await tick();

  // 뿌리가 낡음 → 명시적 상태
  s.root().remove();
  await tick();

  // 정지 이후의 변경 → 아무것도 나오지 않음
  observer.stop();
  s.stage.append(element('div', { id: FIXTURE_SUBTITLE_ID, children: [text('이건 안 나온다')] }));
  await tick();

  assert.deepEqual(log, [
    'cue:오늘 병원에 갑니다',
    'cue:내일 학교에서 만납시다',
    'clear',
    `error:${PageState.ROOT_STALE}`,
  ]);
});

test('내보낸 큐는 모두 JSON 왕복을 견딘다', async () => {
  const s = scene();
  s.observer.start();
  for (const line of ['오늘 갑니다', '내일 갑니다']) {
    s.root().replaceChildren(text(line));
    await tick();
  }
  assert.equal(s.cues.length, 2);
  for (const cue of s.cues) {
    assert.deepEqual(JSON.parse(JSON.stringify(cue)), cue);
  }
});
