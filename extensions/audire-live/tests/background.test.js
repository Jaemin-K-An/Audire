/**
 * 서비스 워커의 메시지 규칙.
 *
 * 여기서 확인되는 것 중 두 개는 타협 대상이 아닙니다.
 *
 * 1. **토큰은 어떤 응답에도 실리지 않는다.** 응답은 콘텐츠 스크립트까지 갈 수 있고,
 *    그것은 임의의 페이지 위에서 도는 코드입니다.
 * 2. **꺼져 있으면 자막이 나가지 않는다.** 기본값이 꺼짐인 것과, 꺼짐이 실제로 요청을
 *    막는 것은 다른 문제입니다.
 */

import assert from 'node:assert/strict';
import test from 'node:test';

import { createRouter } from '../src/background.js';
import { LiveState } from '../src/bridge/audireClient.js';
import { DISABLED, MessageType } from '../src/messages.js';
import { SettingsStore } from '../src/settings.js';

/** chrome.storage.local 을 흉내 내는 메모리 저장소. */
function memoryArea(initial = {}) {
  const data = { ...initial };
  return {
    data,
    async get(key) {
      return key in data ? { [key]: data[key] } : {};
    },
    async set(patch) {
      Object.assign(data, patch);
    },
    async remove(key) {
      delete data[key];
    },
  };
}

/** 요청 경로를 기록하고 미리 정한 응답을 돌려주는 가짜 클라이언트. */
function stubClient(responses = {}) {
  const seen = [];
  return {
    seen,
    make: (baseUrl, token) => {
      const record = { baseUrl, token };
      return {
        async status() {
          seen.push({ ...record, call: 'status' });
          return responses.status ?? { state: LiveState.OK, body: { server: 'ok' } };
        },
        async profiles() {
          seen.push({ ...record, call: 'profiles' });
          return responses.profiles ?? { state: LiveState.OK, body: { profiles: [] } };
        },
        async pair() {
          seen.push({ ...record, call: 'pair' });
          return (
            responses.pair ?? {
              state: LiveState.OK,
              body: { token: 'issued-token', created_at_utc: '2026-08-17T00:00:00+00:00' },
            }
          );
        },
        async scoreCue(cue) {
          seen.push({ ...record, call: 'score-cue', cue });
          return responses.scoreCue ?? { state: LiveState.OK, body: { words: [] }, stale: false };
        },
      };
    },
  };
}

function build(responses, initial) {
  const area = memoryArea(initial);
  const client = stubClient(responses);
  const router = createRouter({
    settings: new SettingsStore({ get: area.get, set: area.set, remove: area.remove }),
    makeClient: client.make,
  });
  return { area, client, router };
}

test('모르는 메시지는 거절된다', async () => {
  const { router } = build();
  const result = await router({ type: 'audire/definitely-not-a-thing' });
  assert.equal(result.ok, false);
  assert.match(result.error, /unknown message/);
});

test('페어링 토큰은 저장되지만 응답에는 실리지 않는다', async () => {
  const { area, router } = build();
  const result = await router({ type: MessageType.PAIR });

  assert.equal(result.ok, true);
  assert.equal(area.data['audire.token'], 'issued-token');
  assert.ok(
    !JSON.stringify(result).includes('issued-token'),
    'the pairing token must never leave the service worker',
  );
});

test('상태 응답은 토큰 유무만 알려준다', async () => {
  const { router } = build(undefined, { 'audire.token': 'issued-token' });
  const state = await router({ type: MessageType.GET_STATE });

  assert.equal(state.hasToken, true);
  assert.ok(!JSON.stringify(state).includes('issued-token'));
});

test('페어링은 토큰 없이 요청된다', async () => {
  const { client, router } = build(undefined, { 'audire.token': 'old-token' });
  await router({ type: MessageType.PAIR });
  const pairCall = client.seen.find((c) => c.call === 'pair');
  assert.equal(pairCall.token, null);
});

test('페어링 후 다음 요청은 새 토큰을 쓴다', async () => {
  // 클라이언트를 캐시하므로, 무효화하지 않으면 옛 토큰으로 계속 요청합니다.
  const { client, router } = build(undefined, { 'audire.token': 'old-token' });
  await router({ type: MessageType.PAIR });
  await router({ type: MessageType.LIST_PROFILES });
  assert.equal(client.seen.at(-1).token, 'issued-token');
});

test('연결 해제 후에는 토큰 없이 요청한다', async () => {
  const { area, client, router } = build(undefined, { 'audire.token': 'old-token' });
  await router({ type: MessageType.UNPAIR });
  await router({ type: MessageType.LIST_PROFILES });

  assert.ok(!('audire.token' in area.data));
  assert.equal(client.seen.at(-1).token, null);
});

test('꺼져 있으면 자막이 서버로 나가지 않는다', async () => {
  const { client, router } = build(undefined, {
    'audire.settings': { enabled: false, profileId: 'L01' },
  });
  const result = await router({
    type: MessageType.SCORE_CUE,
    payload: { cueId: 'c1', text: '민감한 자막', source: 'fixture' },
  });

  assert.equal(result.state, DISABLED);
  assert.equal(client.seen.filter((c) => c.call === 'score-cue').length, 0);
});

test('기본 설정은 꺼짐이다', async () => {
  const { router } = build();
  const state = await router({ type: MessageType.GET_STATE });
  assert.equal(state.settings.enabled, false);
  assert.equal(state.settings.profileId, null);
});

test('프로파일이 없으면 채점하지 않는다', async () => {
  // 임계값이 청취자에게서 나오므로, 프로파일 없는 채점은 의미가 없습니다.
  const { client, router } = build(undefined, {
    'audire.settings': { enabled: true, profileId: null },
  });
  const result = await router({
    type: MessageType.SCORE_CUE,
    payload: { cueId: 'c1', text: '자막', source: 'fixture' },
  });

  assert.equal(result.state, LiveState.UNKNOWN_PROFILE);
  assert.equal(client.seen.filter((c) => c.call === 'score-cue').length, 0);
});

test('켜져 있으면 설정된 프로파일과 자막률로 채점한다', async () => {
  const { client, router } = build(undefined, {
    'audire.settings': { enabled: true, profileId: 'L07', targetCaptionRate: 0.3 },
  });
  await router({
    type: MessageType.SCORE_CUE,
    payload: { cueId: 'c1', text: '오늘 날씨는 맑음', source: 'fixture' },
  });

  const call = client.seen.find((c) => c.call === 'score-cue');
  assert.equal(call.cue.profileId, 'L07');
  assert.equal(call.cue.targetCaptionRate, 0.3);
});

test('오래된 큐 응답은 ok 가 아니다', async () => {
  const { router } = build(
    { scoreCue: { state: LiveState.OK, body: { words: [] }, stale: true } },
    { 'audire.settings': { enabled: true, profileId: 'L07' } },
  );
  const result = await router({
    type: MessageType.SCORE_CUE,
    payload: { cueId: 'c1', text: '지난 자막', source: 'fixture' },
  });

  assert.equal(result.ok, false, 'a stale response must not be applied to the screen');
  assert.equal(result.stale, true);
});

test('원격 주소는 설정으로 들어갈 수 없다', async () => {
  // 여기가 뚫리면 자막이 기기 밖으로 나갑니다.
  const { area, router } = build();
  for (const baseUrl of [
    'https://audire.example.com',
    'http://192.168.0.10:8000',
    'http://127.0.0.1.evil.com',
  ]) {
    const result = await router({ type: MessageType.UPDATE_SETTINGS, payload: { baseUrl } });
    assert.equal(result.ok, false, `${baseUrl} must be rejected`);
  }
  assert.equal(area.data['audire.settings'], undefined, '거절된 설정은 저장되지 않아야 한다');
});

test('범위를 벗어난 자막률은 거절된다', async () => {
  const { router } = build();
  for (const targetCaptionRate of [0, 1, 1.5, -0.2]) {
    const result = await router({
      type: MessageType.UPDATE_SETTINGS,
      payload: { targetCaptionRate },
    });
    assert.equal(result.ok, false, `${targetCaptionRate} must be rejected`);
  }
});

test('화면 식별자가 클라이언트까지 전달된다', async () => {
  // 이것이 빠지면 모든 탭이 같은 순서 카운터를 나눠 쓰고, 한 탭의 자막이 다른 탭 때문에
  // 버려집니다.
  const { client, router } = build(undefined, {
    'audire.settings': { enabled: true, profileId: 'L01' },
  });
  await router({
    type: MessageType.SCORE_CUE,
    payload: { cueId: 'c1', text: '자막', source: 'fixture', streamId: 'tab-7' },
  });
  assert.equal(client.seen.find((c) => c.call === 'score-cue').cue.streamId, 'tab-7');
});

test('설정이 바뀌면 탭에 알리되 값은 싣지 않는다', async () => {
  // 방송에 설정 값을 실으면 그 값이 페이지 위에서 도는 코드로 흘러갑니다. 콘텐츠
  // 스크립트는 다시 물어보면 됩니다.
  const broadcasts = [];
  const original = globalThis.chrome;
  globalThis.chrome = {
    tabs: {
      query: async () => [{ id: 1 }, { id: 2 }],
      sendMessage: async (tabId, message) => broadcasts.push({ tabId, message }),
    },
  };
  try {
    const { router } = build();
    await router({ type: MessageType.UPDATE_SETTINGS, payload: { profileId: 'L03' } });
    await Promise.resolve();
    await Promise.resolve();

    assert.equal(broadcasts.length, 2);
    assert.deepEqual(broadcasts[0].message, { type: MessageType.SETTINGS_CHANGED });
    assert.ok(!JSON.stringify(broadcasts).includes('L03'), 'broadcast must not carry settings');
  } finally {
    if (original === undefined) delete globalThis.chrome;
    else globalThis.chrome = original;
  }
});

test('콘텐츠 스크립트가 없는 탭이 있어도 방송이 죽지 않는다', async () => {
  const original = globalThis.chrome;
  globalThis.chrome = {
    tabs: {
      query: async () => [{ id: 1 }, { id: 2 }],
      sendMessage: async (tabId) => {
        if (tabId === 1) throw new Error('Receiving end does not exist');
      },
    },
  };
  try {
    const { router } = build();
    const result = await router({ type: MessageType.UPDATE_SETTINGS, payload: { enabled: true } });
    assert.equal(result.ok, true);
  } finally {
    if (original === undefined) delete globalThis.chrome;
    else globalThis.chrome = original;
  }
});

test('설정 저장은 나머지 값을 보존한다', async () => {
  const { router } = build();
  await router({ type: MessageType.UPDATE_SETTINGS, payload: { profileId: 'L02' } });
  const result = await router({ type: MessageType.UPDATE_SETTINGS, payload: { enabled: true } });
  assert.equal(result.settings.profileId, 'L02');
  assert.equal(result.settings.enabled, true);
});
