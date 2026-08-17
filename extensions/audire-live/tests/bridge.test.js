/**
 * 로컬 서버로 가는 다리의 규칙.
 *
 * 가장 중요한 것은 **순서**입니다. 라이브 자막에서 요청은 겹치고 응답은 순서를 지키지
 * 않습니다. 그 처리가 없으면 화면이 과거 자막으로 되돌아갑니다.
 */

import assert from 'node:assert/strict';
import test from 'node:test';

import { AudireClient, LiveState } from '../src/bridge/audireClient.js';

/** 미리 정해둔 응답을 돌려주는 fetch. 호출 기록을 남깁니다. */
function stubFetch(responder) {
  const calls = [];
  const impl = async (url, init) => {
    calls.push({ url, init });
    return responder(url, init, calls.length);
  };
  impl.calls = calls;
  return impl;
}

function jsonResponse(body, { ok = true, status = 200 } = {}) {
  return { ok, status, json: async () => body };
}

test('토큰은 헤더로만 나가고 URL 에 실리지 않는다', async () => {
  const fetchImpl = stubFetch(() => jsonResponse({ server: 'ok' }));
  const client = new AudireClient({ token: 'secret-token', fetchImpl });
  await client.status();

  const { url, init } = fetchImpl.calls[0];
  assert.ok(!url.includes('secret-token'), 'token must never appear in the URL');
  assert.equal(init.headers['x-audire-token'], 'secret-token');
});

test('토큰이 없으면 헤더도 붙지 않는다', async () => {
  const fetchImpl = stubFetch(() => jsonResponse({ server: 'ok' }));
  await new AudireClient({ fetchImpl }).status();
  assert.ok(!('x-audire-token' in fetchImpl.calls[0].init.headers));
});

test('서버가 꺼져 있으면 오류가 아니라 상태로 돌아온다', async () => {
  const fetchImpl = stubFetch(() => {
    throw new TypeError('Failed to fetch');
  });
  const result = await new AudireClient({ fetchImpl }).status();
  assert.equal(result.state, LiveState.SERVER_OFFLINE);
});

test('실패 사유가 서로 구분된다', async () => {
  // "안 됨" 하나로 뭉치면 사용자가 서버를 켜야 할지, 페어링을 해야 할지, 교정을 해야
  // 할지 알 수 없습니다.
  const cases = [
    [401, 'not_paired', LiveState.NOT_PAIRED],
    [404, 'unknown_profile', LiveState.UNKNOWN_PROFILE],
    [409, 'profile_not_ready', LiveState.PROFILE_NOT_READY],
    [409, 'contract_mismatch', LiveState.CONTRACT_MISMATCH],
    [503, 'model_unavailable', LiveState.MODEL_UNAVAILABLE],
    [422, 'invalid_cue', LiveState.INVALID_CUE],
  ];
  for (const [status, reason, expected] of cases) {
    const fetchImpl = stubFetch(() =>
      jsonResponse({ detail: { reason, message: '…' } }, { ok: false, status }),
    );
    const result = await new AudireClient({ fetchImpl }).status();
    assert.equal(result.state, expected, `${reason} should map to ${expected}`);
  }
});

test('모르는 사유는 unknown_error 가 된다', async () => {
  const fetchImpl = stubFetch(() =>
    jsonResponse({ detail: { reason: 'something_new' } }, { ok: false, status: 500 }),
  );
  const result = await new AudireClient({ fetchImpl }).status();
  assert.equal(result.state, LiveState.UNKNOWN_ERROR);
});

test('본문이 JSON 이 아니어도 죽지 않는다', async () => {
  const fetchImpl = stubFetch(() => ({
    ok: false,
    status: 502,
    json: async () => {
      throw new SyntaxError('not json');
    },
  }));
  const result = await new AudireClient({ fetchImpl }).status();
  assert.equal(result.state, LiveState.UNKNOWN_ERROR);
});

test('늦게 도착한 옛 응답은 stale 로 표시된다', async () => {
  // 큐 1 이 느리고 큐 2 가 빠른 상황. 큐 1 의 응답이 화면을 되돌리면 안 됩니다.
  let releaseFirst;
  const gate = new Promise((resolve) => {
    releaseFirst = resolve;
  });
  const fetchImpl = stubFetch(async (_url, _init, n) => {
    if (n === 1) await gate;
    return jsonResponse({ cue_id: `cue-${n}` });
  });

  const client = new AudireClient({ fetchImpl });
  const first = client.scoreCue({ profileId: 'p', cueId: 'cue-1', text: '첫째', source: 'fixture' });
  const second = await client.scoreCue({
    profileId: 'p',
    cueId: 'cue-2',
    text: '둘째',
    source: 'fixture',
  });
  releaseFirst();
  const firstResult = await first;

  assert.equal(second.stale, false);
  assert.equal(firstResult.stale, true, 'the older cue must be marked stale');
});

test('순서대로 도착하면 stale 이 아니다', async () => {
  const fetchImpl = stubFetch((_url, _init, n) => jsonResponse({ cue_id: `cue-${n}` }));
  const client = new AudireClient({ fetchImpl });
  const a = await client.scoreCue({ profileId: 'p', cueId: 'a', text: '가', source: 'fixture' });
  const b = await client.scoreCue({ profileId: 'p', cueId: 'b', text: '나', source: 'fixture' });
  assert.equal(a.stale, false);
  assert.equal(b.stale, false);
});

test('목표 자막률은 지정했을 때만 보낸다', async () => {
  const fetchImpl = stubFetch(() => jsonResponse({}));
  const client = new AudireClient({ fetchImpl });

  await client.scoreCue({ profileId: 'p', cueId: 'a', text: '가', source: 'fixture' });
  assert.ok(!('target_caption_rate' in JSON.parse(fetchImpl.calls[0].init.body)));

  await client.scoreCue({
    profileId: 'p',
    cueId: 'b',
    text: '나',
    source: 'fixture',
    targetCaptionRate: 0.35,
  });
  assert.equal(JSON.parse(fetchImpl.calls[1].init.body).target_caption_rate, 0.35);
});
