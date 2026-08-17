/**
 * 매니페스트 불변식.
 *
 * 권한은 조용히 넓어집니다. 어댑터 하나를 붙이다 `<all_urls>` 를 넣고 그대로 남는 일이
 * 흔합니다. 그러면 확장은 사용자가 여는 **모든 페이지**를 읽을 수 있게 되고, 자막만
 * 읽는다는 약속은 코드 리뷰에만 남습니다. 여기서 그 경계를 시험으로 고정합니다.
 */

import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const manifest = JSON.parse(readFileSync(join(root, 'manifest.json'), 'utf8'));

test('매니페스트는 MV3 이다', () => {
  assert.equal(manifest.manifest_version, 3);
});

test('전 사이트 권한을 요구하지 않는다', () => {
  const hosts = [...(manifest.host_permissions ?? [])];
  for (const script of manifest.content_scripts ?? []) {
    hosts.push(...(script.matches ?? []));
  }
  for (const pattern of hosts) {
    assert.ok(
      !/^(<all_urls>|\*:\/\/\*\/\*|https?:\/\/\*\/\*)$/.test(pattern),
      `wildcard host permission is not allowed: ${pattern}`,
    );
  }
});

test('호스트 권한은 로컬 서버로만 나간다', () => {
  // 어댑터가 붙는 사이트는 content_scripts.matches 로 들어가야 합니다. host_permissions
  // 는 확장 자신이 fetch 하는 대상이며, 그것은 로컬 AUDIRE 하나뿐입니다.
  for (const pattern of manifest.host_permissions ?? []) {
    assert.match(pattern, /^http:\/\/(127\.0\.0\.1|localhost)\/\*$/);
  }
});

test('권한 목록은 명시적으로 승인된 것뿐이다', () => {
  // 새 권한을 넣으려면 이 목록을 고쳐야 하고, 그러면 리뷰에서 눈에 띕니다.
  const allowed = new Set(['storage', 'activeTab', 'scripting']);
  for (const permission of manifest.permissions ?? []) {
    assert.ok(allowed.has(permission), `unreviewed permission: ${permission}`);
  }
});

test('선언된 파일이 실제로 존재한다', () => {
  const referenced = [
    manifest.background?.service_worker,
    manifest.action?.default_popup,
    ...(manifest.content_scripts ?? []).flatMap((s) => [...(s.js ?? []), ...(s.css ?? [])]),
    ...Object.values(manifest.icons ?? {}),
  ].filter(Boolean);

  assert.ok(referenced.length > 0);
  for (const relative of referenced) {
    assert.ok(existsSync(join(root, relative)), `manifest references a missing file: ${relative}`);
  }
});

test('확장 CSP 를 완화하지 않는다', () => {
  const csp = manifest.content_security_policy?.extension_pages ?? '';
  assert.ok(!csp.includes('unsafe-eval'), 'unsafe-eval must not be enabled');
  assert.ok(!csp.includes('unsafe-inline'), 'unsafe-inline must not be enabled');
});

test('팝업에 인라인 스크립트가 없다', () => {
  // MV3 CSP 는 인라인 실행을 막습니다. 인라인 핸들러가 들어가면 팝업이 조용히 죽습니다.
  const html = readFileSync(join(root, manifest.action.default_popup), 'utf8');
  assert.ok(!/<script(?![^>]*\ssrc=)[^>]*>[\s\S]*?\S[\s\S]*?<\/script>/i.test(html));
  assert.ok(!/\son[a-z]+\s*=/i.test(html), 'inline event handlers are blocked by MV3 CSP');
});

test('의료기기가 아니라는 사실이 사용자에게 보인다', () => {
  const html = readFileSync(join(root, manifest.action.default_popup), 'utf8');
  assert.match(manifest.description, /의료기기가 아닙니다/);
  assert.match(html, /의료기기가 아닙니다/);
});
