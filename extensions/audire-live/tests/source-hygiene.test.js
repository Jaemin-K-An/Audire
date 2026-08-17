/**
 * 원본 파일 위생.
 *
 * 이 시험은 실제로 일어난 사고에서 나왔습니다. Phase 6 에서 `cueFingerprint` 의 구분자
 * 자리에 **날 NUL 바이트**가 들어갔습니다. 화면에도 diff 에도 보이지 않았고, 동작은
 * 우연히 맞아서 시험 42개가 전부 통과했습니다. Phase 7 에서 그 줄에 변형을 넣으려다
 * 문자열이 일치하지 않아 드러났습니다.
 *
 * 보이지 않는 문자는 리뷰로 잡을 수 없습니다. 기계가 봐야 합니다.
 */

import assert from 'node:assert/strict';
import { readFileSync, readdirSync } from 'node:fs';
import { dirname, extname, join } from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const TEXT_EXTENSIONS = new Set(['.js', '.json', '.html', '.css', '.md']);

function sourceFiles(directory = root) {
  const out = [];
  for (const entry of readdirSync(directory, { withFileTypes: true })) {
    if (entry.name === 'node_modules' || entry.name.startsWith('.')) continue;
    const full = join(directory, entry.name);
    if (entry.isDirectory()) out.push(...sourceFiles(full));
    else if (TEXT_EXTENSIONS.has(extname(entry.name))) out.push(full);
  }
  return out;
}

test('원본에 보이지 않는 제어문자가 없다', () => {
  // 허용: 탭, 줄바꿈, 캐리지리턴. 그 밖의 C0 제어문자와 DEL 은 원본에 날것으로 들어올
  // 이유가 없습니다. 구분자로 필요하면 backslash-u-0000 처럼 escape 로 적습니다.
  const allowed = new Set(['\t', '\n', '\r']);
  const offenders = [];

  for (const file of sourceFiles()) {
    const content = readFileSync(file, 'utf8');
    for (let i = 0; i < content.length; i += 1) {
      const code = content.codePointAt(i);
      const isControl = code < 0x20 || code === 0x7f;
      if (isControl && !allowed.has(content[i])) {
        offenders.push(`${file.slice(root.length + 1)}:${i} U+${code.toString(16).padStart(4, '0')}`);
        break;
      }
    }
  }

  assert.deepEqual(offenders, [], 'raw control characters are invisible in review');
});

test('시험 파일이 실제로 훑어졌다', () => {
  // 위 시험은 파일 목록이 비어도 통과합니다. 목록이 비지 않았음을 확인합니다.
  const files = sourceFiles();
  assert.ok(files.length >= 10, `expected to scan the extension source, found ${files.length} files`);
  assert.ok(files.some((f) => f.endsWith('src/adapters/types.js')));
});
