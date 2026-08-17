/**
 * 픽스처 조작기.
 *
 * 자막 DOM 을 결정적으로 흔듭니다. 실제 자막 플랫폼이 하는 짓들을 재현하는 것이
 * 목적입니다 — 한 문장을 span 여러 개로 쪼개고, 같은 문장을 유지한 채 DOM 만 계속
 * 바꾸고, 컨테이너를 통째로 다시 만들고.
 *
 * 여기에는 확장 코드가 없습니다. 이 파일은 관찰 **대상**이지 관찰자가 아닙니다.
 */

const ROOT_ID = 'audire-fixture-subtitles';

const state = document.getElementById('fixture-state');

function report(message) {
  // 페이지 자신의 상태 표시입니다. 확장의 로그가 아닙니다.
  if (state) state.textContent = message;
}

function root() {
  return document.getElementById(ROOT_ID);
}

function requireRoot() {
  const node = root();
  if (!node) throw new Error('자막 뿌리가 없습니다. remount 를 먼저 부르십시오.');
  return node;
}

/** 문장 하나를 평범한 텍스트로 넣습니다. */
function setSubtitle(text) {
  requireRoot().replaceChildren(document.createTextNode(text));
  report(`자막 설정: ${text.length}자`);
}

/**
 * 문장을 글자 단위 span 으로 쪼개 넣습니다.
 *
 * 자막 플랫폼이 실제로 하는 일이고, 여기서 "안녕"이 "안 녕"이 되면 안 된다는 규칙이
 * 실물로 확인됩니다.
 */
function setSubtitleSpans(text, groupSize = 1) {
  const node = requireRoot();
  const spans = [];
  for (let i = 0; i < text.length; i += groupSize) {
    const span = document.createElement('span');
    span.textContent = text.slice(i, i + groupSize);
    spans.push(span);
  }
  node.replaceChildren(...spans);
  report(`자막 설정: span ${spans.length}개, ${text.length}자`);
}

/**
 * 보이는 문장은 그대로 두고 DOM 만 바꿉니다.
 *
 * 쪼개는 크기를 매번 달리해 구조를 바꾸되 `textContent` 는 불변입니다. 관찰자가
 * DOM 변경이 아니라 **보이는 텍스트**로 중복을 판정하는지 여기서 갈립니다.
 */
function churn(text, times = 20) {
  for (let i = 0; i < times; i += 1) {
    setSubtitleSpans(text, (i % 3) + 1);
  }
  report(`같은 문장으로 DOM 을 ${times}번 변경했습니다`);
}

function clear() {
  requireRoot().replaceChildren();
  report('자막을 비웠습니다');
}

/** 뿌리를 통째로 없앱니다. SPA 가 컨테이너를 파괴하는 상황입니다. */
function removeRoot() {
  root()?.remove();
  report('자막 뿌리를 제거했습니다');
}

/** 같은 id 의 새 뿌리를 다시 답니다. 이전 노드와 다른 객체입니다. */
function remountRoot() {
  removeRoot();
  const fresh = document.createElement('div');
  fresh.id = ROOT_ID;
  fresh.className = 'subtitles';
  fresh.setAttribute('aria-live', 'polite');
  document.getElementById('fixture-stage').append(fresh);
  report('자막 뿌리를 다시 만들었습니다');
  return fresh;
}

/** 화면에 나오는 한국어는 전부 이 시험을 위해 지어낸 문장입니다. */
const SCENARIOS = {
  none: () => clear(),
  simple: () => setSubtitle('오늘 병원에 갑니다'),
  nested: () => setSubtitleSpans('안녕'),
  multi: () => setSubtitleSpans('내일 학교에서 만납시다', 2),
  churn: () => churn('오늘 병원에 갑니다'),
  replace: () => setSubtitle('내일 병원에 갑니다'),
  clear: () => clear(),
  remount: () => {
    remountRoot();
    setSubtitle('뿌리를 다시 만든 뒤의 자막입니다');
  },
};

for (const button of document.querySelectorAll('[data-scenario]')) {
  button.addEventListener('click', () => {
    try {
      SCENARIOS[button.dataset.scenario]();
    } catch (error) {
      report(String(error.message ?? error));
    }
  });
}

/** Phase 11 의 브라우저 E2E 가 부르는 통로. */
window.audireFixture = {
  ROOT_ID,
  setSubtitle,
  setSubtitleSpans,
  churn,
  clear,
  removeRoot,
  remountRoot,
  scenario: (name) => SCENARIOS[name](),
};

report('준비됨');
