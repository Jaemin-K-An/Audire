/**
 * 시험용 최소 DOM.
 *
 * jsdom 을 넣지 않은 이유
 * -----------------------
 * 이 단계에서 시험해야 하는 DOM 성질은 셋뿐입니다.
 *
 * 1. `textContent` 가 자손 텍스트를 **구분자 없이** 잇는다.
 * 2. `getBoundingClientRect()` 가 프로토타입 접근자를 가진 DOMRect 를 준다.
 * 3. MutationObserver 가 **비동기로 묶여** 발화한다.
 *
 * 세 가지 모두 여기서 충실히 재현됩니다. 특히 (1)은 `childNodes` 가 실재해야 의미가
 * 있습니다 — 그래야 "자식을 공백으로 이어붙이는" 결함을 시험이 잡을 수 있습니다.
 * `textContent` 만 가진 납작한 가짜였다면 그 결함을 표현조차 할 수 없습니다.
 *
 * 실제 브라우저에서의 동작은 Phase 11 의 브라우저 E2E 가 확인합니다. 여기는 그 전에
 * 계약을 고정하는 자리입니다.
 */

const CAPTION_RECT_KEYS = ['x', 'y', 'width', 'height', 'top', 'right', 'bottom', 'left'];

/**
 * DOMRect 를 흉내 냅니다.
 *
 * 중요한 것은 좌표 값이 아니라 **좌표가 프로토타입의 접근자라는 사실**입니다. 실제
 * DOMRect 가 그렇고, 그래서 `{...rect}` 가 `{}` 가 됩니다. `toJSON` 이 있는 것도
 * 실제와 같습니다 — 직렬화만으로는 걸러지지 않는다는 점이 여기서 재현됩니다.
 */
export class FakeDOMRect {
  constructor(x, y, width, height) {
    Object.defineProperty(this, '_v', { value: { x, y, width, height }, enumerable: false });
  }

  get x() {
    return this._v.x;
  }
  get y() {
    return this._v.y;
  }
  get width() {
    return this._v.width;
  }
  get height() {
    return this._v.height;
  }
  get top() {
    return this._v.y;
  }
  get left() {
    return this._v.x;
  }
  get right() {
    return this._v.x + this._v.width;
  }
  get bottom() {
    return this._v.y + this._v.height;
  }

  toJSON() {
    return Object.fromEntries(CAPTION_RECT_KEYS.map((key) => [key, this[key]]));
  }
}

// --- MutationObserver ------------------------------------------------------

/** @type {{target: any, options: any, callback: Function, records: any[]}[]} */
const registrations = [];

/** 시험 사이에 관찰자가 새지 않도록 합니다. */
export function resetDom() {
  registrations.length = 0;
}

/** @returns {number} 현재 살아 있는 관찰 등록 수. 누수 시험이 씁니다. */
export function liveObserverCount() {
  return registrations.length;
}

function isAncestorOf(candidate, node) {
  for (let cursor = node?.parentElement; cursor; cursor = cursor.parentElement) {
    if (cursor === candidate) return true;
  }
  return false;
}

let flushQueued = false;

function notify(record) {
  let touched = false;
  for (const registration of registrations) {
    const { target, options } = registration;
    const applies =
      target === record.target || (options.subtree && isAncestorOf(target, record.target));
    if (!applies) continue;
    if (record.type === 'childList' && !options.childList) continue;
    if (record.type === 'characterData' && !options.characterData) continue;
    registration.records.push(record);
    touched = true;
  }
  if (!touched || flushQueued) return;

  // 실제 MutationObserver 는 마이크로태스크에서 **묶어서** 발화합니다. 동기로 부르면
  // 한 번에 도착하는 다발을 재현하지 못해, 중복 억제 시험이 실제보다 쉬워집니다.
  flushQueued = true;
  queueMicrotask(() => {
    flushQueued = false;
    for (const registration of [...registrations]) {
      if (registration.records.length === 0) continue;
      const batch = registration.records.splice(0);
      registration.callback(batch, registration.observer);
    }
  });
}

export class MutationObserverStub {
  constructor(callback) {
    this._callback = callback;
  }

  observe(target, options = {}) {
    registrations.push({
      target,
      options,
      callback: this._callback,
      observer: this,
      records: [],
    });
  }

  disconnect() {
    for (let i = registrations.length - 1; i >= 0; i -= 1) {
      if (registrations[i].observer === this) registrations.splice(i, 1);
    }
  }

  takeRecords() {
    const out = [];
    for (const registration of registrations) {
      if (registration.observer === this) out.push(...registration.records.splice(0));
    }
    return out;
  }
}

/** 마이크로태스크 한 바퀴. 관찰자 발화를 기다릴 때 씁니다. */
export function tick() {
  return new Promise((resolve) => queueMicrotask(resolve));
}

// --- 노드 ------------------------------------------------------------------

/** @param {string} value */
export function text(value) {
  const node = {
    nodeType: 3,
    parentElement: null,
    childNodes: [],
    _text: value,
    get textContent() {
      return node._text;
    },
    set textContent(next) {
      node._text = next;
      notify({ type: 'characterData', target: node });
    },
    get isConnected() {
      return Boolean(node.parentElement?.isConnected);
    },
  };
  return node;
}

/**
 * @param {string} tag
 * @param {{id?: string, className?: string, children?: any[]}} [options]
 */
export function element(tag, options = {}) {
  let rect = new FakeDOMRect(0, 0, 0, 0);

  const node = {
    nodeType: 1,
    tagName: tag.toUpperCase(),
    id: options.id ?? '',
    className: options.className ?? '',
    parentElement: null,
    childNodes: [],
    _isDocumentRoot: false,

    get textContent() {
      // 실제 DOM 과 같습니다: 자손 텍스트를 **구분자 없이** 잇습니다.
      return node.childNodes.map((child) => child.textContent).join('');
    },
    set textContent(value) {
      for (const child of node.childNodes) child.parentElement = null;
      node.childNodes = [];
      if (value !== '') node.append(text(value));
      else notify({ type: 'childList', target: node });
    },

    get isConnected() {
      for (let cursor = node; cursor; cursor = cursor.parentElement) {
        if (cursor._isDocumentRoot) return true;
      }
      return false;
    },

    append(...children) {
      for (const child of children) {
        child.parentElement = node;
        node.childNodes.push(child);
      }
      notify({ type: 'childList', target: node });
      return node;
    },

    replaceChildren(...children) {
      for (const child of node.childNodes) child.parentElement = null;
      node.childNodes = [];
      for (const child of children) {
        child.parentElement = node;
        node.childNodes.push(child);
      }
      notify({ type: 'childList', target: node });
      return node;
    },

    remove() {
      const parent = node.parentElement;
      if (!parent) return;
      parent.childNodes = parent.childNodes.filter((child) => child !== node);
      node.parentElement = null;
      notify({ type: 'childList', target: parent });
    },

    getBoundingClientRect() {
      return rect;
    },

    /** 시험이 배치를 정할 때 씁니다. */
    setRect(next) {
      rect = next;
      return node;
    },
  };

  if (options.children) node.append(...options.children);
  return node;
}

/**
 * `document` 를 흉내 냅니다.
 *
 * `querySelector` 는 `#id` 만 받습니다. 어댑터가 다른 형태의 선택자로 옮겨가면 여기서
 * 곧바로 터지고, 그러면 이 가짜 DOM 이 더는 충분하지 않다는 사실이 조용히 지나가지
 * 않습니다.
 */
export function createDocument(root) {
  root._isDocumentRoot = true;
  return {
    body: root,
    querySelector(selector) {
      if (!selector.startsWith('#') || /[\s>[.:]/.test(selector.slice(1))) {
        throw new Error(
          `the test DOM only supports "#id" selectors, got ${selector} — ` +
            'if the adapter needs more, this helper is no longer adequate',
        );
      }
      const id = selector.slice(1);
      const stack = [root];
      while (stack.length > 0) {
        const node = stack.pop();
        if (node.nodeType === 1 && node.id === id) return node;
        if (node.childNodes) stack.push(...node.childNodes);
      }
      return null;
    },
  };
}
