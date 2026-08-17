/**
 * 콘텐츠 스크립트 진입점.
 *
 * 매니페스트가 선언하는 콘텐츠 스크립트는 **평범한 스크립트**이며 `import` 문을 쓸 수
 * 없습니다. 그렇다고 번들러를 들이면 "브라우저에 실린 코드 == 시험한 코드" 가 깨집니다
 * (ADR-0022).
 *
 * 그래서 이 파일만 평범한 스크립트로 두고, 실제 코드는 동적 import 로 불러옵니다.
 * 불러오는 파일은 Node 시험이 읽는 것과 같은 파일입니다.
 */

(async () => {
  try {
    const url = chrome.runtime.getURL('src/content.js');
    const module = await import(url);
    await module.bootstrap();
  } catch (error) {
    // 페이지 위에서 도는 코드입니다. 여기서 던지면 사이트 콘솔에 확장의 오류가 쌓입니다.
    // 자막 내용은 아직 읽기 전이므로 이 메시지에 실릴 것이 없습니다.
    console.warn('[AUDIRE] content script failed to start:', error?.message ?? error);
  }
})();
