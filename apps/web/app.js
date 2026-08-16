const $ = (selector) => document.querySelector(selector);
let stimuli = [];
let exportsByFormat = {};

function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;"
  })[character]);
}

function message(text, error = false) {
  const out = $("#message");
  out.textContent = text;
  out.style.background = error ? "#b8462c" : "#172326";
  out.classList.add("visible");
  window.setTimeout(() => out.classList.remove("visible"), 4500);
}

async function request(url, options = {}) {
  const response = await fetch(url, options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.detail || `요청 실패 (${response.status})`);
  return payload;
}

async function refresh() {
  const [health, profiles] = await Promise.all([request("/health"), request("/api/profiles")]);
  const status = $("#status");
  status.textContent = health.model_ready && health.asr_available ? "처리 준비됨" : "설정 필요";
  status.className = `pill ${health.model_ready && health.asr_available ? "ready" : "blocked"}`;
  const listener = $("#listener");
  const selected = listener.value;
  listener.innerHTML = profiles.profiles.length
    ? profiles.profiles.map((p) => `<option value="${escapeHtml(p.listener_id)}">${escapeHtml(p.listener_id)} · 교정 ${p.calibration?.n_trials || 0}회</option>`).join("")
    : '<option value="">프로필을 먼저 저장하세요</option>';
  if ([...listener.options].some((option) => option.value === selected)) listener.value = selected;
}

$("#example-profile").addEventListener("click", () => {
  $("#profile-json").value = JSON.stringify({
    listener_id: "L001", source: "manual", is_synthetic: false,
    right: {
      ear: "right",
      audiogram: {ear: "right", thresholds: {"500": {db_hl: 25}, "1000": {db_hl: 30}, "2000": {db_hl: 35}, "4000": {db_hl: 45}}},
      speech: {ear: "right", srt_db_hl: 30, wrs_percent: 84, wrs_presentation_level_db_hl: 65, wrs_word_list: "사용 목록 입력", wrs_n_words: 25}
    },
    hearing_aid_state: "unknown", notes: "직접 식별정보를 입력하지 마세요"
  }, null, 2);
});

$("#save-profile").addEventListener("click", async () => {
  try {
    const body = JSON.parse($("#profile-json").value);
    await request("/api/profiles", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)});
    await refresh();
    $("#listener").value = body.listener_id;
    message("프로필을 private 저장소에 저장했습니다.");
  } catch (error) { message(error.message, true); }
});

$("#start-calibration").addEventListener("click", async () => {
  try {
    const limit = Number($("#calibration-count").value);
    const payload = await request(`/api/calibration/stimuli?limit=${limit}`);
    stimuli = payload.stimuli;
    $("#calibration-form").innerHTML = stimuli.map((item, index) => `
      <div class="calibration-row">
        <span>${index + 1}</span>
        <button type="button" data-speak="${item.syllable}" aria-label="${index + 1}번 음절 재생">▶ 재생</button>
        <input data-response="${item.stimulus_id}" maxlength="32" placeholder="들은 음절" aria-label="${index + 1}번 응답" />
      </div>`).join("");
    $("#submit-calibration").hidden = false;
  } catch (error) { message(error.message, true); }
});

$("#calibration-form").addEventListener("click", (event) => {
  const syllable = event.target.dataset?.speak;
  if (!syllable) return;
  speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(syllable);
  utterance.lang = "ko-KR";
  speechSynthesis.speak(utterance);
});

$("#submit-calibration").addEventListener("click", async () => {
  const listenerId = $("#listener").value;
  if (!listenerId) return message("프로필을 먼저 선택하세요.", true);
  const byId = Object.fromEntries(stimuli.map((item) => [item.stimulus_id, item]));
  const trials = [...document.querySelectorAll("[data-response]")].map((input) => ({
    stimulus_id: input.dataset.response,
    target: byId[input.dataset.response].syllable,
    response: input.value.trim(),
    condition: "browser_tts"
  }));
  try {
    await request(`/api/profiles/${encodeURIComponent(listenerId)}/calibration`, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({trials})});
    await refresh();
    message(`${trials.length}개 교정 응답을 저장했습니다.`);
  } catch (error) { message(error.message, true); }
});

$("#media-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const listenerId = $("#listener").value;
  if (!listenerId) return message("프로필을 먼저 선택하세요.", true);
  const button = event.currentTarget.querySelector("button[type=submit]");
  button.disabled = true;
  try {
    const body = new FormData(event.currentTarget);
    body.set("listener_id", listenerId);
    const payload = await request("/api/process", {method: "POST", body});
    exportsByFormat = payload.exports;
    $("#metrics").innerHTML = `
      <span class="metric">전체 ${payload.summary.n_words}단어</span>
      <span class="metric">표시 ${payload.summary.n_shown}단어</span>
      <span class="metric">자막 비율 ${(payload.summary.caption_ratio * 100).toFixed(1)}%</span>`;
    $("#words").innerHTML = payload.words.map((word) => `<span class="word ${word.is_shown ? "shown" : ""}" title="청취자 위험 ${word.listener_risk.toFixed(3)} · ASR 신뢰도 ${word.asr_confidence == null ? "없음" : word.asr_confidence.toFixed(3)}">${escapeHtml(word.text)}</span>`).join("");
    $("#result").hidden = false;
    message("선택 자막을 생성했습니다.");
  } catch (error) { message(error.message, true); }
  finally { button.disabled = false; }
});

document.querySelectorAll("[data-download]").forEach((button) => button.addEventListener("click", () => {
  const format = button.dataset.download;
  const mime = format === "json" ? "application/json" : "text/plain";
  const blob = new Blob([exportsByFormat[format] || ""], {type: `${mime};charset=utf-8`});
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = `audire-captions.${format}`;
  link.click();
  URL.revokeObjectURL(link.href);
}));

refresh().catch((error) => message(error.message, true));
