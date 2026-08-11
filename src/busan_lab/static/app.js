const state = {
  record: null,
  analysis: null,
  records: [],
  predictions: [],
  predictionDiagnostics: new Map(),
  reviews: [],
  benchmarks: [],
  trainingImports: [],
  trainingReviewQueue: null,
  trainingReviewPromptId: null,
  trainingReviewBusy: false,
};

const $ = (selector) => document.querySelector(selector);

async function requestJson(url, options = {}) {
  let response;
  try {
    response = await fetch(url, options);
  } catch {
    throw new Error(
      "로컬 엔진에 연결할 수 없습니다. busan-lab serve를 실행한 뒤 다시 시도하세요.",
    );
  }
  const contentType = response.headers.get("content-type") || "";
  const payload = contentType.includes("application/json")
    ? await response.json()
    : await response.text();
  if (!response.ok) {
    const detail = payload?.detail || payload || `HTTP ${response.status}`;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return payload;
}

function setMessage(element, message, isError = false) {
  element.textContent = message;
  element.classList.toggle("is-error", isError);
}

async function initialize() {
  const status = $("#connection-status");
  setupSidebarNavigation();
  try {
    await requestJson("/api/health");
    status.classList.add("is-online");
    status.lastChild.textContent = " 로컬 엔진 연결됨";
  } catch {
    status.classList.remove("is-online");
    status.lastChild.textContent = " 엔진 연결 실패";
  }
  drawEmptySignals();
  try {
    await refreshUtteranceOptions();
  } catch (error) {
    setMessage($("#evaluation-message"), `저장 발화 조회 실패: ${error.message}`, true);
  }
  try {
    await loadTrainingReviewImports();
  } catch (error) {
    $("#training-review").hidden = false;
    setMessage(
      $("#training-review-message"),
      `TASK-004 검수 목록 조회 실패: ${error.message}`,
      true,
    );
  }
  try {
    await loadBenchmarks();
  } catch (error) {
    setMessage($("#benchmark-message"), `Benchmark 조회 실패: ${error.message}`, true);
  }
}

async function refreshUtteranceOptions(preferredUtteranceId = null) {
  const records = await requestJson("/api/utterances");
  state.records = records.sort(
    (left, right) => new Date(right.created_at) - new Date(left.created_at),
  );
  const select = $("#utterance-select");
  const selectedId =
    preferredUtteranceId || state.record?.utterance_id || state.records[0]?.utterance_id || "";
  select.replaceChildren();
  if (!state.records.length) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "저장된 발화 없음";
    select.append(option);
    select.disabled = true;
    await loadPredictions();
    return;
  }
  state.records.forEach((record) => {
    const option = document.createElement("option");
    option.value = record.utterance_id;
    option.textContent =
      `${record.ground_truth.surface_text} · ${record.speaker.speaker_id}`;
    select.append(option);
  });
  select.disabled = false;
  select.value = selectedId;
  const selectedRecord =
    state.records.find((record) => record.utterance_id === select.value) || state.records[0];
  if (!state.record || state.record.utterance_id !== selectedRecord.utterance_id) {
    await selectRecord(selectedRecord);
  }
}

async function selectRecord(record) {
  state.record = record;
  renderRecord(record);
  const analysis = await requestJson(`/api/utterances/${record.utterance_id}/analysis`);
  state.analysis = analysis;
  renderSignals(analysis);
  await loadPredictions();
}

$("#utterance-select").addEventListener("change", async (event) => {
  const record = state.records.find((item) => item.utterance_id === event.target.value);
  if (!record) return;
  try {
    await selectRecord(record);
    syncTrainingReviewToUtterance(record.utterance_id);
    setMessage($("#evaluation-message"), "저장된 발화와 모델 기록을 불러왔습니다.");
  } catch (error) {
    setMessage($("#evaluation-message"), error.message, true);
  }
});

const trainingReviewStatusLabels = {
  candidate: "미검수",
  human_reviewed: "검수됨",
  approved: "승인",
  deprecated: "재녹음 필요",
};

async function loadTrainingReviewImports() {
  const imports = await requestJson("/api/training-imports");
  state.trainingImports = imports;
  const panel = $("#training-review");
  if (!imports.length) {
    panel.hidden = true;
    return;
  }

  panel.hidden = false;
  const select = $("#training-import-select");
  const preferredImportId =
    state.trainingReviewQueue?.import_id || imports[0].import_id;
  select.replaceChildren();
  imports.forEach((item) => {
    const option = document.createElement("option");
    option.value = item.import_id;
    option.textContent = `${item.import_id} (${item.entry_count}개)`;
    select.append(option);
  });
  select.value = preferredImportId;
  await loadTrainingReviewQueue(select.value, null, true);
}

async function loadTrainingReviewQueue(
  importId,
  preferredPromptId = null,
  selectCurrentRecord = false,
) {
  const queue = await requestJson(
    `/api/training-imports/${encodeURIComponent(importId)}/review-queue`,
  );
  state.trainingReviewQueue = queue;
  const preferredItem = queue.items.find(
    (item) => item.prompt_id === preferredPromptId,
  );
  const currentItem = queue.items.find(
    (item) => item.prompt_id === state.trainingReviewPromptId,
  );
  const firstUnreviewed = queue.items.find((item) => item.label_status === "candidate");
  const selectedItem = preferredItem || currentItem || firstUnreviewed || queue.items[0];
  state.trainingReviewPromptId = selectedItem.prompt_id;
  renderTrainingReviewQueue();
  if (selectCurrentRecord) await selectTrainingReviewItem(selectedItem.prompt_id);
}

function renderTrainingReviewQueue() {
  const queue = state.trainingReviewQueue;
  if (!queue) return;
  $("#training-review-progress").textContent =
    `${queue.reviewed_count} / ${queue.total_count}`;
  $("#training-review-approved").textContent = queue.approved_count;
  $("#training-review-rerecord").textContent = queue.rerecord_count;
  $("#training-review-status").textContent = queue.candidate_count
    ? `${queue.candidate_count}개 남음`
    : "검수 완료";

  const select = $("#training-review-item-select");
  select.replaceChildren();
  queue.items.forEach((item) => {
    const option = document.createElement("option");
    const status = trainingReviewStatusLabels[item.label_status] || item.label_status;
    option.value = item.prompt_id;
    option.textContent = `${item.prompt_id} | ${status} | ${item.surface_text}`;
    select.append(option);
  });
  renderTrainingReviewCurrent();
}

function renderTrainingReviewCurrent() {
  const queue = state.trainingReviewQueue;
  if (!queue) return;
  const index = queue.items.findIndex(
    (item) => item.prompt_id === state.trainingReviewPromptId,
  );
  const item = queue.items[index];
  if (!item) return;
  const statusLabel = trainingReviewStatusLabels[item.label_status] || item.label_status;
  $("#training-review-position").textContent =
    `${item.position} / ${queue.total_count}`;
  $("#training-review-prompt-id").textContent = item.prompt_id;
  const status = $("#training-review-item-status");
  status.textContent = statusLabel;
  status.dataset.status = item.label_status;
  $("#training-review-item-select").value = item.prompt_id;
  $("#training-review-candidate").textContent =
    `수집 문장: ${item.candidate_surface_text}`;
  $("#training-review-previous").disabled = state.trainingReviewBusy || index <= 0;
  $("#training-review-next").disabled =
    state.trainingReviewBusy || index >= queue.items.length - 1;
  $("#training-review-edit").disabled = state.trainingReviewBusy;
  $("#training-review-approve").disabled =
    state.trainingReviewBusy || item.label_status === "approved";
  $("#training-review-rerecord-button").disabled =
    state.trainingReviewBusy || item.label_status === "deprecated";
  $("#training-import-select").disabled = state.trainingReviewBusy;
  $("#training-review-item-select").disabled = state.trainingReviewBusy;
  const deleteButton = $("#delete-utterance");
  deleteButton.disabled = true;
  deleteButton.title = "TASK-004 Import 목록에 연결된 발화는 삭제할 수 없습니다.";
}

async function selectTrainingReviewItem(promptId) {
  const queue = state.trainingReviewQueue;
  const item = queue?.items.find((candidate) => candidate.prompt_id === promptId);
  if (!item) return;
  state.trainingReviewPromptId = promptId;
  renderTrainingReviewCurrent();
  let record = state.records.find(
    (candidate) => candidate.utterance_id === item.utterance_id,
  );
  if (!record) {
    record = await requestJson(`/api/utterances/${item.utterance_id}`);
    state.records.push(record);
  }
  $("#utterance-select").value = record.utterance_id;
  await selectRecord(record);
}

function syncTrainingReviewToUtterance(utteranceId) {
  const item = state.trainingReviewQueue?.items.find(
    (candidate) => candidate.utterance_id === utteranceId,
  );
  if (!item) {
    const deleteButton = $("#delete-utterance");
    deleteButton.disabled = false;
    deleteButton.removeAttribute("title");
    return;
  }
  state.trainingReviewPromptId = item.prompt_id;
  renderTrainingReviewCurrent();
}

async function moveTrainingReview(direction) {
  const queue = state.trainingReviewQueue;
  if (!queue) return;
  const index = queue.items.findIndex(
    (item) => item.prompt_id === state.trainingReviewPromptId,
  );
  const target = queue.items[index + direction];
  if (!target) return;
  try {
    await selectTrainingReviewItem(target.prompt_id);
    setMessage($("#training-review-message"), "");
  } catch (error) {
    setMessage($("#training-review-message"), error.message, true);
  }
}

function nextUnreviewedItem(queue, currentPromptId) {
  const currentIndex = queue.items.findIndex(
    (item) => item.prompt_id === currentPromptId,
  );
  return (
    queue.items.slice(currentIndex + 1).find(
      (item) => item.label_status === "candidate",
    ) || queue.items.find((item) => item.label_status === "candidate")
  );
}

async function submitTrainingReviewDecision(decision) {
  const queue = state.trainingReviewQueue;
  const promptId = state.trainingReviewPromptId;
  if (!queue || !promptId) return;
  const reviewerId = $("#label-changed-by").value.trim();
  if (!reviewerId) {
    setMessage($("#training-review-message"), "수정자 ID를 입력하세요.", true);
    $("#label-changed-by").focus();
    return;
  }

  state.trainingReviewBusy = true;
  renderTrainingReviewCurrent();
  setMessage($("#training-review-message"), "검수 이력을 저장하는 중입니다.");
  try {
    const record = await requestJson(
      `/api/training-imports/${encodeURIComponent(queue.import_id)}` +
        `/review-queue/${encodeURIComponent(promptId)}/decision`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          reviewer_id: reviewerId,
          decision,
          notes: $("#label-reason").value.trim() || null,
        }),
      },
    );
    const recordIndex = state.records.findIndex(
      (item) => item.utterance_id === record.utterance_id,
    );
    if (recordIndex >= 0) state.records[recordIndex] = record;
    state.record = record;
    $("#label-reason").value = "";
    await loadTrainingReviewQueue(queue.import_id, promptId, false);
    const nextItem = nextUnreviewedItem(state.trainingReviewQueue, promptId);
    if (nextItem) await selectTrainingReviewItem(nextItem.prompt_id);
    else await selectTrainingReviewItem(promptId);
    const resultLabel = decision === "approve" ? "승인" : "재녹음 필요";
    setMessage(
      $("#training-review-message"),
      `${promptId}을 ${resultLabel}로 저장했습니다.`,
    );
  } catch (error) {
    setMessage($("#training-review-message"), error.message, true);
  } finally {
    state.trainingReviewBusy = false;
    renderTrainingReviewCurrent();
  }
}

$("#training-import-select").addEventListener("change", async (event) => {
  try {
    state.trainingReviewPromptId = null;
    await loadTrainingReviewQueue(event.target.value, null, true);
    setMessage($("#training-review-message"), "검수 목록을 불러왔습니다.");
  } catch (error) {
    setMessage($("#training-review-message"), error.message, true);
  }
});

$("#training-review-item-select").addEventListener("change", async (event) => {
  try {
    await selectTrainingReviewItem(event.target.value);
    setMessage($("#training-review-message"), "");
  } catch (error) {
    setMessage($("#training-review-message"), error.message, true);
  }
});

$("#training-review-previous").addEventListener("click", () => {
  void moveTrainingReview(-1);
});

$("#training-review-next").addEventListener("click", () => {
  void moveTrainingReview(1);
});

$("#training-review-edit").addEventListener("click", () => {
  $("#label-surface-text").focus();
  $("#label-edit-form").scrollIntoView({ block: "nearest" });
});

$("#training-review-approve").addEventListener("click", () => {
  void submitTrainingReviewDecision("approve");
});

$("#training-review-rerecord-button").addEventListener("click", () => {
  void submitTrainingReviewDecision("rerecord");
});

function setupSidebarNavigation() {
  const links = [...document.querySelectorAll(".sidebar-nav a")];
  const items = links
    .map((link) => ({
      link,
      target: document.querySelector(link.getAttribute("href")),
    }))
    .filter((item) => item.target);
  let scheduled = false;

  const select = (selectedLink) => {
    links.forEach((link) => {
      if (link === selectedLink) link.setAttribute("aria-current", "location");
      else link.removeAttribute("aria-current");
    });
  };

  const update = () => {
    const threshold = window.innerWidth <= 840 ? 132 : 40;
    let current = items[0];
    items.forEach((item) => {
      if (item.target.getBoundingClientRect().top <= threshold) current = item;
    });
    if (current) select(current.link);
    scheduled = false;
  };

  links.forEach((link) => {
    link.addEventListener("click", () => select(link));
  });
  window.addEventListener(
    "scroll",
    () => {
      if (!scheduled) {
        scheduled = true;
        window.requestAnimationFrame(update);
      }
    },
    { passive: true },
  );
  update();
}

const dropZone = $("#drop-zone");
const fileInput = $("#audio-file");

["dragenter", "dragover"].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.add("is-dragging");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.remove("is-dragging");
  });
});

dropZone.addEventListener("drop", (event) => {
  if (event.dataTransfer.files.length) {
    fileInput.files = event.dataTransfer.files;
    updateFileLabel();
  }
});

fileInput.addEventListener("change", updateFileLabel);

function updateFileLabel() {
  const file = fileInput.files[0];
  $("#file-label").textContent = file
    ? `${file.name} · ${(file.size / 1024 / 1024).toFixed(2)} MB`
    : "음성 파일 선택";
}

function parseQuotedTerms(value) {
  const text = value.trim();
  if (!text) return [];
  const quotedPattern = /["“]([^"”]+)["”]/g;
  const terms = [...text.matchAll(quotedPattern)]
    .map((match) => match[1].trim())
    .filter(Boolean);
  if (!terms.length) return [text];
  const remainder = text
    .replace(quotedPattern, "")
    .replace(/[\s,，]+/g, "");
  if (remainder) {
    throw new Error(
      '여러 Surface 표현은 각각 쌍따옴표로 감싸세요. 예: "와따", "맛있노"',
    );
  }
  return terms;
}

function parseNormalizedTerms(value) {
  const text = value.trim();
  if (!text) return [];
  const quotedTerms = parseQuotedTerms(text);
  if (quotedTerms.length > 1 || /^["“]/.test(text)) return quotedTerms;
  return text
    .split(",")
    .map((term) => term.trim())
    .filter(Boolean);
}

function buildDialectLabels(surfaceValue, normalizedValue) {
  const surfaces = parseQuotedTerms(surfaceValue);
  if (!surfaces.length) return [];
  const normalizedTerms = parseNormalizedTerms(normalizedValue);
  if (surfaces.length === 1) {
    return [{
      surface_form: surfaces[0],
      normalized_forms: normalizedTerms,
      status: "candidate",
    }];
  }
  if (normalizedTerms.length && normalizedTerms.length !== surfaces.length) {
    throw new Error(
      `Surface 표현 ${surfaces.length}개에 맞춰 표준화 후보도 ${surfaces.length}개를 입력하세요.`,
    );
  }
  return surfaces.map((surface, index) => ({
    surface_form: surface,
    normalized_forms: normalizedTerms.length ? [normalizedTerms[index]] : [],
    status: "candidate",
  }));
}

$("#upload-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const form = event.currentTarget;
  const button = form.querySelector("button[type=submit]");
  const message = $("#upload-message");
  const formData = new FormData(form);
  let labels;
  try {
    labels = buildDialectLabels(
      $("#dialect-surface").value,
      $("#dialect-normalized").value,
    );
  } catch (error) {
    setMessage(message, error.message, true);
    return;
  }
  formData.set("dialect_expressions", JSON.stringify(labels));
  for (const name of [
    "storage_allowed",
    "research_use_allowed",
    "model_training_allowed",
  ]) {
    formData.set(name, form.elements[name].checked ? "true" : "false");
  }

  button.disabled = true;
  setMessage(
    message,
    "원본을 보존하고 48kHz 마스터와 ASR·발음·TTS용 WAV를 생성합니다…",
  );
  try {
    const record = await requestJson("/api/utterances", {
      method: "POST",
      body: formData,
    });
    state.record = record;
    renderRecord(record);
    setMessage(
      message,
      "원본·마스터·용도별 파생본을 보존했습니다. 음향 특징을 계산합니다…",
    );
    const analysis = await requestJson(
      `/api/utterances/${record.utterance_id}/analysis`,
    );
    state.analysis = analysis;
    renderSignals(analysis);
    await refreshUtteranceOptions(record.utterance_id);
    await loadPredictions();
    setMessage(message, "발화 증거선이 준비되었습니다.");
    $("#hypothesis").focus();
  } catch (error) {
    setMessage(message, error.message, true);
  } finally {
    button.disabled = false;
  }
});

function renderRecord(record) {
  const quality = record.audio.quality;
  const master = record.audio.master;
  const derivatives = Object.fromEntries(
    record.audio.derivatives.map((asset) => [asset.role, asset]),
  );
  const asr = derivatives.asr_16k_mono || record.audio.derived;
  const pronunciation = derivatives.pronunciation_24k_mono;
  const tts = derivatives.tts_48k_mono;
  const entries = [
    record.utterance_id,
    record.audio.original.sha256,
    master?.sha256 || "legacy record - master 없음",
    asr.sha256,
  ];
  $("#lineage-list").querySelectorAll("dd").forEach((element, index) => {
    element.textContent = entries[index];
    element.title = entries[index];
  });
  const badge = $("#quality-badge");
  badge.textContent = quality.passed ? "정상" : "검토 필요";
  badge.classList.toggle("is-pass", quality.passed);
  badge.classList.toggle("is-warn", !quality.passed);
  $("#metric-rms").textContent = quality.rms_dbfs.toFixed(1);
  $("#metric-clip").textContent = quality.clipping_ratio.toFixed(4);
  $("#metric-silence").textContent = quality.silence_ratio.toFixed(3);
  $("#quality-warnings").textContent = quality.warnings.length
    ? quality.warnings.join(" · ")
    : "품질 임계값을 모두 통과했습니다.";
  const channelLabel = (asset) =>
    asset.channels === 1 ? "mono" : asset.channels === 2 ? "stereo" : `${asset.channels}ch`;
  const formatLabel = (asset) =>
    `${asset.sample_rate_hz / 1000}kHz ${channelLabel(asset)} · ${asset.codec}`;
  $("#audio-tree").textContent = master
    ? [
        `원본 마스터 음원 · ${formatLabel(master)}`,
        `├── ASR 학습용 · ${formatLabel(asr)}`,
        `├── 발음 평가용 · ${formatLabel(pronunciation)}`,
        `└── TTS 학습용 · ${formatLabel(tts)}`,
      ].join("\n")
    : `Legacy 원본\n└── ASR 학습용 · ${formatLabel(asr)}`;
  $("#audio-player").src =
    `/api/utterances/${record.utterance_id}/audio/${asr.role}`;
  $("#signal-meta").textContent =
    `${(asr.duration_ms / 1000).toFixed(2)}s · ` +
    `${record.ground_truth.surface_text}`;
  renderLabelEditor(record);
  syncTrainingReviewToUtterance(record.utterance_id);
}

function addLabelExpressionRow(expression = null) {
  const row = document.createElement("div");
  row.className = "label-expression-row";

  const surfaceLabel = document.createElement("label");
  const surfaceTitle = document.createElement("span");
  surfaceTitle.textContent = "Surface 표현";
  const surfaceInput = document.createElement("input");
  surfaceInput.className = "label-expression-surface";
  surfaceInput.placeholder = '예: "와따", "맛있노"';
  surfaceInput.value = expression?.surface_form || "";
  surfaceLabel.append(surfaceTitle, surfaceInput);

  const normalizedLabel = document.createElement("label");
  const normalizedTitle = document.createElement("span");
  normalizedTitle.textContent = "표준화 후보";
  const normalizedInput = document.createElement("input");
  normalizedInput.className = "label-expression-normalized";
  normalizedInput.placeholder = "예: 주세요, 주십시오";
  normalizedInput.value = expression?.normalized_forms?.join(", ") || "";
  normalizedLabel.append(normalizedTitle, normalizedInput);

  const removeButton = document.createElement("button");
  removeButton.type = "button";
  removeButton.className = "label-expression-remove";
  removeButton.textContent = "삭제";
  removeButton.addEventListener("click", () => row.remove());
  row.append(surfaceLabel, normalizedLabel, removeButton);
  $("#label-expression-list").append(row);
}

function renderLabelEditor(record) {
  const groundTruth = record.ground_truth;
  $("#label-surface-text").value = groundTruth.surface_text;
  $("#label-normalized-meaning").value = groundTruth.normalized_meaning || "";
  $("#label-editor-version").textContent = groundTruth.label_version;
  const list = $("#label-expression-list");
  list.replaceChildren();
  groundTruth.dialect_expressions.forEach((expression) => {
    addLabelExpressionRow(expression);
  });
  if (!groundTruth.dialect_expressions.length) addLabelExpressionRow();
  setMessage($("#label-edit-message"), "");
}

$("#add-label-expression").addEventListener("click", () => {
  addLabelExpressionRow();
});

$("#label-edit-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = $("#label-edit-message");
  if (!state.record) {
    setMessage(message, "수정할 저장 발화를 먼저 선택하세요.", true);
    return;
  }
  let expressions;
  try {
    expressions = [...document.querySelectorAll(".label-expression-row")]
      .flatMap((row) => buildDialectLabels(
        row.querySelector(".label-expression-surface").value,
        row.querySelector(".label-expression-normalized").value,
      ));
  } catch (error) {
    setMessage(message, error.message, true);
    return;
  }
  const button = event.currentTarget.querySelector("button[type=submit]");
  button.disabled = true;
  try {
    const record = await requestJson(
      `/api/utterances/${state.record.utterance_id}/labels`,
      {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          surface_text: $("#label-surface-text").value.trim(),
          normalized_meaning: $("#label-normalized-meaning").value.trim() || null,
          dialect_expressions: expressions,
          changed_by: $("#label-changed-by").value.trim(),
          reason: $("#label-reason").value.trim() || null,
        }),
      },
    );
    state.record = record;
    const recordIndex = state.records.findIndex(
      (item) => item.utterance_id === record.utterance_id,
    );
    if (recordIndex >= 0) state.records[recordIndex] = record;
    renderRecord(record);
    $("#label-reason").value = "";
    await refreshUtteranceOptions(record.utterance_id);
    if (state.trainingReviewQueue) {
      await loadTrainingReviewQueue(
        state.trainingReviewQueue.import_id,
        state.trainingReviewPromptId,
        false,
      );
    }
    setMessage(
      message,
      `${record.ground_truth.label_version}으로 수정하고 이전 라벨을 보존했습니다.`,
    );
  } catch (error) {
    setMessage(message, error.message, true);
  } finally {
    button.disabled = false;
  }
});

$("#delete-utterance").addEventListener("click", async (event) => {
  const message = $("#label-edit-message");
  if (!state.record) {
    setMessage(message, "삭제할 저장 발화를 먼저 선택하세요.", true);
    return;
  }
  const surfaceText = state.record.ground_truth.surface_text;
  const confirmed = window.confirm(
    `“${surfaceText}” 발화와 연결된 분석 결과를 삭제할까요?\n` +
      "삭제된 파일은 data/lab/trash에 보관됩니다.",
  );
  if (!confirmed) return;

  event.currentTarget.disabled = true;
  try {
    const result = await requestJson(`/api/utterances/${state.record.utterance_id}`, {
      method: "DELETE",
    });
    state.record = null;
    state.analysis = null;
    await refreshUtteranceOptions();
    if (!state.records.length) clearRecordView();
    setMessage(
      message,
      `발화를 삭제했습니다. 복구 ID: ${result.archive_id}`,
    );
  } catch (error) {
    setMessage(message, error.message, true);
  } finally {
    event.currentTarget.disabled = false;
  }
});

function clearRecordView() {
  $("#quality-badge").textContent = "미측정";
  $("#quality-badge").classList.remove("is-pass", "is-warn");
  $("#lineage-list").querySelectorAll("dd").forEach((element) => {
    element.textContent = "-";
    element.removeAttribute("title");
  });
  $("#metric-rms").textContent = "-";
  $("#metric-clip").textContent = "-";
  $("#metric-silence").textContent = "-";
  $("#quality-warnings").textContent = "-";
  $("#signal-meta").textContent = "데이터 없음";
  $("#audio-player").removeAttribute("src");
  $("#audio-player").load();
  $("#label-surface-text").value = "";
  $("#label-normalized-meaning").value = "";
  $("#label-expression-list").replaceChildren();
  $("#label-editor-version").textContent = "-";
  drawEmptySignals();
}

$("#confidence").addEventListener("input", (event) => {
  $("#confidence-output").textContent = Number(event.target.value).toFixed(2);
});

$("#evaluation-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = $("#evaluation-message");
  if (!state.record) {
    setMessage(message, "먼저 원본 발화와 Surface 정답을 등록하세요.", true);
    return;
  }
  const request = {
    utterance_id: state.record.utterance_id,
    experiment_id: $("#experiment-id").value.trim(),
    hypothesis_surface_text: $("#hypothesis").value.trim(),
    confidence: Number($("#confidence").value),
    latency_ms: $("#model-latency").value
      ? Number($("#model-latency").value)
      : null,
    model: {
      name: $("#model-name").value.trim(),
      version: $("#model-version").value.trim(),
    },
  };
  try {
    const prediction = await requestJson("/api/predictions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
    });
    const result = prediction.evaluation;
    renderEvaluation(result);
    await loadPredictions(prediction.prediction_id);
    setMessage(
      message,
      prediction.automatic_failure_candidates.length
        ? "예측을 저장하고 사람 검수 대기열에 추가했습니다."
        : "예측과 실험 정보를 저장했습니다.",
    );
  } catch (error) {
    setMessage(message, error.message, true);
  }
});

function renderEvaluation(result) {
  $("#result-cer").textContent = result.cer.toFixed(3);
  $("#result-preservation").textContent =
    `${(result.dialect.preservation_rate * 100).toFixed(0)}%`;
  $("#result-overcorrection").textContent =
    `${(result.dialect.overcorrection_rate * 100).toFixed(0)}%`;
  $("#result-confidence").textContent =
    result.high_confidence_wrong ? "HIGH RISK" : "LOW";
  const container = $("#expression-results");
  container.replaceChildren();
  if (!result.dialect.results.length) {
    container.textContent = "등록된 부산 표현 후보가 없어 표현 보존율은 중립값입니다.";
    return;
  }
  result.dialect.results.forEach((expression) => {
    const row = document.createElement("div");
    row.className = "expression-chip";
    const label = document.createElement("b");
    label.textContent = expression.surface_form;
    const status = document.createElement("span");
    status.textContent =
      `${expression.match_status.toUpperCase()} · ${expression.label_status}`;
    row.append(label, status);
    container.append(row);
  });
}

async function loadPredictions(preferredPredictionId = null) {
  if (!state.record) {
    state.predictions = [];
    state.predictionDiagnostics = new Map();
    renderPredictionList();
    await populatePredictionControls();
    return;
  }
  const predictions = await requestJson(
    `/api/predictions?utterance_id=${state.record.utterance_id}`,
  );
  state.predictions = predictions.sort(
    (left, right) => new Date(right.created_at) - new Date(left.created_at),
  );
  const diagnostics = await Promise.all(
    state.predictions.map((prediction) =>
      requestJson(`/api/predictions/${prediction.prediction_id}/diagnostics`),
    ),
  );
  state.predictionDiagnostics = new Map(
    diagnostics.map((diagnostic) => [diagnostic.prediction_id, diagnostic]),
  );
  renderPredictionList();
  await populatePredictionControls(preferredPredictionId);
}

function renderPredictionList() {
  const container = $("#prediction-list");
  container.replaceChildren();
  $("#prediction-count").textContent =
    `${state.predictions.length} PREDICTION${state.predictions.length === 1 ? "" : "S"}`;
  if (!state.predictions.length) {
    const empty = document.createElement("p");
    empty.className = "empty-state";
    empty.textContent = "저장된 예측이 없습니다.";
    container.append(empty);
    return;
  }
  state.predictions.forEach((prediction) => {
    const result = prediction.evaluation;
    const diagnostics = state.predictionDiagnostics.get(prediction.prediction_id);
    const card = document.createElement("article");
    card.className = "prediction-card";

    const head = document.createElement("div");
    head.className = "prediction-card__head";
    const title = document.createElement("strong");
    title.textContent = `${result.model.name} · ${result.model.version}`;
    const source = document.createElement("span");
    source.textContent = prediction.source.toUpperCase();
    head.append(title, source);

    const metadata = document.createElement("p");
    metadata.className = "prediction-card__meta";
    metadata.textContent =
      `${prediction.experiment_id} · ${prediction.prediction_id}`;

    const hypothesis = document.createElement("p");
    hypothesis.className = "prediction-card__hypothesis";
    hypothesis.textContent = result.hypothesis_surface_text
      ? `“${result.hypothesis_surface_text}”`
      : "“(빈 출력)”";

    const metrics = document.createElement("div");
    metrics.className = "prediction-card__metrics";
    appendPredictionMetric(metrics, "CER", result.cer.toFixed(3));
    appendPredictionMetric(
      metrics,
      "PRESERVE",
      `${(result.dialect.preservation_rate * 100).toFixed(0)}%`,
    );
    appendPredictionMetric(
      metrics,
      "OVER",
      `${(result.dialect.overcorrection_rate * 100).toFixed(0)}%`,
    );
    card.append(head, metadata, hypothesis, metrics);

    const observedErrors = [
      ...new Set(
        (diagnostics?.observations || [])
          .map((observation) => observation.observed_error)
          .filter((error) => error !== "NO_ERROR"),
      ),
    ];
    if (observedErrors.length) {
      const observations = document.createElement("p");
      observations.className = "observation-errors";
      observations.textContent = `관찰 오류 · ${observedErrors.join(" · ")}`;
      card.append(observations);
    }

    const suspectedCauses = [
      ...new Set(
        (diagnostics?.observations || [])
          .map((observation) => observation.suspected_cause)
          .filter((cause) => cause !== "UNKNOWN"),
      ),
    ];
    if (suspectedCauses.length) {
      const causes = document.createElement("p");
      causes.className = "suspected-causes";
      causes.textContent = `원인 후보 · ${suspectedCauses.join(" · ")}`;
      card.append(causes);
    }

    const automaticCandidates = failureCandidatesFor(prediction);
    if (automaticCandidates.length) {
      const failures = document.createElement("p");
      failures.className = "failure-candidates";
      failures.textContent =
        `검수 유형 후보 · ${automaticCandidates.join(" · ")}`;
      card.append(failures);
    }
    container.append(card);
  });
}

function appendPredictionMetric(container, label, value) {
  const metric = document.createElement("div");
  const name = document.createElement("span");
  name.textContent = label;
  const number = document.createElement("strong");
  number.textContent = value;
  metric.append(name, number);
  container.append(metric);
}

function failureCandidatesFor(prediction) {
  const diagnostics = state.predictionDiagnostics.get(prediction.prediction_id);
  return [
    ...new Set([
      ...prediction.automatic_failure_candidates,
      ...(diagnostics?.automatic_failure_candidates || []),
    ]),
  ];
}

async function populatePredictionControls(preferredPredictionId = null) {
  const comparisonA = $("#prediction-a");
  const comparisonB = $("#prediction-b");
  const reviewPrediction = $("#review-prediction");
  [comparisonA, comparisonB, reviewPrediction].forEach((select) => {
    select.replaceChildren();
    state.predictions.forEach((prediction) => {
      const option = document.createElement("option");
      option.value = prediction.prediction_id;
      option.textContent =
        `${prediction.evaluation.model.version} · ${prediction.experiment_id} · ` +
        `CER ${prediction.evaluation.cer.toFixed(3)}`;
      select.append(option);
    });
    select.disabled = !state.predictions.length;
  });

  const comparisonButton = $("#comparison-form button");
  comparisonButton.disabled = state.predictions.length < 2;
  $("#review-form button").disabled = !state.predictions.length;

  if (state.predictions.length >= 2) {
    comparisonA.value = state.predictions[1].prediction_id;
    comparisonB.value = state.predictions[0].prediction_id;
  }
  if (state.predictions.length) {
    const selectedPrediction =
      state.predictions.find(
        (prediction) => prediction.prediction_id === preferredPredictionId,
      ) || state.predictions[0];
    reviewPrediction.value = selectedPrediction.prediction_id;
    updateReviewFailureCandidates();
  } else {
    $("#review-failures").value = "";
    renderReviewHistory();
  }
  await loadReviews();
}

function updateReviewFailureCandidates() {
  const prediction = state.predictions.find(
    (item) => item.prediction_id === $("#review-prediction").value,
  );
  $("#review-failures").value = prediction
    ? failureCandidatesFor(prediction).join(", ")
    : "";
}

$("#review-prediction").addEventListener("change", async () => {
  updateReviewFailureCandidates();
  try {
    await loadReviews();
  } catch (error) {
    setMessage($("#review-message"), error.message, true);
  }
});

$("#comparison-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = $("#comparison-message");
  try {
    const comparison = await requestJson("/api/comparisons", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prediction_a_id: $("#prediction-a").value,
        prediction_b_id: $("#prediction-b").value,
      }),
    });
    renderComparison(comparison);
    setMessage(message, "같은 발화와 정답 기준으로 비교했습니다.");
  } catch (error) {
    setMessage(message, error.message, true);
  }
});

function renderComparison(comparison) {
  const container = $("#comparison-result");
  container.replaceChildren();
  const a = comparison.prediction_a.evaluation.model;
  const b = comparison.prediction_b.evaluation.model;
  const summary = document.createElement("p");
  summary.textContent = `A ${a.version} → B ${b.version} 변화량`;
  container.append(summary);
  appendComparisonMetric(
    container,
    "CER Δ",
    formatSigned(comparison.cer_delta_b_minus_a, 3),
  );
  appendComparisonMetric(
    container,
    "보존율 Δ",
    `${formatSigned(comparison.preservation_delta_b_minus_a * 100, 0)}%p`,
  );
  appendComparisonMetric(
    container,
    "과보정 Δ",
    `${formatSigned(comparison.overcorrection_delta_b_minus_a * 100, 0)}%p`,
  );
  appendComparisonMetric(
    container,
    "신뢰도 Δ",
    formatSigned(comparison.confidence_delta_b_minus_a, 2),
  );
}

function appendComparisonMetric(container, label, value) {
  const metric = document.createElement("div");
  metric.className = "comparison-metric";
  const name = document.createElement("span");
  name.textContent = label;
  const number = document.createElement("strong");
  number.textContent = value;
  metric.append(name, number);
  container.append(metric);
}

function formatSigned(value, digits) {
  if (value == null) return "-";
  const normalized = Number(value);
  return `${normalized > 0 ? "+" : ""}${normalized.toFixed(digits)}`;
}

$("#review-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = $("#review-message");
  const failureTypes = $("#review-failures").value
    .split(",")
    .map((value) => value.trim())
    .filter(Boolean);
  try {
    await requestJson("/api/reviews", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prediction_id: $("#review-prediction").value,
        reviewer_id: $("#reviewer-id").value.trim(),
        verdict: $("#review-verdict").value,
        confirmed_failure_types: failureTypes,
        notes: $("#review-notes").value.trim() || null,
      }),
    });
    $("#review-notes").value = "";
    await loadReviews();
    setMessage(message, "사람 판정을 새 revision으로 저장했습니다.");
  } catch (error) {
    setMessage(message, error.message, true);
  }
});

async function loadReviews() {
  const predictionId = $("#review-prediction").value;
  if (!predictionId) {
    state.reviews = [];
    renderReviewHistory();
    return;
  }
  state.reviews = await requestJson(
    `/api/reviews?prediction_id=${predictionId}`,
  );
  state.reviews.sort(
    (left, right) => new Date(right.created_at) - new Date(left.created_at),
  );
  renderReviewHistory();
}

function renderReviewHistory() {
  const container = $("#review-history");
  container.replaceChildren();
  if (!state.reviews.length) {
    const empty = document.createElement("p");
    empty.className = "empty-state";
    empty.textContent = "저장된 검수가 없습니다.";
    container.append(empty);
    return;
  }
  state.reviews.forEach((review) => {
    const card = document.createElement("article");
    card.className = "review-card";
    const title = document.createElement("strong");
    title.textContent =
      `${review.verdict.toUpperCase()} · ${review.reviewer_id}`;
    const metadata = document.createElement("p");
    metadata.className = "review-card__meta";
    metadata.textContent =
      `${new Date(review.created_at).toLocaleString()} · ${review.review_id}`;
    const failures = document.createElement("p");
    failures.textContent = review.confirmed_failure_types.length
      ? `확정 오류 · ${review.confirmed_failure_types.join(" · ")}`
      : "확정 오류 유형 없음";
    card.append(title, metadata, failures);
    if (review.notes) {
      const notes = document.createElement("p");
      notes.textContent = review.notes;
      card.append(notes);
    }
    container.append(card);
  });
}

async function loadBenchmarks() {
  state.benchmarks = await requestJson("/api/benchmarks");
  const container = $("#benchmark-list");
  container.replaceChildren();
  if (!state.benchmarks.length) {
    const empty = document.createElement("p");
    empty.className = "empty-state";
    empty.textContent = "활성 Benchmark가 없습니다.";
    container.append(empty);
    return;
  }
  state.benchmarks.forEach((manifest) => {
    const row = document.createElement("article");
    row.className = "benchmark-item";
    const summary = document.createElement("div");
    const title = document.createElement("strong");
    title.textContent = `${manifest.benchmark_id}@${manifest.benchmark_version}`;
    const metadata = document.createElement("span");
    metadata.textContent = `${manifest.entries.length}개 발화 · test · 고정됨`;
    summary.append(title, metadata);
    const button = document.createElement("button");
    button.type = "button";
    button.className = "benchmark-delete";
    button.textContent = "고정 해제";
    button.addEventListener("click", async () => {
      if (!window.confirm(
        `${title.textContent}의 고정을 해제할까요?\n` +
          "Manifest는 휴지통으로 이동하며 발화와 오디오는 유지됩니다.",
      )) return;
      button.disabled = true;
      try {
        const result = await requestJson(
          `/api/benchmarks/${encodeURIComponent(manifest.benchmark_id)}/` +
            encodeURIComponent(manifest.benchmark_version),
          { method: "DELETE" },
        );
        setMessage(
          $("#benchmark-message"),
          `${result.benchmark_id}@${result.benchmark_version} 고정을 해제했습니다. ` +
            `${result.entry_count}개 발화와 오디오는 유지됩니다.`,
        );
        await loadBenchmarks();
      } catch (error) {
        setMessage($("#benchmark-message"), error.message, true);
        button.disabled = false;
      }
    });
    row.append(summary, button);
    container.append(row);
  });
}

$("#benchmark-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = $("#benchmark-message");
  if (!state.records.length) {
    setMessage(message, "고정할 발화를 먼저 등록하세요.", true);
    return;
  }
  try {
    const manifest = await requestJson("/api/benchmarks", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        benchmark_id: $("#benchmark-id").value.trim(),
        benchmark_version: $("#benchmark-version").value.trim(),
        utterance_ids: state.records.map((record) => record.utterance_id),
        split: "test",
      }),
    });
    setMessage(
      message,
      `${manifest.benchmark_id}@${manifest.benchmark_version}에 ` +
        `${manifest.entries.length}개 발화를 고정했습니다.`,
    );
    await loadBenchmarks();
  } catch (error) {
    setMessage(message, error.message, true);
  }
});

function setupCanvas(canvas) {
  const ratio = Math.max(1, window.devicePixelRatio || 1);
  const rectangle = canvas.getBoundingClientRect();
  const width = Math.max(1, Math.round(rectangle.width));
  const height = Math.max(1, Math.round(rectangle.height));
  canvas.width = width * ratio;
  canvas.height = height * ratio;
  const context = canvas.getContext("2d");
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  return { context, width, height };
}

function drawEmptySignals() {
  ["#waveform", "#mel", "#f0"].forEach((selector, signalIndex) => {
    const { context, width, height } = setupCanvas($(selector));
    context.clearRect(0, 0, width, height);
    context.strokeStyle = "#c8d8d4";
    context.lineWidth = 1;
    context.setLineDash([4, 7]);
    context.beginPath();
    context.moveTo(0, height / 2);
    context.lineTo(width, height / 2);
    context.stroke();
    context.setLineDash([]);
    context.fillStyle = "#6f8985";
    context.font = "10px SFMono-Regular, monospace";
    context.fillText(
      ["NO WAVEFORM DATA", "NO MEL DATA", "NO F0 DATA"][signalIndex],
      14,
      height / 2 - 11,
    );
  });
}

function renderSignals(analysis) {
  drawWaveform(analysis);
  drawMel(analysis);
  drawF0(analysis);
}

function drawWaveform(analysis) {
  const { context, width, height } = setupCanvas($("#waveform"));
  context.clearRect(0, 0, width, height);
  context.fillStyle = "#edf4f2";
  context.fillRect(0, 0, width, height);
  const mid = height / 2;
  context.beginPath();
  analysis.waveform_max.forEach((value, index) => {
    const x = index / Math.max(1, analysis.waveform_max.length - 1) * width;
    const y = mid - value * (height * 0.42);
    index ? context.lineTo(x, y) : context.moveTo(x, y);
  });
  for (let index = analysis.waveform_min.length - 1; index >= 0; index -= 1) {
    const x = index / Math.max(1, analysis.waveform_min.length - 1) * width;
    const y = mid - analysis.waveform_min[index] * (height * 0.42);
    context.lineTo(x, y);
  }
  context.closePath();
  context.fillStyle = "#009d96";
  context.fill();
  context.strokeStyle = "#102a2c";
  context.lineWidth = 0.8;
  context.stroke();
}

function drawMel(analysis) {
  const { context, width, height } = setupCanvas($("#mel"));
  context.clearRect(0, 0, width, height);
  const rows = analysis.mel_db.length;
  const columns = rows ? analysis.mel_db[0].length : 0;
  if (!rows || !columns) return;
  const cellWidth = width / columns;
  const cellHeight = height / rows;
  for (let row = 0; row < rows; row += 1) {
    for (let column = 0; column < columns; column += 1) {
      const normalized = Math.max(0, Math.min(1, (analysis.mel_db[row][column] + 80) / 80));
      const hue = 181 - normalized * 146;
      const lightness = 96 - normalized * 52;
      context.fillStyle = `hsl(${hue} 70% ${lightness}%)`;
      context.fillRect(
        column * cellWidth,
        height - (row + 1) * cellHeight,
        Math.ceil(cellWidth),
        Math.ceil(cellHeight),
      );
    }
  }
}

function drawF0(analysis) {
  const { context, width, height } = setupCanvas($("#f0"));
  context.clearRect(0, 0, width, height);
  context.fillStyle = "#fbfdfc";
  context.fillRect(0, 0, width, height);
  const voiced = analysis.f0_hz.filter((value) => value !== null);
  const minimum = voiced.length ? Math.min(...voiced) : 65;
  const maximum = voiced.length ? Math.max(...voiced) : 450;
  context.strokeStyle = "#f3aa32";
  context.lineWidth = 2.5;
  context.beginPath();
  let drawing = false;
  analysis.f0_hz.forEach((value, index) => {
    if (value === null) {
      drawing = false;
      return;
    }
    const x = index / Math.max(1, analysis.f0_hz.length - 1) * width;
    const y = height - 13 - ((value - minimum) / Math.max(1, maximum - minimum)) * (height - 26);
    if (drawing) context.lineTo(x, y);
    else context.moveTo(x, y);
    drawing = true;
  });
  context.stroke();
  context.fillStyle = "#355355";
  context.font = "9px SFMono-Regular, monospace";
  context.fillText(`${minimum.toFixed(0)}-${maximum.toFixed(0)} Hz · display-normalized only`, 12, 16);
}

window.addEventListener("resize", () => {
  if (state.analysis) renderSignals(state.analysis);
  else drawEmptySignals();
});

initialize();
