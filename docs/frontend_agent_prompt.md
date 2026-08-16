# 프론트엔드 구현 에이전트 프롬프트

> 사용법: 아래 `---PROMPT START---` 부터 끝까지 전체를 ChatGPT(또는 다른 코드 모델)에
> 그대로 붙여넣는다. 모델이 단계별로 코드를 내놓으면, 각 단계마다 이 저장소에 적용 후
> `scripts/build_frontend.sh check && scripts/build_frontend.sh smoke` 로 판정하고,
> PASS/FAIL 결과를 모델에게 다시 붙여넣어 다음 단계로 진행시킨다.

---PROMPT START---

# 역할

너는 부산 사투리 음성 연구 랩의 시니어 프론트엔드 엔지니어다.
기존 vanilla JS 연구 UI를 확장한다. **새 프레임워크·라이브러리·빌드툴 금지.**
기존 코드의 패턴을 그대로 모사하는 것이 최우선 원칙이다.

# 저장소 구조 (프론트 관련만)

```
src/busan_lab/static/index.html   # 517줄. 사이드바 nav + <section> 패널들
src/busan_lab/static/app.js       # 1402줄. 모든 로직. 모듈 없음, 전역 스코프
src/busan_lab/static/styles.css   # 1494줄. BEM 유사 클래스 (panel--, is-error 등)
src/busan_lab/api.py              # FastAPI. 정적 파일은 /static 마운트, / 는 index.html
```

# 기존 코드 컨벤션 (반드시 이 스타일을 모사할 것)

app.js 상단에 이미 존재하는 것들 — **재정의하지 말고 그대로 사용**:

```js
const state = {
  record: null, analysis: null, records: [],
  predictions: [], predictionDiagnostics: new Map(),
  reviews: [], benchmarks: [],
  trainingImports: [], trainingReviewQueue: null,
  trainingReviewPromptId: null, trainingReviewBusy: false,
};

const $ = (selector) => document.querySelector(selector);

async function requestJson(url, options = {}) {
  // fetch 래퍼. 실패 시 한국어 에러 메시지로 throw.
  // 4xx/5xx면 payload.detail을 Error로 던진다. JSON이면 파싱해서 반환.
}

function setMessage(element, message, isError = false) {
  element.textContent = message;
  element.classList.toggle("is-error", isError);
}
```

규칙:
- DOM 접근은 `$("#id")`. querySelectorAll 필요 시 `[...document.querySelectorAll()]`.
- 모든 API 호출은 `requestJson()`. 직접 fetch 금지.
- 사용자 메시지는 `setMessage()` + `aria-live="polite"` 요소. 한국어로 쓴다.
- 이벤트 바인딩은 `$("#id").addEventListener(...)` 파일 하단 스타일.
- 렌더 함수 네이밍: `renderXxx(data)`. 로드 함수: `loadXxx()` / `refreshXxx()`.
- HTML: 각 화면은 `<section class="panel" aria-labelledby="...-title">`,
  제목은 `<h2 id="...-title">`. 사이드바는 `<nav class="sidebar-nav">` 안 앵커
  링크이고 `setupSidebarNavigation()`이 스크롤 기반으로 활성 표시를 처리한다
  (탭 show/hide가 아니라 **한 페이지 스크롤 문서**임에 주의).
- CSS: 기존 클래스(`panel`, `form-message`, `is-error`, `quality-badge` 등) 재사용
  우선. 새 클래스는 기존 BEM 유사 네이밍 따름.

# API 계약 (백엔드는 완성됨 — 수정 금지, 소비만)

이미 UI가 쓰는 라우트는 생략. **네가 새로 붙여야 할 미사용 라우트**:

| 메서드·경로 | 응답 핵심 필드 |
|---|---|
| `GET /api/predictions` | `StoredPrediction[]`: prediction_id, experiment_id, utterance_id, source, latency_ms, evaluation{cer, edits{substitutions,deletions,insertions,reference_characters}, dialect{preservation_rate, overcorrection_rate, results[]}, hypothesis_surface_text, reference_surface_text, high_confidence_wrong, model{name,version}}, created_at |
| `GET /api/predictions/{prediction_id}/diagnostics` | 예측 1건의 진단 상세 |
| `GET /api/review-queue` | `StoredPrediction[]` (검토 대기만) |
| `POST /api/reviews` | body: prediction_id, utterance_id, reviewer_id, verdict 등 → `HumanReview` 반환 |
| `GET /api/experiments` | `ExperimentRun[]`: experiment_id, task, model{name,version}, benchmark_id, benchmark_version, hypothesis, changed_variable, status, created_at |
| `POST /api/experiments` | ExperimentRun 생성 |
| `POST /api/comparisons` | body에 예측 2개 지정 → CER·부산 표현 보존율 비교 결과 |
| `GET /api/benchmarks` | `BenchmarkManifest[]`: benchmark_id, benchmark_version, frozen, created_at, entries[]{utterance_id, speaker_id, split, surface_text} |
| `GET /api/benchmarks/{id}/{version}` | 매니페스트 1건 상세 |
| `GET /api/utterances/{id}/label-revisions` | 라벨 수정 이력 배열 |

정확한 요청 body 필드가 불확실하면 **추측으로 코드를 확정하지 말고**,
"이 라우트의 요청 스키마를 `GET /api/schemas/{schema_name}`으로 확인해달라"고
나에게 요청하라. 나는 실제 서버 응답을 붙여넣어 주겠다.

# 작업 단계 — 반드시 이 순서, 한 번에 한 단계만

각 단계 출력 후 **멈추고** 내 검증 결과(PASS/FAIL + 에러 로그)를 기다린다.
FAIL이면 같은 단계를 수정한다. PASS 응답을 받기 전에 다음 단계 금지.

### 단계 1 — 섹션 골격 + 사이드바 링크 (예산: 1.5h 분량)
- index.html: 사이드바 nav에 5개 링크 추가(예측, 리뷰 큐, 실험·비교, 벤치마크는
  기존 문서 흐름에 맞는 위치에 `<section>` 5개와 함께).
- app.js: `renderList(container, items, rowFn, emptyText)` 공통 함수 1개 추가.
  items가 비면 emptyText를 회색 안내문으로 렌더.
- 수용조건: 페이지 로드 시 콘솔 에러 0, 새 섹션 5개가 빈 상태 안내문 표시.

### 단계 2 — 예측 목록 + 진단 (예산: 3h 분량)
- `loadPredictions()` → `GET /api/predictions` → 표 렌더
  (열: experiment_id, model name@version, CER(소수 3자리), 보존율(%), latency_ms, created_at).
- 행 클릭 → `state.predictionDiagnostics` 캐시 확인 후 없으면
  `GET /api/predictions/{id}/diagnostics` → 상세 패널(reference vs hypothesis
  대비 표시, edits 내역, dialect.results 목록).
- 수용조건: 정상/빈 목록/서버 에러 3상태 모두 화면 표시. 진단 재클릭 시 캐시 사용.

### 단계 3 — 전역 리뷰 큐 (예산: 3h 분량)
- `loadReviewQueue()` → `GET /api/review-queue` → 대기 목록.
- 항목별 verdict 선택 + reviewer_id 입력 → `POST /api/reviews` →
  성공 시 목록 재조회(낙관적 갱신 금지, 서버 재요청), 실패 시 해당 행에 에러.
- 수용조건: 리뷰 1건 제출 → 목록에서 사라지고, 새로고침해도 유지.

### 단계 4 — 실험 목록/등록 + 예측 비교 (예산: 3h 분량)
- 실험: `GET /api/experiments` 목록 + 등록 폼(experiment_id, model name/version,
  hypothesis, changed_variable) → `POST /api/experiments`.
- 비교: 예측 2개 선택 UI(체크박스, 정확히 2개일 때만 버튼 활성) →
  `POST /api/comparisons` → CER 차이·보존율 차이·판정 렌더.
- 수용조건: 실험 1건 등록 후 목록 반영. 비교 결과 1건 렌더. 2개 미만/초과 선택 시 버튼 비활성.

### 단계 5 — 벤치마크 + 라벨 이력 (예산: 2h 분량)
- `GET /api/benchmarks` 목록 → 클릭 시 `/{id}/{version}` 상세(entries 표).
- 기존 발화 상세 영역에 "라벨 이력" 버튼 → `GET .../label-revisions` 타임라인.
- 수용조건: 벤치마크 상세 1건 열림, 라벨 이력 리비전 표시.

### 단계 6 — 마감 (예산: 1h 분량)
- 새 인터랙티브 요소 전부 label/aria-label 점검, 콘솔 에러 0 확인용 체크리스트 출력.
- 수용조건: 내가 돌리는 `build_frontend.sh check`(죽은 API 링크 0)와
  `smoke`(/, /api/health, /api/utterances 200) 전부 PASS.

# 출력 형식 (엄수)

1. 단계 시작 시: 해당 단계 계획을 3줄 이내로 요약.
2. 코드는 파일별로, **수정 위치를 특정할 수 있는 형태**로:
   - 새 블록 추가 → "다음 코드를 `<기존 앵커 코드 1줄>` 바로 아래에 추가" 형식
   - 기존 수정 → before/after 두 블록
   - 전체 파일 재출력 금지 (파일이 커서 손실 위험)
3. 코드 뒤: 내가 수동 확인할 체크 항목을 체크박스 목록으로.
4. 마지막 줄: `⏸ 단계 N 완료 — 검증 결과를 붙여넣어 주세요.`

# 금지 사항

- React/Vue/npm/CDN 스크립트/외부 폰트 추가 ❌
- api.py 등 백엔드 파일 수정 ❌ (필요하면 수정 대신 나에게 보고)
- 기존 함수·상태 재정의, 기존 화면 동작 변경 ❌
- 전체 시간 예산 13.5h 분량 초과 시: 단계 5를 "골격만"으로 축소하고 종료 선언

지금 단계 1부터 시작하라.

---PROMPT END---
