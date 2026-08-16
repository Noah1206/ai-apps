# 프론트엔드 확장 빌드 계획서

부산 발화 연구 랩 — 기존 정적 연구 UI(`src/busan_lab/static/`) 확장 작업.
**새 프레임워크/의존성 없음.** vanilla JS + FastAPI 정적 서빙 스택 유지.

- 대상 파일: `index.html` (517L), `app.js` (1402L), `styles.css` (1494L)
- 백엔드: `src/busan_lab/api.py` (라우트 20개)
- 검증 도구: `scripts/build_frontend.sh` (check/serve/smoke)

---

## 0. 현황 — 이미 있는 화면 (재작업 금지, 재사용)

| 화면 | 상태 | 쓰는 API |
|---|---|---|
| 발화 등록/업로드 폼 | ✅ 있음 | `POST /api/utterances` |
| 오디오 품질 배지·지표 | ✅ 있음 | (업로드 응답 내장) |
| 신호 분석(파형/Mel/F0) | ✅ 있음 | `GET /api/utterances/{id}/analysis` |
| 발화 선택 드롭다운 | ✅ 있음 | `GET /api/utterances` |
| 라벨 편집 | ✅ 있음 | `PUT /api/utterances/{id}/labels` |
| TASK-004 검수 패널 | ✅ 있음 | `/api/training-imports/*` |
| 모델 평가 폼 | ✅ 있음 | `POST /api/evaluations` |

## 갭 — API는 있는데 UI가 없는 것 (= 이번 작업 범위)

| # | 화면 | 미사용 API | 우선순위 |
|---|---|---|---|
| S1 | 예측 목록 + 진단 뷰 | `GET /api/predictions`, `/predictions/{id}/diagnostics` | 높음 |
| S2 | 전역 리뷰 큐 | `GET /api/review-queue`, `POST /api/reviews` | 높음 |
| S3 | 실험 목록/등록 | `GET/POST /api/experiments` | 중간 |
| S4 | 예측 비교 뷰 | `POST /api/comparisons` | 중간 |
| S5 | 벤치마크 목록 | `GET /api/benchmarks`, `/benchmarks/{id}/{ver}` | 중간 |
| S6 | 라벨 이력(리비전) | `GET /api/utterances/{id}/label-revisions` | 낮음 |

---

## 작업 단계 — 화면별 수용조건 + 시간상한

각 단계는 **수용조건 전부 PASS** AND **시간상한 내** 여야 DONE.
상한 초과 시 → 그 화면은 "골격만(빈 상태+에러표시)"으로 축소(scope cut)하고 다음 단계로.

### 단계 1 — 공통 렌더 유틸 & 새 사이드바 탭 (상한 1.5h)
- [ ] `index.html` 사이드바 nav에 S1~S5 진입 링크 5개 추가 (기존 `setupSidebarNavigation` 재사용)
- [ ] 각 화면용 빈 `<section>` 5개 + `id` 부여, 기본 `hidden`
- [ ] `app.js`에 목록 렌더 공통 함수 1개 추가 (`renderList(el, items, rowFn)`) — 중복 방지
- **DONE**: 탭 클릭 시 해당 섹션만 보이고 나머지 숨김, 콘솔 에러 0

### 단계 2 — S1 예측 목록 + 진단 (상한 3h)
- [ ] `GET /api/predictions` → 표 렌더 (id, experiment_id, 모델, CER, latency)
- [ ] 행 클릭 → `GET /api/predictions/{id}/diagnostics` → 상세 패널
- [ ] 로딩 중 스피너, 빈 목록 시 "예측 없음" 안내, 4xx/5xx 시 에러 메시지 표시
- **DONE**: 위 3개 상태(정상/빈/에러) 전부 화면에 표시됨, smoke의 `/api/predictions` 200

### 단계 3 — S2 전역 리뷰 큐 (상한 3h)
- [ ] `GET /api/review-queue` → 대기 항목 목록
- [ ] 항목별 승인/재검토 액션 → `POST /api/reviews` 후 목록 갱신(낙관적 아님, 재요청)
- [ ] 제출 실패 시 해당 행에 에러, 성공 시 목록에서 제거
- **DONE**: 승인 1건 제출 → 목록에서 사라지고 재조회해도 유지됨

### 단계 4 — S3 실험 + S4 비교 (상한 3h)
- [ ] `GET /api/experiments` 목록, `POST /api/experiments` 등록 폼(이름/버전)
- [ ] 예측 2개 선택 → `POST /api/comparisons` → CER·부산표현 보존율·차이 표시
- **DONE**: 실험 1건 등록되어 목록에 나타남 + 비교 결과 1건 렌더

### 단계 5 — S5 벤치마크 + S6 라벨 이력 (상한 2h)
- [ ] `GET /api/benchmarks` 목록, 항목 클릭 → `/{id}/{version}` 상세
- [ ] 발화 상세에 "라벨 이력" 버튼 → `GET /.../label-revisions` 타임라인
- **DONE**: 벤치마크 상세 1건 열림 + 라벨 이력 최소 1개 리비전 표시

### 단계 6 — 마감 검증 (상한 1h)
- [ ] `scripts/build_frontend.sh check` PASS (죽은 링크 0 — 새 API 참조 포함)
- [ ] `scripts/build_frontend.sh smoke` 5조건 PASS
- [ ] 전 화면 콘솔 에러 0, 접근성: 새 인터랙티브 요소에 `aria-label`/라벨 존재
- **DONE**: 아래 "전체 종료조건" 전부 충족

---

## 전체 종료조건 (제한사항) — 언제 끝인가

**아래 6개 전부 참이면 프론트 확장 완료. 하나라도 실패면 미완.**

1. **기능**: 갭 화면 S1~S6이 각자 정상/빈/에러 3상태를 화면에 표시한다.
2. **정합성**: `build_frontend.sh check` 죽은 링크 0 (app.js의 모든 `/api` 참조 실재).
3. **구동**: `build_frontend.sh smoke` 5조건 PASS (`/`, health, utterances 200).
4. **회귀 없음**: 기존 화면(업로드/신호분석/검수/라벨/평가) 동작 유지 — 콘솔 에러 0.
5. **접근성 최소선**: 새 인터랙티브 요소 전부 라벨/`aria-label` 있음.
6. **시간 상한**: 총 **13.5h** (1.5+3+3+3+2+1). 초과 시 낮은 우선순위(S5/S6)부터
   "골격만"으로 컷하고 1~4를 우선 충족한 상태로 종료한다.

## 하지 않는 것 (scope cut)
- React/Next/빌드툴 도입 ❌ (기존 스택 유지)
- 인증·다중 사용자 ❌ (로컬 연구 UI)
- 실시간(WebSocket) ❌ (요청-응답으로 충분)
- 신규 백엔드 API ❌ (기존 라우트만 소비; 없으면 그 화면은 범위 밖)
