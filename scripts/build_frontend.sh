#!/usr/bin/env bash
# build_frontend.sh — 부산 발화 연구 랩 프론트엔드 빌드/검증 스크립트
#
# 목적: 이미 존재하는 정적 연구 UI(src/busan_lab/static/)를 이 폴더의
#       실제 FastAPI 라우트에 맞춰 빌드·구동·검증한다. 새 프레임워크/의존성
#       추가 없음 — 순수 vanilla JS + FastAPI 정적 서빙 스택을 그대로 쓴다.
#
# 사용법:
#   scripts/build_frontend.sh            # 전체: 검증→서버 기동→스모크테스트
#   scripts/build_frontend.sh serve      # 서버만 기동
#   scripts/build_frontend.sh check      # 정적 자산 존재/라우트 정합성만 검사
#   scripts/build_frontend.sh smoke      # 실행 중인 서버에 스모크 테스트
#
# 완료(제한/종료) 조건 — 아래 5개가 모두 PASS면 "프론트 완성"으로 간주:
#   1) 3개 정적 자산(index.html/app.js/styles.css)이 존재하고 비어있지 않다
#   2) app.js가 참조하는 모든 /api 경로가 api.py에 실제로 존재한다 (죽은 링크 0)
#   3) 서버가 뜨고 GET / 가 200 + index.html을 반환한다
#   4) GET /api/health 가 {"status":"ok"} 를 반환한다
#   5) 핵심 화면 데이터(GET /api/utterances)가 200으로 응답한다
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATIC="$ROOT/src/busan_lab/static"
API="$ROOT/src/busan_lab/api.py"
HOST="127.0.0.1"
PORT="${PORT:-8000}"
BASE="http://$HOST:$PORT"

pass() { printf "  \033[32mPASS\033[0m %s\n" "$1"; }
fail() { printf "  \033[31mFAIL\033[0m %s\n" "$1"; FAILED=1; }
FAILED=0

# ── 1) 정적 자산 존재 검사 ────────────────────────────────────────────────
check_assets() {
  echo "[1/2] 정적 자산 검사"
  for f in index.html app.js styles.css; do
    if [[ -s "$STATIC/$f" ]]; then pass "$f 존재 ($(wc -l <"$STATIC/$f" | tr -d ' ') lines)"
    else fail "$STATIC/$f 없음 또는 빈 파일"; fi
  done
}

# ── 2) 라우트 정합성: app.js의 /api 참조가 전부 api.py에 있는가 ─────────────
# ponytail: grep 기반 정적 검사. 경로 문자열이 템플릿 리터럴로 조립되면 놓칠 수
#           있음 — 그때는 smoke 단계(실서버 호출)가 실제 실패로 잡아준다.
check_routes() {
  echo "[2/2] 라우트 정합성 (app.js → api.py)"
  local missing=0
  # app.js에서 "/api/..." 리터럴만 추출, {id} 같은 동적 세그먼트는 정규화
  local paths
  paths=$(grep -oE '/api/[a-zA-Z0-9_/{}-]+' "$STATIC/app.js" \
    | sed -E 's#\$\{[^}]+\}#{id}#g; s#/[0-9a-fA-F-]{8,}#/{id}#g' \
    | sort -u)
  while IFS= read -r p; do
    [[ -z "$p" ]] && continue
    # {id} 세그먼트를 api.py의 {name} 패턴과 매칭하기 위해 접두어만 확인
    local prefix="${p%%/\{id\}*}"
    if grep -qF "\"$prefix" "$API" || grep -qF "'$prefix" "$API"; then
      pass "$p"
    else
      fail "$p → api.py에 대응 라우트 없음(죽은 링크)"; missing=1
    fi
  done <<<"$paths"
  return $missing
}

# ── 서버 기동 ─────────────────────────────────────────────────────────────
serve() {
  echo "서버 기동: $BASE  (Ctrl+C 로 종료)"
  cd "$ROOT"
  exec uv run uvicorn busan_lab.api:create_app --factory --host "$HOST" --port "$PORT" --reload
}

# ── 스모크 테스트: 실행 중인 서버 대상 ────────────────────────────────────
smoke() {
  echo "스모크 테스트 → $BASE"
  local code
  code=$(curl -s -o /dev/null -w '%{http_code}' "$BASE/")            && [[ "$code" == 200 ]] && pass "GET / → 200" || fail "GET / → $code"
  curl -s "$BASE/api/health" | grep -q '"status":"ok"'               && pass "GET /api/health ok"    || fail "GET /api/health 비정상"
  code=$(curl -s -o /dev/null -w '%{http_code}' "$BASE/api/utterances") && [[ "$code" == 200 ]] && pass "GET /api/utterances → 200" || fail "GET /api/utterances → $code"
}

# ── 오케스트레이션 ────────────────────────────────────────────────────────
case "${1:-all}" in
  check) check_assets; check_routes ;;
  serve) serve ;;
  smoke) smoke ;;
  all)
    check_assets
    check_routes || true
    echo "서버 백그라운드 기동 후 스모크…"
    ( cd "$ROOT" && uv run uvicorn busan_lab.api:create_app --factory --host "$HOST" --port "$PORT" ) &
    SRV=$!
    trap 'kill $SRV 2>/dev/null || true' EXIT
    # 서버 뜰 때까지 최대 15초 대기
    for _ in $(seq 1 30); do
      curl -s -o /dev/null "$BASE/api/health" && break || sleep 0.5
    done
    smoke
    ;;
  *) echo "usage: $0 [all|check|serve|smoke]"; exit 2 ;;
esac

if [[ "$FAILED" -ne 0 ]]; then
  echo; echo "❌ 종료조건 미충족 — 위 FAIL 항목 해결 필요"; exit 1
else
  echo; echo "✅ 완료조건 전부 PASS — 프론트엔드 빌드/검증 통과"
fi
