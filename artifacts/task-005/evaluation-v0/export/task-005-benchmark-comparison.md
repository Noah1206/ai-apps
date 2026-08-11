# TASK-005 Nemotron Benchmark Comparison

Status: `automatic_evaluation_complete_human_review_pending`

## Automatic metrics

| Metric | NeMo pretrained | Busan adapter best | Change |
|---|---:|---:|---:|
| CER | 0.6167 | 0.0500 | -0.5667 |
| Dialect preservation | 0.2000 | 0.9333 | +0.7333 |
| Context overcorrection | 0.0000 | 0.0000 | 0.0000 |
| Empty predictions | 3 | 0 | -3 |
| p50 latency (ms) | 407.43 | 448.54 | +41.11 |
| p95 latency (ms) | 9954.68 | 3392.04 | -6562.63 |

CER 기준 발화별 결과는 개선 8개, 동률 2개, 악화 0개다. 방언 보존율 기준은
개선 9개, 동률 1개, 악화 0개다.

## Utterance comparison

| Reference | Pretrained | Fine-tuned best | CER result |
|---|---|---|---|
| 와따 맛있노 | *(empty)* | 와따 맛있노? | improved |
| 마, 괜찮다 아이가. | 막 괜찮다 이거 | 막 괜찮다 아이가. | improved |
| 내일 같이 가재이 | *(empty)* | 내일 같이 가재이. | improved |
| 여기 좀 앉으이소. | 어 이 조만칠이서 | 아 여기 좀 앉으이소. | improved |
| 지금 뭐 하노? | 지금 뭐 하노 | 지금 뭐 하노? | tied |
| 그거 아이다. | *(empty)* | 그거 아이다. | improved |
| 오늘 와 이리 춥노? | 옷 와이디 춤로 | 오늘 와이리 춥노? | improved |
| 밥 묵었나? | 밤 무거나 | 밥 묵었나? | improved |
| 니 지금 어데고? | 니 지금 어디가 | 니 지금 어데고? | improved |
| 국밥 하나 주이소 | 국밥 하나 주의소 | 국밥아나 주이소. | tied |

## Applied evaluation contract

- CER: Unicode NFC 후 Unicode separator와 punctuation을 제거한다. 단어는 바꾸지 않는다.
- 방언 보존: NFC surface form의 정확한 부분 문자열 포함 여부다.
- 과보정: NFC normalized form의 정확한 부분 문자열 포함 여부다.
- WER: 현재 계약에 없어 계산하지 않았다.

## Human review and limitations

모델명이 없는 A/B 검수 목록 10개를 생성했지만 사람 판정은 아직 0건이다. 따라서
fine-tuned 모델의 성능 우위는 아직 확정하지 않는다.

Train은 선언상 단일 화자이고 Benchmark의 `sample1`~`sample10` ID는 실제 인물의
독립성을 증명하지 않는다. 정확한 전사 중복은 없지만 유사 방언 표현은 의도적으로
Train에 포함돼 있다. Benchmark 방언 표현 15개도 모두 candidate 상태다. 이 결과는
현재 Pilot에서 강한 자동 개선 증거이며, 일반 부산 화자 성능의 확정 증거는 아니다.
