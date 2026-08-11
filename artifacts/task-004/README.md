# TASK-004 Phase A

`training-split-assignments.example.json`은 형식 예시이며 UUID는 실제 새 학습
발화 ID로 모두 교체해야 한다. frozen `busan-surface-v0@1.0.0`의 10개 ID는
절대 넣지 않는다.

Phase A가 확정한 계약:

```text
새 학습 동의 발화
→ 사람 Surface label 검수
→ train/validation 화자 단위 배정
→ Benchmark·split 누출 검사
→ immutable TrainingDatasetManifest
→ hash 검증된 모델 중립 Training ZIP
```

아직 300~500개 실제 데이터, Fine-tuning 코드나 특정 모델용 변환기는 만들지
않는다.

## Phase B 수집 준비

- `collection-prompts-v0.jsonl`: 사람 녹음용 후보 문장 50개
- `COLLECTION_GUIDE.md`: 325개 Pilot 화자·split·녹음·검수 계획
- `SOLO_SPEAKER_300.md`: 한 사람이 녹음할 train 전용 후보 문장 300개

프롬프트는 사람 검수 전까지 `candidate`다. 파일의 문장을 자동으로 정답
Surface transcript로 복사하지 않고 실제 녹음에서 들린 형태를 기록한다.
