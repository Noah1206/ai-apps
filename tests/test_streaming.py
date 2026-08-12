from __future__ import annotations

import pytest
from pydantic import ValidationError

from busan_lab.schemas.asr import ModelDescriptor
from busan_lab.schemas.streaming import (
    Gate3Criteria,
    Gate3Evidence,
    StreamingSessionStatus,
    StreamingTranscriptEvent,
)
from busan_lab.streaming import (
    EnergyEndpointDetector,
    StreamingSessionError,
    StreamingTranscriptSession,
    assess_gate3,
)


def _model() -> ModelDescriptor:
    return ModelDescriptor(
        name="nvidia/nemotron-3.5-asr-streaming-0.6b",
        version="f3d333391852ba876df169dcc9ba902d25b6ab0b",
        model_provider="NVIDIA",
        model_family="FastConformer-RNNT",
        decoder_type="RNNT",
        target_language="ko-KR",
        fine_tuned=True,
    )


def test_streaming_session_publishes_stable_prefix_and_final_metrics() -> None:
    session = StreamingTranscriptSession(session_id="stream-1", model=_model())
    transcripts = ("니", "니 지금", "니 지금 어", "니 지금 어데", "니 지금 어데고")
    events = [
        session.observe(
            transcript,
            audio_end_ms=(index + 1) * 320,
            emitted_at_ms=(index + 1) * 100,
            inference_latency_ms=20,
            final=index == len(transcripts) - 1,
            endpoint_detected=index == len(transcripts) - 1,
        )
        for index, transcript in enumerate(transcripts)
    ]

    assert events[2].stable_prefix == "니"
    assert events[2].unstable_suffix == " 지금 어"
    assert events[-1].partial is False
    assert events[-1].stable_prefix == transcripts[-1]
    assert events[-1].unstable_suffix == ""
    assert session.status is StreamingSessionStatus.FINALIZED

    metrics = session.metrics(surface_final_transcript="니 지금 어데고")
    assert metrics.event_count == 5
    assert metrics.partial_event_count == 4
    assert metrics.endpoint_count == 1
    assert metrics.first_partial_latency_ms == 100
    assert metrics.final_agreement_with_surface_asr == 1
    assert metrics.surface_final_cer == 0


def test_streaming_session_detects_stable_prefix_retraction() -> None:
    session = StreamingTranscriptSession(
        session_id="stream-retraction",
        model=_model(),
        stabilization_window=2,
    )
    session.observe("밥 묵", audio_end_ms=320, emitted_at_ms=20, inference_latency_ms=20)
    session.observe("밥 묵", audio_end_ms=640, emitted_at_ms=40, inference_latency_ms=20)
    session.observe("밤 먹", audio_end_ms=960, emitted_at_ms=60, inference_latency_ms=20)
    session.observe(
        "밤 먹었나",
        audio_end_ms=1280,
        emitted_at_ms=80,
        inference_latency_ms=20,
        final=True,
        endpoint_detected=True,
    )

    metrics = session.metrics(surface_final_transcript="밥 묵었나")
    assert metrics.stable_prefix_observations == 1
    assert metrics.stable_prefix_violations == 1
    assert metrics.partial_stability == 0
    assert metrics.final_agreement_with_surface_asr < 1


def test_stable_prefix_compares_the_first_value_when_shortest_is_later() -> None:
    session = StreamingTranscriptSession(
        session_id="stream-shortest-middle",
        model=_model(),
        stabilization_window=3,
    )
    session.observe("가나다", audio_end_ms=320, emitted_at_ms=20, inference_latency_ms=20)
    session.observe("나다", audio_end_ms=640, emitted_at_ms=40, inference_latency_ms=20)
    event = session.observe(
        "나다라",
        audio_end_ms=960,
        emitted_at_ms=60,
        inference_latency_ms=20,
    )

    assert event.stable_prefix == ""
    assert event.unstable_suffix == "나다라"


def test_streaming_session_reset_and_cancel_are_explicit() -> None:
    session = StreamingTranscriptSession(session_id="stream-lifecycle", model=_model())
    session.observe("국밥", audio_end_ms=320, emitted_at_ms=10, inference_latency_ms=10)
    session.cancel()
    assert session.status is StreamingSessionStatus.CANCELLED
    with pytest.raises(StreamingSessionError, match="cancelled"):
        session.observe("국밥 하나", audio_end_ms=640, emitted_at_ms=20, inference_latency_ms=10)

    session.reset()
    event = session.observe(
        "새 세션",
        audio_end_ms=320,
        emitted_at_ms=10,
        inference_latency_ms=10,
        final=True,
    )
    assert event.generation == 1
    assert event.sequence == 0
    assert session.status is StreamingSessionStatus.FINALIZED


def test_streaming_event_rejects_inconsistent_prefix_partition() -> None:
    with pytest.raises(ValidationError, match="exactly form transcript"):
        StreamingTranscriptEvent(
            session_id="stream-invalid",
            generation=0,
            sequence=0,
            partial=True,
            stable_prefix="니 지금",
            unstable_suffix="어데",
            transcript="니 지금 어데",
            stability=0.5,
            audio_start_ms=0,
            audio_end_ms=320,
            emitted_at_ms=20,
            inference_latency_ms=20,
            model=_model(),
        )


def test_energy_endpoint_detector_requires_speech_then_trailing_silence() -> None:
    detector = EnergyEndpointDetector(
        speech_threshold_dbfs=-40,
        minimum_speech_ms=200,
        trailing_silence_ms=800,
    )
    detections = []
    frame_end_ms = 0.0
    for rms in [0.0] * 10 + [0.1] * 15 + [0.0] * 40:
        frame_end_ms += 20
        detections.append(
            detector.observe(
                normalized_rms=rms,
                frame_duration_ms=20,
                frame_end_ms=frame_end_ms,
            )
        )

    assert sum(detections) == 1
    assert detector.endpoint_ms == 1300
    detector.reset()
    assert detector.endpoint_detected is False
    assert detector.endpoint_ms is None


def _criteria() -> Gate3Criteria:
    return Gate3Criteria(
        criteria_id="gate3-test",
        status="frozen_before_batch_outputs",
        frozen_date="2026-08-12",
        model_sha256="a" * 64,
        benchmark_manifest_sha256="b" * 64,
        expected_case_count=20,
        minimum_speaker_count=10,
        maximum_empty_final_count=0,
        minimum_nonempty_partial_rate=1,
        minimum_mean_partial_stability=0.98,
        maximum_trace_retraction_rate=0.05,
        minimum_exact_surface_agreement_rate=0.95,
        maximum_surface_cer=0.01,
        maximum_warm_first_partial_p95_ms=2500,
        maximum_finalization_lag_p95_ms=500,
        maximum_post_warmup_chunk_inference_p95_ms=320,
        maximum_trace_realtime_factor_p95=1.15,
        minimum_synthetic_endpoint_f1=0.9,
        maximum_endpoint_early_trigger_rate=0.1,
        maximum_endpoint_delay_p95_ms=1000,
        minimum_cancellation_correctness=1,
        minimum_reset_correctness=1,
        maximum_allocated_memory_growth_bytes=64 * 1024 * 1024,
        maximum_reserved_memory_growth_bytes=256 * 1024 * 1024,
        expected_adapter_module_count=24,
        require_every_adapter_called=True,
        require_every_session_state_released=True,
        confidence_policy="explicitly_unsupported_no_fabricated_values",
        quality_generalization_claim_allowed=False,
        endpoint_evidence_scope="synthetic_appended_silence_only",
    )


def _passing_evidence() -> Gate3Evidence:
    return Gate3Evidence(
        criteria_id="gate3-test",
        model_sha256="a" * 64,
        benchmark_manifest_sha256="b" * 64,
        case_count=20,
        unique_case_count=20,
        speaker_count=10,
        complete_trace_count=20,
        empty_final_count=0,
        nonempty_partial_rate=1,
        mean_partial_stability=1,
        trace_retraction_rate=0,
        exact_surface_agreement_rate=1,
        aggregate_surface_cer=0,
        warm_first_partial_p95_ms=2000,
        finalization_lag_p95_ms=200,
        post_warmup_chunk_inference_p95_ms=100,
        trace_realtime_factor_p95=1.05,
        synthetic_endpoint_precision=1,
        synthetic_endpoint_recall=1,
        synthetic_endpoint_f1=1,
        endpoint_early_trigger_rate=0,
        endpoint_delay_p95_ms=800,
        cancellation_correctness=1,
        reset_correctness=1,
        allocated_memory_growth_bytes=0,
        reserved_memory_growth_bytes=0,
        adapter_module_count=24,
        every_adapter_called=True,
        every_session_state_released=True,
        confidence_policy="explicitly_unsupported_no_fabricated_values",
    )


def test_gate3_assessment_applies_frozen_thresholds() -> None:
    passing = assess_gate3(_criteria(), _passing_evidence())
    assert passing.decision == "PASS"
    assert passing.failed_checks == ()

    failed_evidence = _passing_evidence().model_copy(
        update={"synthetic_endpoint_f1": 0.5, "empty_final_count": 1}
    )
    failing = assess_gate3(_criteria(), failed_evidence)
    assert failing.decision == "FAIL"
    assert set(failing.failed_checks) == {"empty_finals", "endpoint_f1"}
