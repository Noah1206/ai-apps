"""Model-neutral state tracking and metrics for Gate 3 Streaming ASR."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from statistics import fmean
from typing import Any

from busan_lab.evaluation.metrics import character_error_rate
from busan_lab.schemas.asr import ModelDescriptor
from busan_lab.schemas.streaming import (
    Gate3Assessment,
    Gate3Check,
    Gate3Criteria,
    Gate3Decision,
    Gate3Evidence,
    StreamingSessionStatus,
    StreamingTraceMetrics,
    StreamingTranscriptEvent,
)


class StreamingSessionError(RuntimeError):
    """Raised when chunks violate the streaming session lifecycle."""


class EnergyEndpointDetector:
    """Deterministic RMS endpoint baseline for file/replay engineering tests."""

    speech_observed_ms: float
    trailing_silence_observed_ms: float
    endpoint_detected: bool
    endpoint_ms: float | None

    def __init__(
        self,
        *,
        speech_threshold_dbfs: float = -42.0,
        minimum_speech_ms: float = 200.0,
        trailing_silence_ms: float = 800.0,
    ) -> None:
        if speech_threshold_dbfs >= 0:
            raise ValueError("speech_threshold_dbfs must be negative")
        if minimum_speech_ms <= 0 or trailing_silence_ms <= 0:
            raise ValueError("endpoint durations must be positive")
        self.speech_threshold = 10 ** (speech_threshold_dbfs / 20)
        self.minimum_speech_ms = minimum_speech_ms
        self.trailing_silence_ms = trailing_silence_ms
        self.reset()

    def observe(
        self, *, normalized_rms: float, frame_duration_ms: float, frame_end_ms: float
    ) -> bool:
        if not 0 <= normalized_rms <= 1:
            raise ValueError("normalized_rms must be between zero and one")
        if frame_duration_ms <= 0 or frame_end_ms < frame_duration_ms:
            raise ValueError("invalid endpoint frame timing")
        if self.endpoint_detected:
            return False
        if normalized_rms >= self.speech_threshold:
            self.speech_observed_ms += frame_duration_ms
            self.trailing_silence_observed_ms = 0.0
        elif self.speech_observed_ms >= self.minimum_speech_ms:
            self.trailing_silence_observed_ms += frame_duration_ms
        if self.trailing_silence_observed_ms >= self.trailing_silence_ms:
            self.endpoint_detected = True
            self.endpoint_ms = frame_end_ms
            return True
        return False

    def reset(self) -> None:
        self.speech_observed_ms = 0.0
        self.trailing_silence_observed_ms = 0.0
        self.endpoint_detected = False
        self.endpoint_ms = None


class StreamingTranscriptSession:
    """Track stable/unstable text without coupling the lab to one GPU runtime."""

    _status: StreamingSessionStatus
    _history: deque[str]
    _events: list[StreamingTranscriptEvent]
    _stable_prefix: str
    _stable_prefix_violations: int
    _audio_end_ms: float

    def __init__(
        self,
        *,
        session_id: str,
        model: ModelDescriptor,
        stabilization_window: int = 3,
    ) -> None:
        if not session_id.strip():
            raise ValueError("session_id must not be blank")
        if stabilization_window < 2:
            raise ValueError("stabilization_window must be at least 2")
        self.session_id = session_id
        self.model = model
        self.stabilization_window = stabilization_window
        self.generation = 0
        self._reset_state()

    @property
    def status(self) -> StreamingSessionStatus:
        return self._status

    @property
    def events(self) -> tuple[StreamingTranscriptEvent, ...]:
        return tuple(self._events)

    def observe(
        self,
        transcript: str,
        *,
        audio_end_ms: float,
        emitted_at_ms: float,
        inference_latency_ms: float,
        endpoint_detected: bool = False,
        final: bool = False,
        confidence: float | None = None,
    ) -> StreamingTranscriptEvent:
        if self._status is not StreamingSessionStatus.ACTIVE:
            raise StreamingSessionError(
                f"session {self.session_id!r} is {self._status.value}; reset before observing"
            )
        if audio_end_ms < self._audio_end_ms:
            raise StreamingSessionError("audio_end_ms must be monotonic")

        audio_start_ms = self._audio_end_ms
        self._history.append(transcript)
        stable_prefix = transcript if final else self._candidate_stable_prefix()
        if self._stable_prefix and not stable_prefix.startswith(self._stable_prefix):
            self._stable_prefix_violations += 1
        self._stable_prefix = stable_prefix
        unstable_suffix = "" if final else transcript[len(stable_prefix) :]
        stability = _stability_ratio(stable_prefix, transcript)
        event = StreamingTranscriptEvent(
            session_id=self.session_id,
            generation=self.generation,
            sequence=len(self._events),
            partial=not final,
            stable_prefix=stable_prefix,
            unstable_suffix=unstable_suffix,
            transcript=transcript,
            stability=stability,
            audio_start_ms=audio_start_ms,
            audio_end_ms=audio_end_ms,
            emitted_at_ms=emitted_at_ms,
            inference_latency_ms=inference_latency_ms,
            endpoint_detected=endpoint_detected,
            confidence=confidence,
            confidence_supported=confidence is not None,
            model=self.model,
        )
        self._events.append(event)
        self._audio_end_ms = audio_end_ms
        if final:
            self._status = StreamingSessionStatus.FINALIZED
        return event

    def cancel(self) -> None:
        if self._status is not StreamingSessionStatus.ACTIVE:
            raise StreamingSessionError(f"cannot cancel a {self._status.value} session")
        self._status = StreamingSessionStatus.CANCELLED

    def reset(self) -> None:
        self.generation += 1
        self._reset_state()

    def metrics(
        self,
        *,
        surface_final_transcript: str | None = None,
        peak_device_memory_bytes: int | None = None,
    ) -> StreamingTraceMetrics:
        final_event = next((event for event in reversed(self._events) if not event.partial), None)
        final_transcript = final_event.transcript if final_event is not None else ""
        stable_events = [event for event in self._events if event.partial and event.stable_prefix]
        stable_violations = self._stable_prefix_violations
        if final_event is not None:
            stable_violations = sum(
                not final_transcript.startswith(event.stable_prefix) for event in stable_events
            )
        partial_stability = 1 - stable_violations / len(stable_events) if stable_events else 1.0
        partial_events = [event for event in self._events if event.partial]
        first_partial = next((event for event in partial_events if event.transcript), None)
        chunk_latencies = [event.inference_latency_ms for event in self._events]
        surface_cer: float | None = None
        final_agreement: float | None = None
        if surface_final_transcript is not None and final_event is not None:
            surface_cer, _ = character_error_rate(surface_final_transcript, final_transcript)
            final_agreement = max(0.0, 1.0 - surface_cer)
        return StreamingTraceMetrics(
            session_id=self.session_id,
            generation=self.generation,
            event_count=len(self._events),
            partial_event_count=len(partial_events),
            endpoint_count=sum(event.endpoint_detected for event in self._events),
            audio_duration_ms=self._audio_end_ms,
            first_partial_latency_ms=(
                first_partial.emitted_at_ms if first_partial is not None else None
            ),
            final_latency_ms=(final_event.emitted_at_ms if final_event is not None else None),
            mean_chunk_inference_latency_ms=(fmean(chunk_latencies) if chunk_latencies else None),
            stable_prefix_observations=len(stable_events),
            stable_prefix_violations=stable_violations,
            partial_stability=partial_stability,
            final_transcript=final_transcript,
            surface_final_cer=surface_cer,
            final_agreement_with_surface_asr=final_agreement,
            peak_device_memory_bytes=peak_device_memory_bytes,
        )

    def _candidate_stable_prefix(self) -> str:
        if len(self._history) < self.stabilization_window:
            return ""
        return _longest_common_prefix(tuple(self._history))

    def _reset_state(self) -> None:
        self._status = StreamingSessionStatus.ACTIVE
        self._history = deque(maxlen=self.stabilization_window)
        self._events = []
        self._stable_prefix = ""
        self._stable_prefix_violations = 0
        self._audio_end_ms = 0.0


def _longest_common_prefix(values: Sequence[str]) -> str:
    if not values:
        return ""
    shortest = min(values, key=len)
    for index, character in enumerate(shortest):
        if any(value[index] != character for value in values):
            return shortest[:index]
    return shortest


def _stability_ratio(stable_prefix: str, transcript: str) -> float:
    if not transcript:
        return 1.0
    return len(stable_prefix) / len(transcript)


def assess_gate3(criteria: Gate3Criteria, evidence: Gate3Evidence) -> Gate3Assessment:
    """Apply only the thresholds frozen before batch output inspection."""

    checks: list[Gate3Check] = []

    def add(name: str, passed: bool, observed: Any, requirement: Any) -> None:
        checks.append(
            Gate3Check(
                name=name,
                passed=passed,
                observed=observed,
                requirement=requirement,
            )
        )

    add(
        "criteria_identity",
        evidence.criteria_id == criteria.criteria_id,
        evidence.criteria_id,
        criteria.criteria_id,
    )
    add(
        "model_identity",
        evidence.model_sha256 == criteria.model_sha256,
        evidence.model_sha256,
        criteria.model_sha256,
    )
    add(
        "benchmark_identity",
        evidence.benchmark_manifest_sha256 == criteria.benchmark_manifest_sha256,
        evidence.benchmark_manifest_sha256,
        criteria.benchmark_manifest_sha256,
    )
    add(
        "case_count",
        evidence.case_count == criteria.expected_case_count,
        evidence.case_count,
        criteria.expected_case_count,
    )
    add(
        "unique_cases",
        evidence.unique_case_count == evidence.case_count,
        evidence.unique_case_count,
        evidence.case_count,
    )
    add(
        "speaker_count",
        evidence.speaker_count >= criteria.minimum_speaker_count,
        evidence.speaker_count,
        criteria.minimum_speaker_count,
    )
    add(
        "complete_traces",
        evidence.complete_trace_count == criteria.expected_case_count,
        evidence.complete_trace_count,
        criteria.expected_case_count,
    )
    add(
        "empty_finals",
        evidence.empty_final_count <= criteria.maximum_empty_final_count,
        evidence.empty_final_count,
        criteria.maximum_empty_final_count,
    )
    add(
        "nonempty_partials",
        evidence.nonempty_partial_rate >= criteria.minimum_nonempty_partial_rate,
        evidence.nonempty_partial_rate,
        criteria.minimum_nonempty_partial_rate,
    )
    add(
        "partial_stability",
        evidence.mean_partial_stability >= criteria.minimum_mean_partial_stability,
        evidence.mean_partial_stability,
        criteria.minimum_mean_partial_stability,
    )
    add(
        "stable_prefix_retractions",
        evidence.trace_retraction_rate <= criteria.maximum_trace_retraction_rate,
        evidence.trace_retraction_rate,
        criteria.maximum_trace_retraction_rate,
    )
    add(
        "exact_surface_agreement",
        evidence.exact_surface_agreement_rate >= criteria.minimum_exact_surface_agreement_rate,
        evidence.exact_surface_agreement_rate,
        criteria.minimum_exact_surface_agreement_rate,
    )
    add(
        "surface_cer",
        evidence.aggregate_surface_cer <= criteria.maximum_surface_cer,
        evidence.aggregate_surface_cer,
        criteria.maximum_surface_cer,
    )
    add(
        "first_partial_latency",
        evidence.warm_first_partial_p95_ms <= criteria.maximum_warm_first_partial_p95_ms,
        evidence.warm_first_partial_p95_ms,
        criteria.maximum_warm_first_partial_p95_ms,
    )
    add(
        "finalization_lag",
        evidence.finalization_lag_p95_ms <= criteria.maximum_finalization_lag_p95_ms,
        evidence.finalization_lag_p95_ms,
        criteria.maximum_finalization_lag_p95_ms,
    )
    add(
        "chunk_inference_latency",
        evidence.post_warmup_chunk_inference_p95_ms
        <= criteria.maximum_post_warmup_chunk_inference_p95_ms,
        evidence.post_warmup_chunk_inference_p95_ms,
        criteria.maximum_post_warmup_chunk_inference_p95_ms,
    )
    add(
        "realtime_factor",
        evidence.trace_realtime_factor_p95 <= criteria.maximum_trace_realtime_factor_p95,
        evidence.trace_realtime_factor_p95,
        criteria.maximum_trace_realtime_factor_p95,
    )
    add(
        "endpoint_f1",
        evidence.synthetic_endpoint_f1 >= criteria.minimum_synthetic_endpoint_f1,
        evidence.synthetic_endpoint_f1,
        criteria.minimum_synthetic_endpoint_f1,
    )
    add(
        "endpoint_early_trigger",
        evidence.endpoint_early_trigger_rate <= criteria.maximum_endpoint_early_trigger_rate,
        evidence.endpoint_early_trigger_rate,
        criteria.maximum_endpoint_early_trigger_rate,
    )
    add(
        "endpoint_delay",
        evidence.endpoint_delay_p95_ms <= criteria.maximum_endpoint_delay_p95_ms,
        evidence.endpoint_delay_p95_ms,
        criteria.maximum_endpoint_delay_p95_ms,
    )
    add(
        "cancellation",
        evidence.cancellation_correctness >= criteria.minimum_cancellation_correctness,
        evidence.cancellation_correctness,
        criteria.minimum_cancellation_correctness,
    )
    add(
        "reset",
        evidence.reset_correctness >= criteria.minimum_reset_correctness,
        evidence.reset_correctness,
        criteria.minimum_reset_correctness,
    )
    add(
        "allocated_memory_growth",
        evidence.allocated_memory_growth_bytes <= criteria.maximum_allocated_memory_growth_bytes,
        evidence.allocated_memory_growth_bytes,
        criteria.maximum_allocated_memory_growth_bytes,
    )
    add(
        "reserved_memory_growth",
        evidence.reserved_memory_growth_bytes <= criteria.maximum_reserved_memory_growth_bytes,
        evidence.reserved_memory_growth_bytes,
        criteria.maximum_reserved_memory_growth_bytes,
    )
    add(
        "adapter_modules",
        evidence.adapter_module_count == criteria.expected_adapter_module_count,
        evidence.adapter_module_count,
        criteria.expected_adapter_module_count,
    )
    add(
        "every_adapter_called",
        evidence.every_adapter_called is criteria.require_every_adapter_called,
        evidence.every_adapter_called,
        criteria.require_every_adapter_called,
    )
    add(
        "session_state_release",
        evidence.every_session_state_released is criteria.require_every_session_state_released,
        evidence.every_session_state_released,
        criteria.require_every_session_state_released,
    )
    add(
        "confidence_policy",
        evidence.confidence_policy == criteria.confidence_policy,
        evidence.confidence_policy,
        criteria.confidence_policy,
    )
    add(
        "infrastructure",
        not evidence.infrastructure_failures,
        len(evidence.infrastructure_failures),
        0,
    )
    failed = tuple(check.name for check in checks if not check.passed)
    passed = tuple(check.name for check in checks if check.passed)
    return Gate3Assessment(
        criteria_id=criteria.criteria_id,
        decision=Gate3Decision.PASS if not failed else Gate3Decision.FAIL,
        checks=tuple(checks),
        passed_checks=passed,
        failed_checks=failed,
    )
