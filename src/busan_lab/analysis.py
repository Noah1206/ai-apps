"""Lightweight waveform, log-Mel, and exploratory F0 extraction."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from busan_lab.audio import read_pcm16_wav
from busan_lab.schemas.audio import AcousticSnapshot


def analyze_audio(
    path: Path,
    *,
    max_seconds: float = 60,
    waveform_bins: int = 900,
    mel_bins: int = 48,
) -> AcousticSnapshot:
    samples = read_pcm16_wav(path)
    sample_rate = 16_000
    max_samples = int(max_seconds * sample_rate)
    truncated = samples.size > max_samples
    analyzed = samples[:max_samples]

    waveform_times, waveform_min, waveform_max = _waveform_envelope(
        analyzed, sample_rate, waveform_bins
    )
    mel_times, mel_frequencies, mel_db = _log_mel(analyzed, sample_rate, mel_bins=mel_bins)
    f0_times, f0_hz, f0_confidence = _estimate_f0(analyzed, sample_rate)

    return AcousticSnapshot(
        analyzed_duration_ms=analyzed.size / sample_rate * 1000,
        truncated=truncated,
        waveform_times_ms=tuple(float(value) for value in waveform_times),
        waveform_min=tuple(float(value) for value in waveform_min),
        waveform_max=tuple(float(value) for value in waveform_max),
        mel_times_ms=tuple(float(value) for value in mel_times),
        mel_frequencies_hz=tuple(float(value) for value in mel_frequencies),
        mel_db=tuple(tuple(float(value) for value in frequency_bin) for frequency_bin in mel_db),
        f0_times_ms=tuple(float(value) for value in f0_times),
        f0_hz=tuple(None if value is None else float(value) for value in f0_hz),
        f0_confidence=tuple(float(value) for value in f0_confidence),
    )


def _waveform_envelope(
    samples: NDArray[np.float32],
    sample_rate: int,
    bins: int,
) -> tuple[NDArray[np.float64], NDArray[np.float32], NDArray[np.float32]]:
    bin_count = max(1, min(bins, samples.size))
    boundaries = np.linspace(0, samples.size, bin_count + 1, dtype=np.int64)
    minimum = np.empty(bin_count, dtype=np.float32)
    maximum = np.empty(bin_count, dtype=np.float32)
    for index in range(bin_count):
        segment = samples[boundaries[index] : boundaries[index + 1]]
        minimum[index] = np.min(segment)
        maximum[index] = np.max(segment)
    centers = (boundaries[:-1] + boundaries[1:]) / 2
    return centers / sample_rate * 1000, minimum, maximum


def _log_mel(
    samples: NDArray[np.float32],
    sample_rate: int,
    *,
    mel_bins: int,
    fft_size: int = 512,
    window_size: int = 400,
    hop_size: int = 160,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    if samples.size < window_size:
        samples = np.pad(samples, (0, window_size - samples.size))
    frame_count = 1 + (samples.size - window_size) // hop_size
    indices = (
        np.arange(window_size, dtype=np.int64)[None, :]
        + hop_size * np.arange(frame_count, dtype=np.int64)[:, None]
    )
    frames = samples[indices] * np.hanning(window_size)
    spectrum = np.abs(np.fft.rfft(frames, n=fft_size, axis=1)) ** 2
    filterbank, centers_hz = _mel_filterbank(sample_rate, fft_size, mel_bins)
    mel_power = spectrum @ filterbank.T
    mel_db = 10 * np.log10(np.maximum(mel_power, 1e-12))
    mel_db -= np.max(mel_db)
    mel_db = np.maximum(mel_db, -80)
    times_ms = (np.arange(frame_count) * hop_size + window_size / 2) / sample_rate * 1000
    return times_ms, centers_hz, mel_db.T


def _mel_filterbank(
    sample_rate: int,
    fft_size: int,
    mel_bins: int,
    minimum_hz: float = 40,
    maximum_hz: float = 7600,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    def hz_to_mel(value: float | NDArray[np.float64]) -> float | NDArray[np.float64]:
        return 2595 * np.log10(1 + np.asarray(value) / 700)

    def mel_to_hz(value: NDArray[np.float64]) -> NDArray[np.float64]:
        return 700 * (10 ** (value / 2595) - 1)

    mel_points = np.linspace(
        float(hz_to_mel(minimum_hz)),
        float(hz_to_mel(maximum_hz)),
        mel_bins + 2,
    )
    hz_points = mel_to_hz(mel_points)
    fft_frequencies = np.fft.rfftfreq(fft_size, d=1 / sample_rate)
    filters = np.zeros((mel_bins, fft_frequencies.size), dtype=np.float64)
    for index in range(mel_bins):
        left, center, right = hz_points[index : index + 3]
        lower = (fft_frequencies - left) / max(center - left, 1e-9)
        upper = (right - fft_frequencies) / max(right - center, 1e-9)
        filters[index] = np.maximum(0, np.minimum(lower, upper))
    return filters, hz_points[1:-1]


def _estimate_f0(
    samples: NDArray[np.float32],
    sample_rate: int,
    *,
    minimum_hz: float = 65,
    maximum_hz: float = 450,
    window_size: int = 640,
    hop_size: int = 320,
) -> tuple[NDArray[np.float64], list[float | None], NDArray[np.float64]]:
    if samples.size < window_size:
        samples = np.pad(samples, (0, window_size - samples.size))
    frame_count = 1 + (samples.size - window_size) // hop_size
    times = (np.arange(frame_count) * hop_size + window_size / 2) / sample_rate * 1000
    minimum_lag = max(1, math.floor(sample_rate / maximum_hz))
    maximum_lag = min(window_size - 1, math.ceil(sample_rate / minimum_hz))
    frequencies: list[float | None] = []
    confidences = np.zeros(frame_count, dtype=np.float64)

    window = np.hanning(window_size)
    for index in range(frame_count):
        frame = samples[index * hop_size : index * hop_size + window_size]
        windowed = (frame - np.mean(frame)) * window
        rms = float(np.sqrt(np.mean(windowed**2)))
        if rms < 0.008:
            frequencies.append(None)
            continue
        autocorrelation = np.correlate(windowed, windowed, mode="full")[window_size - 1 :]
        zero_lag = float(autocorrelation[0])
        if zero_lag <= 1e-9:
            frequencies.append(None)
            continue
        search = autocorrelation[minimum_lag : maximum_lag + 1]
        best_offset = int(np.argmax(search))
        best_lag = best_offset + minimum_lag
        confidence = float(search[best_offset] / zero_lag)
        confidences[index] = max(0, min(confidence, 1))
        frequencies.append(sample_rate / best_lag if confidence >= 0.3 else None)
    return times, frequencies, confidences
