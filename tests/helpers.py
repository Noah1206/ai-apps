"""Test fixtures that generate audio without external files."""

from __future__ import annotations

import io
import math
import wave


def make_wav_bytes(
    *,
    sample_rate: int = 44_100,
    channels: int = 2,
    duration_seconds: float = 0.6,
    frequency_hz: float = 220,
    amplitude: float = 0.25,
) -> bytes:
    frame_count = int(sample_rate * duration_seconds)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        frames = bytearray()
        for index in range(frame_count):
            value = int(
                32767 * amplitude * math.sin(2 * math.pi * frequency_hz * index / sample_rate)
            )
            frame = value.to_bytes(2, byteorder="little", signed=True)
            frames.extend(frame * channels)
        wav.writeframes(bytes(frames))
    return buffer.getvalue()
