"""Generates synthetic click-track WAV fixtures for engine testing.

No copyrighted material: pure numpy-synthesized clicks at known BPM.
Run with any Python 3.9+ that has numpy installed (no project venv needed).
"""
import struct
import wave
from pathlib import Path

import numpy as np

SR = 22050
DURATION_S = 20.0
CLICK_MS = 15
BPMS = (90, 120, 140)

OUT_DIR = Path(__file__).parent


def make_click_track(bpm: int) -> np.ndarray:
    total_samples = int(SR * DURATION_S)
    audio = np.zeros(total_samples, dtype=np.float64)
    beat_interval_s = 60.0 / bpm
    click_samples = int(SR * CLICK_MS / 1000)
    t = np.linspace(0, CLICK_MS / 1000, click_samples, endpoint=False)
    # 2 kHz tone burst with a fast exponential decay envelope
    click = np.sin(2 * np.pi * 2000 * t) * np.exp(-t * 60)

    beat_time = 0.0
    while beat_time < DURATION_S:
        start = int(beat_time * SR)
        end = min(start + click_samples, total_samples)
        audio[start:end] += click[: end - start]
        beat_time += beat_interval_s

    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.9
    return audio


def write_wav(path: Path, audio: np.ndarray) -> None:
    pcm16 = (audio * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SR)
        f.writeframes(pcm16.tobytes())


if __name__ == "__main__":
    for bpm in BPMS:
        out_path = OUT_DIR / f"click_{bpm}bpm.wav"
        write_wav(out_path, make_click_track(bpm))
        print(f"wrote {out_path} ({bpm} BPM)")
