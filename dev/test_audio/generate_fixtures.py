"""Generates synthetic audio fixtures for engine testing.

No copyrighted material: everything here is numpy-synthesized.
Run with any Python 3.9+ that has numpy installed (no project venv needed).

Three families:

* `click_<bpm>bpm.wav` -- bare click tracks, 20 s. The half-BPM ones exist
  because the whole point of the 0.5 step is telling 112.5 from 113.
* `groove_112_5bpm.wav` -- kick/snare/hat/bass instead of identical clicks, so
  the fine refinement is tested against an uneven onset pattern rather than a
  pure impulse train it cannot help but lock onto.
* `loop_16beats_112_5bpm.wav` -- trimmed to exactly 16 beats, which is what
  the engine's loop snapping is for.
"""
import wave
from pathlib import Path

import numpy as np

SR = 22050
DURATION_S = 20.0
CLICK_MS = 15
BPMS = (90, 120, 140, 112.5, 87.5)

OUT_DIR = Path(__file__).parent


def fixture_name(prefix: str, bpm: float, suffix: str = "") -> str:
    """112.5 -> click_112_5bpm.wav; 90 -> click_90bpm.wav (unchanged)."""
    text = f"{bpm:g}".replace(".", "_")
    return f"{prefix}_{suffix}{text}bpm.wav"


def _click(frequency: float, decay: float, length_ms: float) -> np.ndarray:
    samples = int(SR * length_ms / 1000)
    t = np.linspace(0, length_ms / 1000, samples, endpoint=False)
    return np.sin(2 * np.pi * frequency * t) * np.exp(-t * decay)


def _place(audio: np.ndarray, sound: np.ndarray, at_second: float,
           gain: float = 1.0) -> None:
    start = int(at_second * SR)
    if start >= audio.size:
        return
    end = min(start + sound.size, audio.size)
    audio[start:end] += sound[: end - start] * gain


def _normalize(audio: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(audio))
    return audio / peak * 0.9 if peak > 0 else audio


def make_click_track(bpm: float, duration_s: float = DURATION_S) -> np.ndarray:
    audio = np.zeros(int(SR * duration_s), dtype=np.float64)
    click = _click(2000, 60, CLICK_MS)
    interval = 60.0 / bpm
    beat_time = 0.0
    while beat_time < duration_s:
        _place(audio, click, beat_time)
        beat_time += interval
    return _normalize(audio)


def make_groove(bpm: float, duration_s: float = DURATION_S) -> np.ndarray:
    """A plain 4/4 pattern: the onsets differ in weight and spacing, which is
    what a real beat does and a click track does not."""
    audio = np.zeros(int(SR * duration_s), dtype=np.float64)
    beat = 60.0 / bpm
    kick = _click(58, 22, 220)
    snare = _click(190, 38, 150) + 0.6 * _click(1400, 55, 150)
    hat = _click(7200, 220, 45)

    bar = 0
    while bar * 4 * beat < duration_s:
        bar_start = bar * 4 * beat
        _place(audio, kick, bar_start, 1.0)
        _place(audio, kick, bar_start + 2 * beat, 0.85)
        _place(audio, snare, bar_start + beat, 0.7)
        _place(audio, snare, bar_start + 3 * beat, 0.7)
        for eighth in range(8):
            _place(audio, hat, bar_start + eighth * beat / 2,
                   0.30 if eighth % 2 else 0.18)
        # A held bass note per bar, so the spectrum is not onsets alone.
        note_samples = int(SR * 4 * beat)
        t = np.arange(note_samples) / SR
        note = 0.35 * np.sin(2 * np.pi * 55 * t) * np.exp(-t * 1.1)
        _place(audio, note, bar_start, 1.0)
        bar += 1
    return _normalize(audio)


def write_wav(path: Path, audio: np.ndarray) -> None:
    pcm16 = (audio * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SR)
        f.writeframes(pcm16.tobytes())


if __name__ == "__main__":
    for bpm in BPMS:
        path = OUT_DIR / fixture_name("click", bpm)
        write_wav(path, make_click_track(bpm))
        print(f"wrote {path.name} ({bpm:g} BPM click track)")

    groove_bpm = 112.5
    path = OUT_DIR / fixture_name("groove", groove_bpm)
    write_wav(path, make_groove(groove_bpm))
    print(f"wrote {path.name} ({groove_bpm:g} BPM groove)")

    # Exactly 16 beats, the case loop snapping exists for.
    beats = 16
    loop_length = beats * 60.0 / groove_bpm
    path = OUT_DIR / f"loop_{beats}beats_112_5bpm.wav"
    write_wav(path, make_groove(groove_bpm, duration_s=loop_length))
    print(f"wrote {path.name} ({beats} beats, {loop_length:.4f} s)")
