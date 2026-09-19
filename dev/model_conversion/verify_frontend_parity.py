"""Verifies the engine's numpy front end matches librosa's, sample for sample.

The engine used to call librosa for decoding, resampling and the mel
spectrogram. librosa drags in numba, scipy and joblib, which dominated both
the frozen build's size and its import time, so the spectrogram is now spelled
out in numpy and only soundfile/soxr remain -- which is what librosa itself
calls underneath.

That trade is only safe if the samples reaching the model are unchanged, so
this script asserts it directly, on every fixture:

1. `load_audio` vs `librosa.load(sr=11025)` -- bit-identical.
2. the same, with an offset and duration applied.
3. `mel_spectrogram` vs `librosa.feature.melspectrogram(power=1, ...)`.
4. the mel filter bank vs `librosa.filters.mel(norm="slaney")`.
5. end-to-end BPM through onnxruntime, engine front end vs librosa front end.

Run with the parity venv (see README), which is the only place librosa is
installed -- it is a dev-only dependency and must not reach engine/src.
"""
import sys
from pathlib import Path

import librosa
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "engine" / "src"))

import tempocnn_engine as engine  # noqa: E402 -- needs the path above

FIXTURES = sorted((REPO_ROOT / "dev" / "test_audio").glob("*.wav"))

# Bit-identical is the bar for the audio path and the filter bank. The mel
# spectrogram is a float32 matrix product against librosa's einsum, so the last
# bit differs on a ten-thousandth of the cells; observed relative error is
# 2e-13. The bar is set just loose enough to allow that and nothing else.
MEL_TOLERANCE = 1e-9
failures = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}"
          + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        failures.append(name)


def librosa_mel(path: str, offset: float = 0.0,
                duration: float | None = None) -> np.ndarray:
    samples, _ = librosa.load(path, sr=engine.SR, offset=offset, duration=duration)
    return librosa.feature.melspectrogram(
        y=samples, sr=engine.SR, n_fft=engine.N_FFT, hop_length=engine.HOP_LENGTH,
        power=1, n_mels=engine.N_MELS, fmin=engine.FMIN, fmax=engine.FMAX,
    )


def bpm_from_spectrogram(spectrogram: np.ndarray, session, input_name: str) -> float:
    data = np.reshape(spectrogram, (1, spectrogram.shape[0], spectrogram.shape[1], 1))
    if data.shape[2] < engine.WINDOW_FRAMES:
        data = engine._ensure_length(data, engine.WINDOW_FRAMES)
    windows = engine._to_sliding_window(data, engine.WINDOW_FRAMES, engine.WINDOW_HOP)
    prediction = session.run(
        None, {input_name: engine.std_normalizer(windows)})[0]
    index, _ = engine.quad_interpol_argmax(np.average(prediction, axis=0))
    return index + engine.BPM_OFFSET


def main() -> None:
    import onnxruntime as ort

    print("Filter bank")
    reference_bank = librosa.filters.mel(
        sr=engine.SR, n_fft=engine.N_FFT, n_mels=engine.N_MELS,
        fmin=engine.FMIN, fmax=engine.FMAX)
    own_bank = engine.mel_filterbank()
    check("mel filter bank bit-identical",
          np.array_equal(own_bank, reference_bank),
          f"max diff {np.max(np.abs(own_bank - reference_bank)):.3e}")

    session = ort.InferenceSession(str(engine.model_path()),
                                  providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    for fixture in FIXTURES:
        print(fixture.name)
        path = str(fixture)

        own_samples = engine.load_audio(path)
        reference_samples, _ = librosa.load(path, sr=engine.SR)
        check("decode+resample bit-identical",
              np.array_equal(own_samples, reference_samples),
              f"shapes {own_samples.shape} vs {reference_samples.shape}")

        # A region out of the middle: the case the plugin now always uses.
        total = len(reference_samples) / engine.SR
        if total > 3.0:
            offset, duration = 1.0, total - 2.0
            own_region = engine.load_audio(path, offset=offset, duration=duration)
            reference_region, _ = librosa.load(
                path, sr=engine.SR, offset=offset, duration=duration)
            check("offset/duration region bit-identical",
                  np.array_equal(own_region, reference_region),
                  f"shapes {own_region.shape} vs {reference_region.shape}")

        own_mel = engine.mel_spectrogram(own_samples)
        reference_mel = librosa_mel(path)
        same_shape = own_mel.shape == reference_mel.shape
        check("mel spectrogram shape", same_shape,
              f"{own_mel.shape} vs {reference_mel.shape}")
        if same_shape:
            difference = np.max(np.abs(own_mel - reference_mel))
            relative = difference / max(float(np.max(reference_mel)), 1e-9)
            check("mel spectrogram values", relative < MEL_TOLERANCE,
                  f"relative max diff {relative:.3e}")

            own_bpm = bpm_from_spectrogram(own_mel, session, input_name)
            reference_bpm = bpm_from_spectrogram(reference_mel, session, input_name)
            check("end-to-end BPM identical",
                  abs(own_bpm - reference_bpm) < 1e-6,
                  f"{own_bpm:.6f} vs {reference_bpm:.6f}")

    if failures:
        print(f"RESULT: FAIL ({len(failures)}): " + ", ".join(sorted(set(failures))))
        raise SystemExit(1)
    print("RESULT: OK")


if __name__ == "__main__":
    main()
