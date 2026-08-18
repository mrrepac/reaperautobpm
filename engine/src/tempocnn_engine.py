"""TempoCNN-based BPM engine for the "Detect beat BPM" REAPER action.

Feature extraction (read_features/_ensure_length/_add_zeros/_to_sliding_window)
and normalization (std_normalizer) below are ported from tempo-cnn
(https://github.com/hendriks73/tempo-cnn), Copyright Hendrik Schreiber,
licensed under AGPLv3 (https://www.gnu.org/licenses/agpl-3.0.html).
The model file deeptemp_k16.onnx is a format conversion of that project's
deeptemp_k16.h5, same license.

Usage: tempocnn_engine.exe <audio_path>
Prints "bpm: <float>" and "confidence: <float 0-1>" to stdout on success
(exit 0). On failure, prints an error message and exits non-zero.
"""
import sys
from pathlib import Path

import numpy as np
import librosa
import onnxruntime as ort

SR = 11025
N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 40
FMIN = 20
FMAX = 5000
WINDOW_FRAMES = 256
WINDOW_HOP = 128
BPM_OFFSET = 30


def read_features(path: str) -> np.ndarray:
    y, _ = librosa.load(path, sr=SR)
    data = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH,
        power=1, n_mels=N_MELS, fmin=FMIN, fmax=FMAX,
    )
    data = np.reshape(data, (1, data.shape[0], data.shape[1], 1))
    if data.shape[2] < WINDOW_FRAMES:
        data = _ensure_length(data, WINDOW_FRAMES)
    return _to_sliding_window(data, WINDOW_FRAMES, WINDOW_HOP)


def _ensure_length(data: np.ndarray, length: int) -> np.ndarray:
    padded = np.zeros((1, data.shape[1], length, 1), dtype=data.dtype)
    padded[0, :, 0:data.shape[2], 0] = data[0, :, :, 0]
    return padded


def _to_sliding_window(data: np.ndarray, window_length: int, hop_length: int) -> np.ndarray:
    total_frames = data.shape[2]
    windows = []
    span = ((total_frames - window_length) // hop_length + 1) * hop_length
    for offset in range(0, span, hop_length):
        windows.append(np.copy(data[:, :, offset:window_length + offset, :]))
    return np.concatenate(windows, axis=0)


def std_normalizer(data: np.ndarray) -> np.ndarray:
    data = data.astype(np.float64)
    mean = np.mean(data)
    std = np.std(data)
    if std != 0.0:
        data = (data - mean) / std
    return data.astype(np.float32)


def quad_interpol_argmax(y: np.ndarray) -> tuple[float, float]:
    x = int(np.argmax(y))
    if x == 0 or x == y.shape[0] - 1:
        return float(x), float(y[x])
    z = np.polyfit([x - 1, x, x + 1], [y[x - 1], y[x], y[x + 1]], 2)
    argmax = -z[1] / (2.0 * z[0])
    height = z[2] - (z[1] ** 2.0) / (4.0 * z[0])
    return float(argmax), float(height)


def model_path() -> Path:
    return Path(sys.argv[0]).resolve().parent / "deeptemp_k16.onnx"


def estimate(audio_path: str) -> tuple[float, float]:
    data = read_features(audio_path)
    norm_data = std_normalizer(data)

    session = ort.InferenceSession(str(model_path()))
    input_name = session.get_inputs()[0].name
    prediction = session.run(None, {input_name: norm_data})[0]

    averaged = np.average(prediction, axis=0)
    index, height = quad_interpol_argmax(averaged)
    bpm = index + BPM_OFFSET
    confidence = max(0.0, min(1.0, height))
    return bpm, confidence


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: tempocnn_engine.exe <audio_path>", file=sys.stderr)
        return 2
    audio_path = sys.argv[1]
    if not Path(audio_path).is_file():
        print(f"audio file not found: {audio_path}", file=sys.stderr)
        return 1
    try:
        bpm, confidence = estimate(audio_path)
    except Exception as exc:  # noqa: BLE001 -- top-level CLI boundary
        print(f"tempocnn engine failed: {exc}", file=sys.stderr)
        return 1

    print(f"bpm: {bpm:.2f}")
    print(f"confidence: {confidence:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
