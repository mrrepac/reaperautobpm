"""Verifies the ONNX conversion matches the original Keras model numerically.

Two checks:
1. Same input tensor -> Keras output vs ONNX output must be near-identical.
2. End-to-end BPM on synthetic fixtures must match between the original
   TempoClassifier (Keras) and onnxruntime, using tempo-cnn's own feature
   extraction for both (this script does NOT test the vendored engine copy
   of feature extraction -- that's Task 6).
"""
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tempocnn.classifier import TempoClassifier, std_normalizer
from tempocnn.feature import read_features

REPO_ROOT = Path(__file__).resolve().parents[2]
ONNX_PATH = REPO_ROOT / "engine" / "deeptemp_k16.onnx"
FIXTURES = {
    "click_90bpm.wav": 90,
    "click_120bpm.wav": 120,
    "click_140bpm.wav": 140,
}


def main() -> None:
    classifier = TempoClassifier(model_name="deeptemp_k16")
    session = ort.InferenceSession(str(ONNX_PATH))
    input_name = session.get_inputs()[0].name

    any_failure = False

    # Check 1: raw tensor comparison on the first fixture's features.
    first_file = REPO_ROOT / "dev" / "test_audio" / next(iter(FIXTURES))
    data = read_features(str(first_file))
    norm_data = std_normalizer(data).astype(np.float32)

    keras_out = classifier.model.predict(norm_data, norm_data.shape[0])
    onnx_out = session.run(None, {input_name: norm_data})[0]

    max_abs_diff = float(np.max(np.abs(keras_out - onnx_out)))
    print(f"[tensor parity] max abs diff = {max_abs_diff:.6f}")
    if max_abs_diff > 1e-3:
        print("[tensor parity] FAIL: difference too large")
        any_failure = True
    else:
        print("[tensor parity] OK")

    # Check 2: end-to-end BPM comparison on all fixtures.
    for filename, expected_bpm in FIXTURES.items():
        path = REPO_ROOT / "dev" / "test_audio" / filename
        data = read_features(str(path))
        norm_data = std_normalizer(data).astype(np.float32)

        keras_bpm = classifier.estimate_tempo(data, interpolate=True)

        onnx_pred = session.run(None, {input_name: norm_data})[0]
        onnx_avg = np.average(onnx_pred, axis=0)
        onnx_index, _ = TempoClassifier.quad_interpol_argmax(onnx_avg)
        onnx_bpm = onnx_index + 30

        diff = abs(keras_bpm - onnx_bpm)
        print(
            f"[{filename}] keras={keras_bpm:.2f} onnx={onnx_bpm:.2f} "
            f"synth_bpm={expected_bpm} diff={diff:.4f}"
        )
        if diff > 0.5:
            print(f"[{filename}] FAIL: keras/onnx BPM diverge")
            any_failure = True

    if any_failure:
        print("RESULT: FAIL")
        raise SystemExit(1)
    print("RESULT: OK")


if __name__ == "__main__":
    main()
