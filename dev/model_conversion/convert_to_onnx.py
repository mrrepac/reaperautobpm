"""One-time conversion of tempo-cnn's deeptemp_k16 Keras model to ONNX.

Run with the Python 3.11 venv in this directory (see README.md).
Source model: https://github.com/hendriks73/tempo-cnn (AGPLv3).
"""
from pathlib import Path

import tf2onnx
import tensorflow as tf
from tempocnn.classifier import TempoClassifier

OUTPUT_PATH = Path(__file__).resolve().parents[2] / "engine" / "deeptemp_k16.onnx"


def main() -> None:
    classifier = TempoClassifier(model_name="deeptemp_k16")
    keras_model = classifier.model
    print("Loaded Keras model:")
    keras_model.summary()

    input_signature = [
        tf.TensorSpec(keras_model.input_shape, tf.float32, name="mel_windows")
    ]
    onnx_model, _ = tf2onnx.convert.from_keras(
        keras_model, input_signature=input_signature, opset=13
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"Wrote {OUTPUT_PATH} ({OUTPUT_PATH.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
