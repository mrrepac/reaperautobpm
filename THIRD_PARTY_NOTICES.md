# Third-party notices

## tempo-cnn

This project's BPM engine is built on a format conversion (Keras → ONNX) of
the pretrained `deeptemp_k16` model from tempo-cnn, and vendors its feature
extraction code (see the header comment in `engine/src/tempocnn_engine.py`).

- Project: tempo-cnn
- Author: Hendrik Schreiber
- Model: `deeptemp_k16`
- Source project: <https://github.com/hendriks73/tempo-cnn>
- Original model file: `tempocnn/models/deeptemp_k16.h5` in that repository
- License: GNU Affero General Public License v3
- License text: <https://www.gnu.org/licenses/agpl-3.0.html>

Reference paper: Hendrik Schreiber, Meinard Müller, "A Single-Step Approach
to Musical Tempo Estimation Using a Convolutional Neural Network", ISMIR 2018.

SHA-256 of the shipped files:

`tempocnn_engine.exe`:
```text
2D4EB0498903C3C4060A7169490BCBC6067E83A6229CF2984F6ABC426288CF32
```

`deeptemp_k16.onnx` (converted from the original `.h5`, same license):
```text
4D41FCC576C0B510DFB807A5A19BC04A1D77EDF6AF9341A9F7982B06F1B921C5
```

## Bundled runtime dependencies

`engine/tempocnn_engine.exe` is a PyInstaller onefile bundle that embeds
its Python runtime dependencies, including the following third-party
libraries and their own licenses:

- **libsndfile** — LGPL-2.1-or-later —
  <https://github.com/libsndfile/libsndfile> —
  <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
  (bundled via the `soundfile` Python package's native binary)
- **librosa** — ISC License — <https://github.com/librosa/librosa>
- **soundfile** (Python wrapper) — BSD-3-Clause —
  <https://github.com/bastibe/python-soundfile>
- **NumPy** — BSD-3-Clause — <https://github.com/numpy/numpy>
- **SciPy** — BSD-3-Clause — <https://github.com/scipy/scipy>
- **numba** — BSD-2-Clause — <https://github.com/numba/numba>
- **llvmlite** — BSD-2-Clause — <https://github.com/numba/llvmlite>
- **ONNX Runtime** — MIT License —
  <https://github.com/microsoft/onnxruntime>

As LGPL-2.1-or-later requires that users be able to relink against a
modified version of the library, note that `libsndfile` is dynamically
loaded by `soundfile` at its embedded version; consult the project's
release notes at the link above for the exact version in use.
