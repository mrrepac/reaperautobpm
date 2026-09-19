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

`deeptemp_k16.onnx` (converted from the original `.h5`, same license):
```text
4D41FCC576C0B510DFB807A5A19BC04A1D77EDF6AF9341A9F7982B06F1B921C5
```

`tempocnn_engine/tempocnn_engine.exe`:
```text
F228089572025EA432D021601DC416A32D599B1431AF91468E9DC6311385B963
```

The model hash is the stable one: it identifies the converted weights and does
not depend on how the engine was built. The executable's hash belongs to this
particular PyInstaller build and will differ after any rebuild, including one
from unchanged sources — see `dev/model_conversion/README.md`.

## Bundled runtime dependencies

`engine/tempocnn_engine/` is a PyInstaller directory bundle: the executable
plus the Python runtime and its dependencies in `_internal/`, including the
following third-party libraries and their own licenses:

- **libsndfile** — LGPL-2.1-or-later —
  <https://github.com/libsndfile/libsndfile> —
  <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
  (bundled via the `soundfile` Python package's native binary)
- **libsoxr** — LGPL-2.1-or-later — <https://sourceforge.net/projects/soxr/>
  (bundled via the `soxr` Python package's native binary, which also embeds
  **PFFFT** — BSD-like, see `LICENSE-PFFFT.txt` in that package)
- **soundfile** (Python wrapper) — BSD-3-Clause —
  <https://github.com/bastibe/python-soundfile>
- **soxr** (Python wrapper) — LGPL-2.1-or-later —
  <https://github.com/dofuuz/python-soxr>
- **NumPy** — BSD-3-Clause — <https://github.com/numpy/numpy>
- **ONNX Runtime** — MIT License —
  <https://github.com/microsoft/onnxruntime>

librosa (ISC), SciPy, numba and llvmlite were listed here through v2.1.0 and
are no longer bundled: decoding and resampling call soundfile and soxr
directly, and the mel spectrogram is computed in numpy. See
`dev/model_conversion/verify_frontend_parity.py`, which asserts the result is
unchanged.

As LGPL-2.1-or-later requires that users be able to relink against a
modified version of the library, note that `libsndfile` and `libsoxr` are
loaded by their Python wrappers at the versions embedded in
`engine/tempocnn_engine/_internal/`; consult each project's release notes at
the links above for the exact versions in use.
