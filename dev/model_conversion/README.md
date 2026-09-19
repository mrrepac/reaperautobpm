# Model conversion and engine build (dev-only)

Two one-time steps, in two separate virtual environments. End users never run
anything in this directory.

## 1. Model: Keras → ONNX

Produces `engine/deeptemp_k16.onnx` from the published tempo-cnn Keras weights.

Requires Python 3.11 (TensorFlow 2.17.0 has no wheel for newer Pythons at
the time this was written). Set up once:

    py -3.11 -m venv dev/model_conversion/.venv311
    dev/model_conversion/.venv311/Scripts/python.exe -m pip install -r dev/model_conversion/requirements.txt

Then run, in order:

    dev/model_conversion/.venv311/Scripts/python.exe dev/model_conversion/convert_to_onnx.py
    dev/model_conversion/.venv311/Scripts/python.exe dev/model_conversion/verify_parity.py

Both must print `OK` at the end. See
`docs/superpowers/plans/2026-08-18-tempocnn-engine.md` for what each step
checks.

### Version notes

`requirements.txt` pins `onnx==1.16.1` rather than a loose `onnx>=1.16.0`.
Left unpinned, pip resolves onnx 1.19.x, which requires an `ml_dtypes`
build exposing `float4_e2m1fn`; that conflicts with
`tensorflow-intel==2.17.0`'s `ml_dtypes<0.5.0` constraint and fails at
import time with:

    AttributeError: module 'ml_dtypes' has no attribute 'float4_e2m1fn'

`onnx==1.16.1` has no `ml_dtypes` dependency at all, so it installs
conflict-free alongside TensorFlow 2.17.0. All other packages installed
at the versions specified/implied by `requirements.txt` with no further
changes needed (tensorflow 2.17.0 has a working `cp311-win_amd64` wheel).

## 2. Engine executable

Built in its own venv. The runtime dependencies in `engine/src/requirements.txt`
must not be installed into the conversion venv above, or vice versa.

    py -3.14 -m venv dev/model_conversion/.venv314
    dev/model_conversion/.venv314/Scripts/python.exe -m pip install -r engine/src/requirements.txt

Then, from the repo root:

    dev/model_conversion/.venv314/Scripts/pyinstaller.exe --noconfirm --onedir --name tempocnn_engine --distpath dev/model_conversion/dist --workpath dev/model_conversion/build --specpath dev/model_conversion engine/src/tempocnn_engine.py

and move the result into place, replacing the previous build:

    rm -rf engine/tempocnn_engine
    mv dev/model_conversion/dist/tempocnn_engine engine/tempocnn_engine

`--onedir`, not `--onefile`. A onefile build unpacks its whole bundle into
`%TEMP%` on every single run, which cost about 4 seconds per analysis — more
than the analysis itself. The same code as a directory bundle starts in about
0.3 s. The trade is a folder instead of one file, and the folder must stay
intact: the executable looks for `_internal/` beside itself.

The model file `deeptemp_k16.onnx` is **not** embedded. It stays in `engine/`,
one level above the executable; `model_path()` searches the executable's own
directory and then its parent, so both this layout and a source run find it.

### After building, verify

    dev/model_conversion/.venvparity/Scripts/python.exe dev/model_conversion/verify_frontend_parity.py
    dev/model_conversion/.venv314/Scripts/python.exe dev/test/test_engine.py

The first asserts the numpy front end still matches librosa sample for sample;
the second asserts the tempos it produces. Both must print `RESULT: OK`.

The parity check is the only place librosa is needed, and it must stay out of
the build venv so it cannot be bundled by accident:

    py -3.11 -m venv dev/model_conversion/.venvparity
    dev/model_conversion/.venvparity/Scripts/python.exe -m pip install "librosa>=0.11.0" soundfile soxr "numpy<2.0" "onnxruntime>=1.18.0"

Finally, update the SHA-256 in `THIRD_PARTY_NOTICES.md`: the executable's hash
changes on every rebuild, including one from unchanged sources.
