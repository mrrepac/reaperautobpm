# Model conversion (dev-only)

One-time step to produce `engine/deeptemp_k16.onnx` from the published
tempo-cnn Keras weights. Not part of the shipped plugin — end users never
run anything in this directory.

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

## Version notes

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
