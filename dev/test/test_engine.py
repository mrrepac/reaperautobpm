"""Accuracy checks for the engine itself, outside REAPER and outside Lua.

test_action.py drives the ReaScript and only cares that a number comes back.
This one cares which number: it is the regression net that makes the front end
safe to touch, and the evidence that the 0.5 BPM step is measured rather than
asserted.

Run:  python dev/test/test_engine.py            (frozen engine if built)
      python dev/test/test_engine.py --source    (engine/src directly)

The frozen engine needs nothing installed. Running from source needs the build
venv (see dev/model_conversion/README.md) -- pass that interpreter as
--python <path> if it is not the one running this script.
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
FIXTURES = REPO / "dev" / "test_audio"
FROZEN = REPO / "engine" / "tempocnn_engine" / "tempocnn_engine.exe"
SOURCE = REPO / "engine" / "src" / "tempocnn_engine.py"

sys.path.insert(0, str(FIXTURES))

# Metric level is deliberately not folded, so an answer of 180 for 90 BPM
# material is correct behaviour. What must hold is that once the level is
# divided out, the 0.5 grid lands on the truth exactly -- which needs the
# remaining error under a quarter BPM.
LEVELS = (0.25, 0.5, 1.0, 2.0, 4.0)
GRID_MARGIN = 0.25
BPM_STEP = 0.5

failures = []


def check(name, condition, detail=""):
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}"
          + (f" -- {detail}" if detail else ""))
    if not condition:
        failures.append(name)


def run_engine(command, *arguments):
    result = subprocess.run(command + [str(a) for a in arguments],
                            capture_output=True, text=True, encoding="utf-8")
    if result.returncode != 0:
        return None, (result.stdout + result.stderr).strip()
    values = {}
    for line in result.stdout.splitlines():
        key, _, value = line.partition(":")
        values[key.strip()] = value.strip()
    return values, ""


def on_grid(bpm, expected):
    """Error after dividing out the metric level, and the grid value it gives."""
    best = min(LEVELS, key=lambda level: abs(bpm / level - expected))
    folded = bpm / best
    snapped = round(folded / BPM_STEP) * BPM_STEP
    return abs(folded - expected), snapped, best


def check_accuracy(command, path, expected, label, *arguments):
    values, error = run_engine(command, path, *arguments)
    if values is None:
        check(f"{label}: engine ran", False, error[:160])
        return None
    bpm = float(values["bpm"])
    deviation, snapped, level = on_grid(bpm, expected)
    check(f"{label}: {expected:g} BPM on the 0.5 grid",
          snapped == expected and deviation < GRID_MARGIN,
          f"got {bpm:.3f} (level x{level:g} -> {snapped:g}, off by {deviation:.3f})")
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", action="store_true",
                        help="run engine/src instead of the frozen build")
    parser.add_argument("--python", default=sys.executable)
    options = parser.parse_args()

    if options.source:
        command = [options.python, str(SOURCE)]
    else:
        if not FROZEN.is_file():
            print(f"frozen engine not found: {FROZEN}\n"
                  f"build it (see dev/model_conversion/README.md) or pass --source")
            raise SystemExit(1)
        command = [str(FROZEN)]
    print(f"engine: {' '.join(command)}")

    print("Committed fixtures")
    for name, expected in (("click_90bpm", 90), ("click_120bpm", 120),
                           ("click_140bpm", 140), ("click_112_5bpm", 112.5),
                           ("click_87_5bpm", 87.5),
                           ("groove_112_5bpm", 112.5)):
        check_accuracy(command, FIXTURES / f"{name}.wav", expected, name)

    print("Loop snapping")
    loop = FIXTURES / "loop_16beats_112_5bpm.wav"
    length = 16 * 60.0 / 112.5
    values, error = run_engine(command, loop, "--snap-length", f"{length:.6f}")
    if values is None:
        check("16 whole beats snap exactly", False, error[:160])
    else:
        check("16 whole beats snap exactly",
              values.get("origin") == "loop" and values.get("loop_beats") == "16"
              and abs(float(values["bpm"]) - 112.5) < 1e-6,
              f"origin={values.get('origin')} beats={values.get('loop_beats')} "
              f"bpm={values.get('bpm')}")

    values, error = run_engine(command, loop, "--snap-length", "7.1")
    check("a length that is not whole bars does not snap",
          values is not None and values.get("origin") != "loop",
          f"origin={values.get('origin') if values else error[:80]}")

    print("Generated material (not committed: too large)")
    from generate_fixtures import make_click_track, make_groove, write_wav

    with tempfile.TemporaryDirectory(prefix="reaperplug_engine_") as directory:
        workdir = Path(directory)

        # Long and genuinely constant: the case a 0.5 step is claimed for.
        for expected in (112.5, 87.5, 128, 174.5):
            path = workdir / ("long_" + f"{expected:g}".replace(".", "_") + ".wav")
            write_wav(path, make_groove(expected, duration_s=120.0))
            check_accuracy(command, path, expected, f"120 s groove at {expected:g}")

        # The point of the region: a beat lifted out of a file whose other
        # parts sit at another tempo. Analysing the whole file must answer
        # something else, and the region must answer the region.
        import numpy as np
        outer, inner = 90, 140
        head = make_click_track(outer, duration_s=30.0)
        middle = make_groove(inner, duration_s=30.0)
        tail = make_click_track(outer, duration_s=30.0)
        mixed = workdir / "mixed_tempo.wav"
        write_wav(mixed, np.concatenate([head, middle, tail]))

        check_accuracy(command, mixed, inner, "region 30-60 s reads its own tempo",
                       "--offset", "30", "--duration", "30")

        whole, error = run_engine(command, mixed)
        if whole is None:
            check("whole file does not read as the region's tempo", False, error[:160])
        else:
            deviation, _, _ = on_grid(float(whole["bpm"]), inner)
            check("whole file does not read as the region's tempo",
                  deviation >= GRID_MARGIN,
                  f"whole file gave {whole['bpm']} (agreement {whole.get('agreement')})")
            check("mixed tempo shows up as window disagreement",
                  float(whole.get("agreement", 1)) < 1.0,
                  f"agreement {whole.get('agreement')}")

    if failures:
        print(f"RESULT: FAIL ({len(failures)})")
        for name in failures:
            print(f"  - {name}")
        raise SystemExit(1)
    print("RESULT: OK")


if __name__ == "__main__":
    main()
