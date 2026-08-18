"""Run the BPM ReaScript against a mock REAPER, with the real engine behind it.

Covers what cannot be checked by reading the script: the deferred polling loop,
the progress window's lifecycle, cancellation, and the temp-file bookkeeping —
including a non-ASCII audio path all the way through the real engine.

Usage: python dev/test/test_action.py   (needs lupa; see dev/test/README.md)
"""
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from lupa import LuaRuntime

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
SCRIPT = REPO / "Detect beat BPM and set project tempo.lua"
MOCK = HERE / "mock_reaper.lua"

# A Cyrillic directory and filename: the case that used to break the launch.
WORKDIR = Path(tempfile.mkdtemp(prefix="reaperplug_")) / "Музыка тест"
WORKDIR.mkdir(parents=True)
AUDIO = WORKDIR / "Бит_Мр--Точка.wav"
shutil.copyfile(REPO / "dev" / "test_audio" / "click_120bpm.wav", AUDIO)

failures = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        failures.append(name)


_guid_counter = 0


class Host:
    """The parts of the environment Lua cannot provide for itself."""

    def __init__(self, clock="real"):
        self.clock = clock
        self.virtual = 0.0
        self.origin = time.monotonic()
        self.launched = []
        self.processes = []

    def unique(self):
        # Globally monotonic, like REAPER's genGuid: two runs must never land
        # on the same temp filenames, or their engine processes collide.
        global _guid_counter
        _guid_counter += 1
        return _guid_counter

    def wait_all(self):
        for process in self.processes:
            try:
                process.wait(timeout=120)
            except subprocess.TimeoutExpired:
                process.kill()

    def now(self):
        if self.clock == "virtual":
            return self.virtual
        return time.monotonic() - self.origin

    def tick(self):
        # REAPER runs deferred scripts around 30 Hz.
        if self.clock == "virtual":
            self.virtual += 1 / 30
        else:
            time.sleep(1 / 30)

    def file_exists(self, path):
        return os.path.isfile(path)

    def enumerate(self, directory, index):
        try:
            names = sorted(
                n for n in os.listdir(directory)
                if os.path.isfile(os.path.join(directory, n))
            )
        except OSError:
            return None
        return names[int(index)] if int(index) < len(names) else None

    def exec(self, cmdline, mode):
        # REAPER converts its UTF-8 command line to UTF-16 and calls
        # CreateProcessW; subprocess does the same, so the mock matches it.
        self.launched.append(cmdline)
        self.processes.append(subprocess.Popen(cmdline, shell=False))
        return ""


def run_case(name, options, clock="real", timeout_ticks=4000):
    print(f"\n{name}")
    host = Host(clock)
    lua = LuaRuntime(unpack_returned_tuples=True)
    mock = lua.eval(f"loadfile([[{MOCK}]])")(host)
    mock.configure(lua.table_from({**options, "script_path": str(SCRIPT)}))
    mock.run(str(SCRIPT), timeout_ticks)
    # Anything the script left behind is only visible once the background chain
    # it started has finished writing.
    host.wait_all()
    return mock, host


def wait_for_leftovers(timeout=90):
    """The detached runner outlives the launcher, so its output arrives late."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if engine_leftovers():
            return engine_leftovers()
        time.sleep(0.2)
    return []


def engine_leftovers():
    return sorted(
        p.name for p in (REPO / "engine").iterdir()
        if p.name.startswith(("bpm_job_", "bpm_result_"))
    )


print("=" * 70)
print("BPM action, mock REAPER host, real engine")
print("=" * 70)

# 1 -- happy path, Cyrillic audio path, real engine round trip.
mock, host = run_case(
    "1. Detection end to end (Cyrillic path)",
    {"audio_path": str(AUDIO)},
)
check("engine was launched", len(host.launched) == 1)
check("tempo applied", mock.bpm_set == 120, f"got {mock.bpm_set}")
check("progress window opened and closed", not mock.window_open())
check("toolbar lit then cleared", list(mock.toolbar.values()) == [1, 0],
      str(list(mock.toolbar.values())))
check("no error dialog", mock.messages() == "", mock.messages())
check("no leftover temp files", engine_leftovers() == [], str(engine_leftovers()))

# 2 -- user answers the confirmation prompt with a different tempo.
mock, _ = run_case(
    "2. Manual override in the prompt",
    {"audio_path": str(AUDIO), "prompt_answer": "70"},
)
check("typed tempo wins", mock.bpm_set == 70, f"got {mock.bpm_set}")

# 3 -- user cancels the confirmation prompt.
mock, _ = run_case(
    "3. Prompt cancelled",
    {"audio_path": str(AUDIO), "cancel_prompt": True},
)
check("tempo untouched", mock.bpm_set is None, f"got {mock.bpm_set}")
check("no leftover temp files", engine_leftovers() == [], str(engine_leftovers()))

# 4 -- Escape in the progress window, mid-analysis.
mock, _ = run_case(
    "4. Cancelled from the progress window",
    {"audio_path": str(AUDIO), "cancel_after": 3},
)
check("tempo untouched", mock.bpm_set is None, f"got {mock.bpm_set}")
check("window closed", not mock.window_open())
check("toolbar cleared", list(mock.toolbar.values()) == [1, 0],
      str(list(mock.toolbar.values())))
check("cancel is silent", mock.messages() == "", mock.messages())
# The engine was left running, so its result lands after nobody is waiting.
# That orphan is expected; the next run is what sweeps it up.
orphans = wait_for_leftovers()
check("cancelled run leaves an orphan", orphans != [], "engine wrote nothing")

# 5 -- launch silently goes nowhere: must fail fast, not sit out the 3 minutes.
mock, _ = run_case(
    "5. Engine never starts",
    {"audio_path": str(AUDIO), "swallow_launch": True},
    clock="virtual",
)
check("reported as a failed launch", "не запустился" in mock.messages(), mock.messages())
check("gave up near the 20s mark", 20 <= mock.ticks / 30 < 30, f"{mock.ticks / 30:.0f}s")
check("tempo untouched", mock.bpm_set is None)
check("own job file removed, and case 4's orphan swept up",
      engine_leftovers() == [], str(engine_leftovers()))

# 6 -- exact BPM in the filename skips the engine entirely.
named = WORKDIR / "Трек 128 BPM.wav"
shutil.copyfile(AUDIO, named)
mock, host = run_case("6. Filename hint bypasses the engine", {"audio_path": str(named)})
check("engine not launched", len(host.launched) == 0)
check("hinted tempo applied", mock.bpm_set == 128, f"got {mock.bpm_set}")
check("no progress window", not mock.window_open())

shutil.rmtree(WORKDIR.parent, ignore_errors=True)

print("\n" + "=" * 70)
if failures:
    print(f"{len(failures)} FAILED: {', '.join(failures)}")
    sys.exit(1)
print("all checks passed")
