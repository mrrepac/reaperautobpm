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
import wave
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


def build_mixed_fixture(path):
    """90 / 112.5 / 90 BPM in three 20 s thirds, spliced from the committed
    fixtures with the standard library so the test needs no numpy."""
    parts = ["click_90bpm", "groove_112_5bpm", "click_90bpm"]
    frames = []
    for name in parts:
        with wave.open(str(REPO / "dev" / "test_audio" / f"{name}.wav"), "rb") as f:
            params = f.getparams()
            frames.append(f.readframes(f.getnframes()))
    with wave.open(str(path), "wb") as out:
        out.setnchannels(params.nchannels)
        out.setsampwidth(params.sampwidth)
        out.setframerate(params.framerate)
        out.writeframes(b"".join(frames))


def prompt_titles(mock):
    """Lua tables come back as objects, and iterating one yields its keys."""
    return [str(value) for value in mock.prompts().values()]


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


def run_case(name, options, clock="real", timeout_ticks=4000, passes=1):
    print(f"\n{name}")
    host = Host(clock)
    lua = LuaRuntime(unpack_returned_tuples=True)
    mock = lua.eval(f"loadfile([[{MOCK}]])")(host)
    mock.configure(lua.table_from({**options, "script_path": str(SCRIPT)}))
    mock.run(str(SCRIPT), timeout_ticks, passes)
    # Anything the script left behind is only visible once the background chain
    # it started has finished writing.
    host.wait_all()
    return mock, host


def wait_for_orphan_result(timeout=90):
    """The detached runner outlives the launcher, so its output arrives late.

    Waited on specifically: the settled result file, not the `.running`
    heartbeat that precedes it. Sweeping while only the heartbeat exists leaves
    the runner to rename it into place afterwards, re-creating a leftover behind
    the sweep -- a race the engine is now fast enough to lose reliably.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        settled = [name for name in engine_leftovers()
                   if name.startswith("bpm_result_") and name.endswith(".txt")]
        if settled:
            return settled
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
orphans = wait_for_orphan_result()
check("cancelled run leaves an orphan", orphans != [], "engine wrote no result")

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

# 6 -- a filename hint that contradicts the engine. The hint no longer short
# circuits the analysis: a tempo written into a name or a tag is frequently
# wrong about the metric level, so both are obtained and the engine wins.
named = WORKDIR / "Трек 128 BPM.wav"
shutil.copyfile(AUDIO, named)
mock, host = run_case("6. Filename hint contradicts the engine",
                      {"audio_path": str(named)})
check("engine still launched", len(host.launched) == 1)
check("engine tempo wins", mock.bpm_set == 120, f"got {mock.bpm_set}")
check("disagreement is spelled out", "≠" in "".join(prompt_titles(mock)),
      "".join(prompt_titles(mock)))

# 7 -- a filename hint the engine agrees with keeps its own precision. 119.75
# is not on the 0.5 grid, and an exact number is not improved by being snapped
# to one.
agreeing = WORKDIR / "Трек 119.75 BPM.wav"
shutil.copyfile(AUDIO, agreeing)
mock, _ = run_case("7. Filename hint the engine agrees with",
                   {"audio_path": str(agreeing)})
check("exact hint applied unrounded", mock.bpm_set == 119.75, f"got {mock.bpm_set}")
check("agreement is spelled out", "≈" in "".join(prompt_titles(mock)),
      "".join(prompt_titles(mock)))

# 8 -- a trimmed item is analysed over its own region only. The file runs
# 90 / 112.5 / 90 BPM in 20 s thirds; the whole file reads as 180, so a result
# of 112.5 can only come from the middle third actually being what was sent.
mixed = WORKDIR / "Смешанный темп.wav"
build_mixed_fixture(mixed)
mock, host = run_case(
    "8. Trimmed item is analysed over its own region",
    {"audio_path": str(mixed), "start_offset": 20.0, "item_length": 20.0,
     "source_length": 60.0},
)
check("engine launched", len(host.launched) == 1)
check("region tempo applied, not the whole file's", mock.bpm_set == 112.5,
      f"got {mock.bpm_set}")

# 9 -- the same item twice in one session: the second run is served from the
# result cache and must not start the engine again.
mock, host = run_case(
    "9. Second run on the same item hits the cache",
    {"audio_path": str(AUDIO)}, passes=2,
)
check("engine launched once for two runs", len(host.launched) == 1,
      f"{len(host.launched)} launches")
check("both runs applied a tempo", mock.bpm_set == 120, f"got {mock.bpm_set}")
check("two prompts shown", len(prompt_titles(mock)) == 2,
      str(prompt_titles(mock)))
check("second prompt says it came from the cache",
      "кэш" in prompt_titles(mock)[1], prompt_titles(mock)[1])
check("no progress window the second time", not mock.window_open())

with wave.open(str(REPO / "dev/test_audio/click_140bpm.wav"), "rb") as audio:
    params = audio.getparams()
    fast = audio.readframes(audio.getnframes())
with wave.open(str(REPO / "dev/test_audio/click_90bpm.wav"), "rb") as audio:
    slow = audio.readframes(audio.getnframes())

wrapped = WORKDIR / "wrapped.wav"
with wave.open(str(wrapped), "wb") as audio:
    audio.setparams(params)
    audio.writeframes(fast + slow * 2)
mock, _ = run_case(
    "10. Looped item crosses the end of the source",
    {"audio_path": str(wrapped), "loop_source": True, "start_offset": 59,
     "item_length": 21, "source_length": 60},
)
check("analysis includes the source head after wrapping", mock.bpm_set == 140,
      f"got {mock.bpm_set}")

silence = WORKDIR / "silence.wav"
with wave.open(str(silence), "wb") as audio:
    audio.setparams(params)
    audio.writeframes(bytes(len(fast)))
mock, _ = run_case(
    "11. Digital silence",
    {"audio_path": str(silence), "item_length": 20, "source_length": 20},
)
check("silence does not offer or apply a tempo",
      not prompt_titles(mock) and mock.bpm_set is None)
check("silence is reported", "тишина" in mock.messages(), mock.messages())

shutil.rmtree(WORKDIR.parent, ignore_errors=True)

print("\n" + "=" * 70)
if failures:
    print(f"{len(failures)} FAILED: {', '.join(failures)}")
    sys.exit(1)
print("all checks passed")
