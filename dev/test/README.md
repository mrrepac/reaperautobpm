# Tests (dev-only)

Three checks that run outside REAPER. Nothing here ships with the plugin.

The first two need a Lua interpreter, provided by `lupa` — a Python package
with Lua built in, so nothing has to be installed system-wide:

    python -m pip install -r dev/test/requirements.txt

## Syntax check

    python dev/test/check_syntax.py "Detect beat BPM and set project tempo.lua"

Compiles the script without running it and lists every `reaper.*` / `gfx.*`
name it touches, so a typo in an API name shows up here rather than at the
first user click.

## Behaviour check

    python dev/test/test_action.py

Runs the action against `mock_reaper.lua`, a stand-in host implementing just
enough of `reaper.*` and `gfx.*` to drive it. The engine is **not** mocked: the
mock's `ExecProcess` starts the real PowerShell chain and the real engine, so a
full run takes roughly half a minute.

The audio fixture is copied into a directory named in Cyrillic, which is the
case that used to fail with "audio file not found".

Covered:

1. detection end to end, tempo applied;
2. a tempo typed into the confirmation prompt overriding the detected one;
3. the prompt cancelled, leaving the tempo alone;
4. Escape in the progress window, mid-analysis;
5. a launch that silently goes nowhere, caught by the heartbeat after ~20s
   rather than the full three-minute timeout;
6. a filename hint that contradicts the engine — the engine wins and the
   disagreement is named in the prompt;
7. a filename hint the engine agrees with, applied at its own precision
   (119.75, which the 0.5 grid would otherwise destroy);
8. a trimmed item analysed over its own region: the fixture runs 90 / 112.5 /
   90 BPM in 20 s thirds and reads as 180 whole, so a result of 112.5 proves
   the middle third is what actually reached the engine;
9. the same item run twice in one session, served from the result cache the
   second time without starting the engine again.

Case 4 deliberately leaves an orphaned result file behind — the engine keeps
running after the user walks away — and case 5 asserts the next run sweeps it
up.

`mock_reaper.lua` fakes only what REAPER would provide. Anything Lua cannot do
on its own (a unicode-correct process launch, sleeping, the wall clock) is
handed in from Python, so the mock cannot accidentally paper over the
encoding bug these tests exist to catch. It answers `GetMediaItemTakeInfo_Value`
and `GetMediaItemInfo_Value` per parameter rather than with one number, so a
swapped parameter name cannot pass unnoticed.

## Engine accuracy check

    python dev/test/test_engine.py              # the frozen engine
    python dev/test/test_engine.py --source     # engine/src, needs the build venv

Where the behaviour check only wants a number back, this one cares which
number. It is the regression net that makes the front end safe to touch, and
the evidence behind the 0.5 BPM step.

Metric level is deliberately not folded, so 180 for 90 BPM material is correct;
what the test asserts is that once the level is divided out, the 0.5 grid lands
on the truth exactly — which needs the remaining error under a quarter BPM.

Covered:

* every committed fixture, including the half-BPM ones (112.5, 87.5);
* loop snapping: 16 whole beats land on exactly 112.500, and a length that is
  not whole bars does not snap at all;
* 120-second constant-tempo grooves at 112.5, 87.5, 128 and 174.5 — all exact;
* a file running 90 / 140 / 90 BPM in 30 s thirds: the middle region must read
  as its own tempo, the whole file must not, and the disagreement must show up
  in the reported window agreement.

The longer material is generated into a temp directory by
`dev/test_audio/generate_fixtures.py` rather than committed, because it is
megabytes per file. The frozen engine needs nothing installed; `--source`
needs the build venv from `dev/model_conversion/README.md`, passed as
`--python <path>` if it is not the interpreter running the test.
