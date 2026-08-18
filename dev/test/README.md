# Tests (dev-only)

Two checks that run outside REAPER. Nothing here ships with the plugin.

Both need a Lua interpreter, provided by `lupa` — a Python package with Lua
built in, so nothing has to be installed system-wide:

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
mock's `ExecProcess` starts the real PowerShell chain and the real
`tempocnn_engine.exe`, so a full run takes roughly half a minute.

The audio fixture is copied into a directory named in Cyrillic, which is the
case that used to fail with "audio file not found".

Covered:

1. detection end to end, tempo applied;
2. a tempo typed into the confirmation prompt overriding the detected one;
3. the prompt cancelled, leaving the tempo alone;
4. Escape in the progress window, mid-analysis;
5. a launch that silently goes nowhere, caught by the heartbeat after ~20s
   rather than the full three-minute timeout;
6. an exact BPM in the filename bypassing the engine.

Case 4 deliberately leaves an orphaned result file behind — the engine keeps
running after the user walks away — and case 5 asserts the next run sweeps it
up.

`mock_reaper.lua` fakes only what REAPER would provide. Anything Lua cannot do
on its own (a unicode-correct process launch, sleeping, the wall clock) is
handed in from Python, so the mock cannot accidentally paper over the
encoding bug these tests exist to catch.
