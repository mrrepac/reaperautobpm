"""Compile-check a ReaScript without running it."""
import sys
from lupa import LuaRuntime

path = sys.argv[1]
source = open(path, encoding="utf-8").read()

lua = LuaRuntime(unpack_returned_tuples=True)
check = lua.eval(
    "function(src, name)"
    "  local chunk, err = load(src, name)"
    "  return {ok = chunk ~= nil, err = err}"
    "end"
)
result = check(source, path)
if not result["ok"]:
    print(f"SYNTAX ERROR: {result['err']}")
    sys.exit(1)
print(f"syntax OK ({len(source.splitlines())} lines)")

# Every reaper.* / gfx.* name the script touches, so a typo in an API name
# shows up here rather than at the first user click.
import re
used = sorted(set(re.findall(r"\b(?:reaper|gfx)\.\w+", source)))
print("API symbols used:", ", ".join(used))
