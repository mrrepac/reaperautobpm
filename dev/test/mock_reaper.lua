-- Minimal ReaScript host: enough of reaper.* and gfx.* to run the BPM action
-- outside REAPER. Injected from Python, which supplies the pieces Lua cannot
-- do on its own (unicode-correct process launch, sleeping, wall clock).
local host = ...

local M = { log = {}, deferred = nil, atexit = nil, bpm_set = nil, toolbar = {} }

local function record(kind, value)
  M.log[#M.log + 1] = { kind = kind, value = value }
end

local config = {}
function M.configure(options)
  config = options
end

reaper = {}
gfx = {}

-- Project / items -------------------------------------------------------------

local ITEM, TAKE, SOURCE, PROJECT = "item", "take", "source", "project"

function reaper.EnumProjects() return PROJECT end
function reaper.CountSelectedMediaItems() return 1 end
function reaper.GetSelectedMediaItem() return ITEM end
function reaper.GetActiveTake() return TAKE end
function reaper.TakeIsMIDI() return false end
function reaper.GetMediaItemTake_Source() return SOURCE end
function reaper.GetMediaSourceParent() return nil end
function reaper.GetMediaSourceFileName() return config.audio_path end
function reaper.GetMediaFileMetadata() return 0, "" end

-- Per-parameter, because the script now reads the take's start offset and the
-- source length to work out which region of the file to analyse; a mock that
-- answered the same number to every query would hide a swapped parameter.
function reaper.GetMediaItemTakeInfo_Value(_, parameter)
  if parameter == "D_PLAYRATE" then return config.playrate or 1 end
  if parameter == "D_STARTOFFS" then return config.start_offset or 0 end
  return 0
end

function reaper.GetMediaSourceLength()
  return config.source_length or 0, false
end
function reaper.ValidatePtr2() return true end
function reaper.CountMediaItems() return 1 end
function reaper.GetMediaItem() return ITEM end
function reaper.GetSelectedTrack() return nil end
function reaper.GetMediaItem_Track() return "track" end
function reaper.GetMediaItemInfo_Value(_, parameter)
  if parameter == "D_LENGTH" then return config.item_length or 0 end
  if parameter == "B_LOOPSRC" then return config.loop_source and 1 or 0 end
  return 0
end
function reaper.GetCursorPositionEx() return 0 end

-- Tempo -----------------------------------------------------------------------

function reaper.SetCurrentBPM(_, bpm) M.bpm_set = bpm; record("bpm", bpm) end
function reaper.SetMediaItemInfo_Value() end
function reaper.Undo_BeginBlock2(project) assert(project == PROJECT) end
function reaper.Undo_EndBlock2(project) assert(project == PROJECT) end
function reaper.UpdateArrange() end

-- Script context --------------------------------------------------------------

function reaper.get_action_context()
  return false, config.script_path, 0, 42
end

function reaper.genGuid() return "{TEST-" .. tostring(host.unique()) .. "}" end
function reaper.time_precise() return host.now() end
function reaper.file_exists(path) return host.file_exists(path) end
function reaper.EnumerateFiles(dir, index) return host.enumerate(dir, index) end

-- State -----------------------------------------------------------------------

local ext = {}
function reaper.SetExtState(section, key, value) ext[section .. key] = value end
function reaper.GetExtState(section, key) return ext[section .. key] or "" end
function reaper.DeleteExtState(section, key) ext[section .. key] = nil end

function reaper.atexit(fn) M.atexit = fn end
function reaper.defer(fn) M.deferred = fn end

function reaper.ExecProcess(cmdline, mode)
  record("exec", cmdline)
  if config.swallow_launch then return "" end
  return host.exec(cmdline, mode)
end

-- UI --------------------------------------------------------------------------

function reaper.MB(message) record("message", message) end

function reaper.GetUserInputs(title, _, _, initial)
  record("prompt", title)
  if config.cancel_prompt then return false, "" end
  return true, config.prompt_answer or initial
end

function reaper.SetToggleCommandState(_, _, state)
  M.toolbar[#M.toolbar + 1] = state
end
function reaper.RefreshToolbar2() end
function reaper.my_getViewport() return 0, 0, 1920, 1080 end

local gfx_open = false
function gfx.init() gfx_open = true; record("gfx_init", true) end
function gfx.quit() gfx_open = false; record("gfx_quit", true) end
function gfx.getchar()
  if config.cancel_after and M.ticks >= config.cancel_after then return 27 end
  return 0
end
function gfx.setfont() end
function gfx.set() end
function gfx.rect() end
function gfx.drawstr() end
function gfx.update() end
gfx.w, gfx.h, gfx.x, gfx.y = 460, 128, 0, 0

function M.window_open() return gfx_open end

-- Driver ----------------------------------------------------------------------

M.ticks = 0

function M.run(script_path, timeout_ticks, passes)
  local chunk, err = loadfile(script_path)
  if not chunk then error("load failed: " .. tostring(err)) end

  -- Several passes share one host, which is how the session result cache
  -- becomes observable: pass two must not launch the engine again.
  for _ = 1, (passes or 1) do
    M.atexit, M.deferred = nil, nil
    chunk()

    while M.deferred and M.ticks < (timeout_ticks or 4000) do
      local fn = M.deferred
      M.deferred = nil
      M.ticks = M.ticks + 1
      fn()
      if M.deferred then host.tick() end
    end
    if M.atexit then M.atexit() end
  end
end

function M.prompts()
  local out = {}
  for _, entry in ipairs(M.log) do
    if entry.kind == "prompt" then out[#out + 1] = entry.value end
  end
  return out
end

function M.messages()
  local out = {}
  for _, entry in ipairs(M.log) do
    if entry.kind == "message" then out[#out + 1] = entry.value end
  end
  return table.concat(out, " || ")
end

return M
