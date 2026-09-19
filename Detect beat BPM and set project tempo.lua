-- @description Detect beat BPM with TempoCNN and set project tempo
-- @version 3.0.0
-- @author reaperplug
-- @about
--   Uses TempoCNN (deeptemp_k16).
--   Select an audio item (or park the edit cursor on one), then run the script.
--   Only the item's own audio region is analysed, and the detected BPM is shown
--   for confirmation before the project tempo changes.

local ANALYSIS_TIMEOUT_MS = 180000

local MIN_ACCEPTED_BPM = 20
local MAX_ACCEPTED_BPM = 960

-- Tempo grid. Half-BPM tempos are real material -- 112.5 and 87.5 both turned
-- up in practice -- and the engine's fine refinement resolves them to about
-- 0.05 BPM, so a 0.5 step is measured rather than decorative.
local BPM_STEP = 0.5
-- A value the engine derived exactly (whole bars in the item, or a number read
-- out of the file) is not improved by the grid, so it is only snapped when it
-- lands on one anyway. Otherwise an exact 117.333 would become a wrong 117.5.
local EXACT_SNAP_TOLERANCE = 0.06

local ENGINE_FILENAME = "tempocnn_engine" .. package.config:sub(1, 1) .. "tempocnn_engine.exe"
local LAUNCHER_FILENAME = "launch_engine.ps1"
local RUNNER_FILENAME = "run_engine.ps1"
local MODEL_FILENAME = "deeptemp_k16.onnx"
local RESULT_PREFIX = "bpm_result_"
local JOB_PREFIX = "bpm_job_"

-- How long the runner may take to announce itself before the launch counts as
-- failed. Starting PowerShell twice is slow but nowhere near this.
local LAUNCH_CONFIRM_MS = 20000

local EXT_SECTION = "reaperplug_engine_bpm"
local EXT_RUNNING_KEY = "analysis_started_at"
local EXT_CACHE_KEY = "analysis_cache"

-- Results are kept for the session only. Keyed on path, size and region, which
-- cannot notice a file rewritten to the same byte count -- a session-long
-- window makes that unreachable in practice, and re-running on the same item
-- (to compare half against double, say) is the case worth serving.
local CACHE_LIMIT = 48

-- Below this share of windows agreeing, the file does not have one tempo, and
-- a single project tempo is the wrong answer for it.
local AGREEMENT_FLOOR = 0.6
-- A runner-up metric level is worth naming only if the network gave it weight.
local CANDIDATE_FLOOR = 0.05

local SEPARATOR = package.config:sub(1, 1)

local function fail(message)
  reaper.MB(message, "TempoCNN BPM Detector", 0)
  return nil
end

-- Progress window -------------------------------------------------------------

local PROGRESS_WIDTH = 460
local PROGRESS_HEIGHT = 128

local progress = { visible = false, caption = "" }

local function utf8_tail(text, max_bytes)
  if #text <= max_bytes then return text end
  local start = #text - max_bytes + 1
  -- Never start mid-character: continuation bytes are 0x80..0xBF.
  while start <= #text do
    local byte = text:byte(start)
    if byte < 0x80 or byte >= 0xC0 then break end
    start = start + 1
  end
  return "…" .. text:sub(start)
end

function progress.show(caption)
  -- Four returns, not five: the viewport rect is written back into the first
  -- four arguments and the function itself returns nothing.
  local left, top, right, bottom = reaper.my_getViewport(
    0, 0, 0, 0, 0, 0, 0, 0, true)
  local x, y = 120, 120
  if left and right > left and bottom > top then
    x = math.floor((left + right - PROGRESS_WIDTH) / 2)
    y = math.floor((top + bottom - PROGRESS_HEIGHT) / 2)
  end

  progress.caption = utf8_tail(caption, 52)
  gfx.init("Detect BPM", PROGRESS_WIDTH, PROGRESS_HEIGHT, 0, x, y)
  gfx.setfont(1, "Segoe UI", 17)
  gfx.setfont(2, "Segoe UI", 13)
  progress.visible = true
end

function progress.close()
  if not progress.visible then return end
  gfx.quit()
  progress.visible = false
end

-- Returns false once the user asks to stop, by Escape or by closing the window.
function progress.update(elapsed, stage)
  if not progress.visible then return true end

  local char = gfx.getchar()
  if char < 0 or char == 27 then return false end

  gfx.set(0.11, 0.11, 0.13, 1)
  gfx.rect(0, 0, gfx.w, gfx.h, 1)

  gfx.setfont(1)
  gfx.set(0.93, 0.93, 0.95, 1)
  gfx.x, gfx.y = 20, 22
  gfx.drawstr(progress.caption)

  -- The engine reports no intermediate progress, so the bar sweeps instead of
  -- filling: an honest "still working", not a fabricated percentage.
  local bar_x, bar_y = 20, 66
  local bar_w, bar_h = gfx.w - 40, 8
  gfx.set(0.20, 0.20, 0.23, 1)
  gfx.rect(bar_x, bar_y, bar_w, bar_h, 1)

  local knob_w = bar_w * 0.28
  local span = bar_w + knob_w
  local offset = (elapsed * 0.6) % 1.0 * span - knob_w
  local from = math.max(bar_x, bar_x + offset)
  local to = math.min(bar_x + bar_w, bar_x + offset + knob_w)
  if to > from then
    gfx.set(0.36, 0.62, 0.93, 1)
    gfx.rect(from, bar_y, to - from, bar_h, 1)
  end

  gfx.setfont(2)
  gfx.set(0.60, 0.60, 0.65, 1)
  gfx.x, gfx.y = 20, 92
  gfx.drawstr(string.format("%s · %d с · Esc — отмена", stage, math.floor(elapsed)))

  gfx.update()
  return true
end

-- Toolbar ---------------------------------------------------------------------

local toolbar = {}

function toolbar.set(state)
  local _, _, section, command = reaper.get_action_context()
  if not section or not command or command <= 0 then return end
  reaper.SetToggleCommandState(section, command, state)
  reaper.RefreshToolbar2(section, command)
end

-- Tempo values ----------------------------------------------------------------

-- The detected tempo is snapped to the 0.5 grid; a value the engine derived
-- exactly keeps its own precision unless the grid happens to fit it.
local function quantize_bpm(bpm, exact)
  local snapped = math.floor(bpm / BPM_STEP + 0.5) * BPM_STEP
  if exact and math.abs(snapped - bpm) > EXACT_SNAP_TOLERANCE then
    return math.floor(bpm * 1000 + 0.5) / 1000
  end
  return snapped
end

-- A typed tempo is an assertion, not an estimate, so it is taken as written --
-- including the 56.25 that halving a 112.5 track actually needs.
local function round_typed_bpm(bpm)
  return math.floor(bpm * 1000 + 0.5) / 1000
end

local function format_bpm(bpm)
  local text = string.format("%.3f", bpm)
  text = text:gsub("0+$", ""):gsub("%.$", "")
  return text
end

-- Target item -----------------------------------------------------------------

local function audio_take(item)
  local take = reaper.GetActiveTake(item)
  if not take or reaper.TakeIsMIDI(take) then return nil end
  return take
end

local function item_covers(item, position)
  local start = reaper.GetMediaItemInfo_Value(item, "D_POSITION")
  local length = reaper.GetMediaItemInfo_Value(item, "D_LENGTH")
  return position >= start and position <= start + length
end

-- Selection first, then the edit cursor, then a project that holds a single
-- audio item. The mouse is deliberately ignored: when the action runs from a
-- toolbar button the pointer sits on that button, nowhere near the beat.
local function resolve_target(project)
  local selected_count = reaper.CountSelectedMediaItems(project)
  if selected_count > 1 then
    return nil, nil,
      "Выделено несколько items. Оставьте один — какой из них бит, скрипт не угадывает."
  end
  if selected_count == 1 then
    local item = reaper.GetSelectedMediaItem(project, 0)
    local take = audio_take(item)
    if not take then
      return nil, nil, "Выделенный item должен содержать аудио."
    end
    return item, take
  end

  local cursor = reaper.GetCursorPositionEx(project)
  local selected_track = reaper.GetSelectedTrack(project, 0)
  local under_cursor = {}
  local only_item, only_take, audio_count = nil, nil, 0

  for index = 0, reaper.CountMediaItems(project) - 1 do
    local item = reaper.GetMediaItem(project, index)
    local take = audio_take(item)
    if take then
      audio_count = audio_count + 1
      only_item, only_take = item, take
      if item_covers(item, cursor) then
        under_cursor[#under_cursor + 1] = { item = item, take = take }
      end
    end
  end

  if #under_cursor > 0 then
    -- Several items can sit under the cursor on stacked tracks. A selected
    -- track breaks the tie; otherwise the topmost track wins.
    local chosen = under_cursor[1]
    if selected_track then
      for _, candidate in ipairs(under_cursor) do
        if reaper.GetMediaItem_Track(candidate.item) == selected_track then
          chosen = candidate
          break
        end
      end
    end
    return chosen.item, chosen.take
  end

  if audio_count == 1 then
    return only_item, only_take
  end

  return nil, nil,
    "Нечего анализировать. Выделите audio item с битом или поставьте на него edit-курсор."
end

-- Returns the file-backed source, plus whether the take reached it through a
-- wrapper (a section or a reversed take). A wrapper puts the take's own offsets
-- in different coordinates than the file's, so the region is not computed then.
local function root_source(take)
  local source = reaper.GetMediaItemTake_Source(take)
  local nested = false
  while source do
    local parent = reaper.GetMediaSourceParent(source)
    if not parent then break end
    nested = true
    source = parent
  end
  return source, nested
end

local function take_playrate(take)
  local playrate = reaper.GetMediaItemTakeInfo_Value(take, "D_PLAYRATE")
  if not playrate or playrate <= 0 then return 1 end
  return playrate
end

-- Analysis region -------------------------------------------------------------

-- Everything here is in source time: the offset already is, and the item's
-- length is multiplied by the play rate to get there.
--
-- Without this the engine was handed the whole file no matter how the item was
-- trimmed, so eight bars lifted out of an hour-long set were answered with the
-- average tempo of the set.
local function analysis_region(item, take, source, nested)
  if nested then return {} end

  local playrate = take_playrate(take)
  local offset = reaper.GetMediaItemTakeInfo_Value(take, "D_STARTOFFS") or 0
  local length = (reaper.GetMediaItemInfo_Value(item, "D_LENGTH") or 0) * playrate
  if offset < 0 then offset = 0 end
  if length <= 0 then return {} end

  local source_length = reaper.GetMediaSourceLength(source)
  if source_length and source_length > 0 then
    if offset >= source_length then return {} end
    -- A looped item runs past its source; one pass through it is the material.
    if offset + length > source_length then
      length = source_length - offset
    end
    -- Whole file anyway: leave the region out so the engine reads it in one go.
    if offset <= 0.001 and length >= source_length - 0.001 then
      return { snap_length = length }
    end
  end

  return { offset = offset, duration = length, snap_length = length }
end

-- Exact hints -----------------------------------------------------------------

local function parse_bpm(text)
  if not text then return nil end
  text = text:gsub(",", ".")
  local value = tonumber(text:match("%d+%.?%d*"))
  if value and value >= MIN_ACCEPTED_BPM and value <= 400 then return value end
  return nil
end

local function exact_bpm_hint(source)
  if reaper.GetMediaFileMetadata then
    local ok, identifiers = reaper.GetMediaFileMetadata(source, "")
    if ok and ok ~= 0 and identifiers then
      for identifier in identifiers:gmatch("[^\r\n]+") do
        local upper = identifier:upper()
        if upper:find("BPM", 1, true) or upper:find("TEMPO", 1, true)
          or upper:find("TBPM", 1, true) then
          local value_ok, value = reaper.GetMediaFileMetadata(source, identifier)
          local bpm = value_ok and value_ok ~= 0 and parse_bpm(value) or nil
          if bpm then return bpm, "metadata " .. identifier end
        end
      end
    end
  end

  local filename = reaper.GetMediaSourceFileName(source) or ""
  local value = filename:match("(%d+%.?%d*)%s*[Bb][Pp][Mm]")
    or filename:match("[Bb][Pp][Mm]%s*[_%- ]*(%d+%.?%d*)")
  local bpm = parse_bpm(value)
  if bpm then return bpm, "filename" end
  return nil
end

-- Engine ----------------------------------------------------------------------

local function script_directory()
  local _, script_path = reaper.get_action_context()
  return script_path and script_path:match("^(.*)[/\\]") or nil
end

local function quote_argument(value)
  return '"' .. value:gsub('"', '\\"') .. '"'
end

local function engine_paths()
  local directory = script_directory()
  if not directory then
    return nil, "Не удалось определить папку ReaScript."
  end
  local engine_directory = directory .. SEPARATOR .. "engine"
  local paths = {
    directory = engine_directory,
    engine = engine_directory .. SEPARATOR .. ENGINE_FILENAME,
    runner = engine_directory .. SEPARATOR .. RUNNER_FILENAME,
    launcher = engine_directory .. SEPARATOR .. LAUNCHER_FILENAME,
    model = engine_directory .. SEPARATOR .. MODEL_FILENAME,
  }
  for _, key in ipairs({ "engine", "runner", "launcher", "model" }) do
    if not reaper.file_exists(paths[key]) then
      return nil, "Не найден файл движка:\n" .. paths[key]
    end
  end
  return paths
end

-- Orphans appear whenever the polling loop dies before reading its result:
-- REAPER closing mid-analysis, or Escape, which stops every deferred script.
local function remove_stale_results(engine_directory)
  local stale = {}
  local index = 0
  while true do
    local name = reaper.EnumerateFiles(engine_directory, index)
    if not name then break end
    if name:sub(1, #RESULT_PREFIX) == RESULT_PREFIX
      or name:sub(1, #JOB_PREFIX) == JOB_PREFIX then
      stale[#stale + 1] = name
    end
    index = index + 1
  end
  -- Collected first: deleting during enumeration would shift the indices.
  for _, name in ipairs(stale) do
    os.remove(engine_directory .. SEPARATOR .. name)
  end
end

local function parse_candidates(text)
  local candidates = {}
  if not text then return candidates end
  for bpm, mass in text:gmatch("([%d%.]+):([%d%.]+)") do
    candidates[#candidates + 1] = { bpm = tonumber(bpm), mass = tonumber(mass) }
  end
  return candidates
end

local function parse_engine_output(output)
  local engine_exit = output and tonumber(output:match("__ENGINE_EXIT__=(%-?%d+)")) or nil
  if engine_exit == nil then
    return nil, "Результат анализа повреждён или не завершён."
  end
  if engine_exit ~= 0 then
    return nil, "Движок завершился с кодом " .. tostring(engine_exit) .. ":\n" .. output
  end
  local bpm = tonumber(output:match("bpm:%s*([%d%.]+)"))
  if not bpm or bpm <= 0 then
    return nil, "Движок не смог определить BPM:\n" .. (output or "")
  end
  return {
    bpm = bpm,
    bpm_cnn = tonumber(output:match("bpm_cnn:%s*([%d%.]+)")) or bpm,
    confidence = tonumber(output:match("confidence:%s*([%d%.%-]+)")) or 0,
    origin = output:match("origin:%s*(%a+)") or "cnn",
    agreement = tonumber(output:match("agreement:%s*([%d%.]+)")) or 1,
    loop_beats = tonumber(output:match("loop_beats:%s*(%d+)")),
    candidates = output:match("candidates:%s*([^\r\n]*)") or "",
  }
end

local function read_text_file(path)
  local file = io.open(path, "rb")
  if not file then return nil end
  local content = file:read("*a")
  file:close()
  return content
end

-- REAPER hands out paths as UTF-8. Written back byte for byte, so the runner
-- can read the job file as UTF-8 and recover the original name exactly.
local function write_text_file(path, content)
  local file = io.open(path, "wb")
  if not file then return false end
  file:write(content)
  file:close()
  return true
end

-- os.execute routes through the ANSI C runtime, which reinterprets REAPER's
-- UTF-8 bytes through the system codepage and destroys every non-Latin path.
-- ExecProcess is REAPER's own launcher and converts to UTF-16 properly.
local function start_detached(command)
  if reaper.ExecProcess then
    -- Detached mode returns nothing useful, so success is not judged here:
    -- the runner's heartbeat file is what confirms the launch took.
    reaper.ExecProcess(command, -2)
    return true
  end

  local execute_result, execute_reason, execute_code = os.execute(command)
  if execute_result == true or execute_result == 0 then return true end
  return false, "Не удалось запустить движок в фоне. Код: "
    .. tostring(execute_code or execute_result)
    .. "; причина: " .. tostring(execute_reason or "неизвестна")
end

-- Result cache ----------------------------------------------------------------

local function file_size(path)
  local file = io.open(path, "rb")
  if not file then return 0 end
  local size = file:seek("end") or 0
  file:close()
  return size
end

local function cache_signature(path, region)
  return string.format("%s|%d|%.3f|%.3f", path, file_size(path),
    region.offset or 0, region.duration or 0)
end

local function cache_records()
  local records = {}
  for record in reaper.GetExtState(EXT_SECTION, EXT_CACHE_KEY):gmatch("[^\n]+") do
    records[#records + 1] = record
  end
  return records
end

local function cache_lookup(signature)
  for _, record in ipairs(cache_records()) do
    local fields = {}
    for field in (record .. "\t"):gmatch("([^\t]*)\t") do
      fields[#fields + 1] = field
    end
    if fields[1] == signature then
      return {
        bpm = tonumber(fields[2]),
        bpm_cnn = tonumber(fields[3]) or tonumber(fields[2]),
        confidence = tonumber(fields[4]) or 0,
        origin = fields[5] or "cnn",
        agreement = tonumber(fields[6]) or 1,
        loop_beats = tonumber(fields[7]),
        candidates = fields[8] or "",
        cached = true,
      }
    end
  end
  return nil
end

local function cache_store(signature, result)
  local record = table.concat({
    signature,
    string.format("%.4f", result.bpm),
    string.format("%.4f", result.bpm_cnn or result.bpm),
    string.format("%.4f", result.confidence or 0),
    result.origin or "cnn",
    string.format("%.4f", result.agreement or 1),
    result.loop_beats and tostring(result.loop_beats) or "",
    result.candidates or "",
  }, "\t")

  local kept = { record }
  for _, existing in ipairs(cache_records()) do
    if #kept >= CACHE_LIMIT then break end
    if existing:sub(1, #signature + 1) ~= signature .. "\t" then
      kept[#kept + 1] = existing
    end
  end
  reaper.SetExtState(EXT_SECTION, EXT_CACHE_KEY, table.concat(kept, "\n"), false)
end

-- Analysis state --------------------------------------------------------------

-- Only the run that claimed the flag may clear it, so a second invocation
-- bouncing off "analysis already running" cannot release the first one's claim.
local owned = { analysis = false, job = nil, result = nil }

local function set_analysis_running(running)
  if running then
    owned.analysis = true
    reaper.SetExtState(EXT_SECTION, EXT_RUNNING_KEY, tostring(os.time()), false)
  else
    owned.analysis = false
    reaper.DeleteExtState(EXT_SECTION, EXT_RUNNING_KEY, false)
  end
end

-- Escape tears deferred scripts down without unwinding, and closing REAPER
-- mid-analysis does the same. This is the only cleanup that always runs.
reaper.atexit(function()
  progress.close()
  -- Everything below belongs to a run this instance owns. A second invocation
  -- that bounced off the busy check must not clear the first one's toolbar
  -- light or delete its files.
  if not owned.analysis then return end
  toolbar.set(0)
  if owned.job then os.remove(owned.job) end
  if owned.result then
    os.remove(owned.result)
    os.remove(owned.result .. ".running")
  end
  reaper.DeleteExtState(EXT_SECTION, EXT_RUNNING_KEY, false)
end)

local function analysis_in_progress()
  local started = tonumber(reaper.GetExtState(EXT_SECTION, EXT_RUNNING_KEY))
  if not started then return false end
  -- Escape kills deferred scripts without unwinding, so the flag can outlive
  -- the run it describes. Anything older than the timeout is a leftover.
  if os.time() - started > ANALYSIS_TIMEOUT_MS / 1000 + 30 then
    set_analysis_running(false)
    return false
  end
  return true
end

local function job_contents(audio_path, region)
  local lines = { "path=" .. audio_path }
  if region.offset and region.offset > 0 then
    lines[#lines + 1] = string.format("offset=%.6f", region.offset)
  end
  if region.duration and region.duration > 0 then
    lines[#lines + 1] = string.format("duration=%.6f", region.duration)
  end
  if region.snap_length and region.snap_length > 0 then
    lines[#lines + 1] = string.format("snap_length=%.6f", region.snap_length)
  end
  return table.concat(lines, "\n") .. "\n"
end

local function async_engine(audio_path, region, on_complete)
  local paths, paths_error = engine_paths()
  if not paths then
    fail(paths_error)
    return false
  end

  remove_stale_results(paths.directory)

  local unique = reaper.genGuid():gsub("[^%w]", "")
  local result_path = paths.directory .. SEPARATOR .. RESULT_PREFIX .. unique .. ".txt"
  local job_path = paths.directory .. SEPARATOR .. JOB_PREFIX .. unique .. ".txt"

  -- The audio path never travels on a command line: it is the one path that
  -- routinely lives outside the ASCII range, and every shell hop is a chance
  -- to mangle it. The engine folder sits next to the script and is passed as
  -- arguments; ExecProcess keeps those intact.
  if not write_text_file(job_path, job_contents(audio_path, region)) then
    fail("Не удалось записать задание для движка:\n" .. job_path)
    return false
  end

  local windows_directory = os.getenv("WINDIR") or "C:\\Windows"
  local powershell_path = windows_directory
    .. "\\System32\\WindowsPowerShell\\v1.0\\powershell.exe"
  if not reaper.file_exists(powershell_path) then
    os.remove(job_path)
    fail("Не найден системный Windows PowerShell:\n" .. powershell_path)
    return false
  end

  -- The system path contains no spaces. Leaving the executable unquoted avoids
  -- the special first-quote parsing of the cmd.exe hop both launchers go through.
  local launch_command = powershell_path
    .. " -NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File "
    .. quote_argument(paths.launcher) .. " "
    .. quote_argument(paths.runner) .. " "
    .. quote_argument(paths.engine) .. " "
    .. quote_argument(job_path) .. " "
    .. quote_argument(result_path)

  local launch_ok, launch_error = start_detached(launch_command)
  if not launch_ok then
    os.remove(job_path)
    fail(launch_error)
    return false
  end

  set_analysis_running(true)
  owned.job, owned.result = job_path, result_path
  local started = reaper.time_precise()
  -- The runner touches this the moment it starts, which turns a launch that
  -- silently went nowhere into a fast error instead of a three-minute wait.
  local heartbeat_path = result_path .. ".running"
  local running = false

  toolbar.set(1)
  progress.show(audio_path:match("[^/\\]+$") or "Анализ BPM")

  local function finish()
    os.remove(job_path)
    set_analysis_running(false)
    progress.close()
    toolbar.set(0)
  end

  local function poll()
    if reaper.file_exists(result_path) then
      local output = read_text_file(result_path)
      os.remove(result_path)
      finish()
      on_complete(parse_engine_output(output))
      return
    end

    local elapsed = reaper.time_precise() - started
    -- Checked until it appears, then left alone: the runner deletes it by
    -- renaming it onto the result path.
    if not running and reaper.file_exists(heartbeat_path) then running = true end

    if not progress.update(elapsed, running and "Анализ" or "Запуск движка") then
      -- Cancelled by the user. The engine process is left to finish and die on
      -- its own; its leftovers are swept up by the next run.
      finish()
      return
    end

    local elapsed_ms = elapsed * 1000

    if elapsed_ms > LAUNCH_CONFIRM_MS and not running then
      finish()
      fail("Движок не запустился. Проверьте, что PowerShell не заблокирован политикой"
        .. " безопасности, и что файлы в папке engine на месте.")
      return
    end

    if elapsed_ms > ANALYSIS_TIMEOUT_MS then
      finish()
      fail("Движок не завершил анализ за отведённое время.")
      return
    end

    reaper.defer(poll)
  end

  reaper.defer(poll)
  return true
end

-- Tempo -----------------------------------------------------------------------

local function apply_result(project, item, bpm)
  reaper.Undo_BeginBlock()
  -- Project tempo must change without stretching the reference beat.
  reaper.SetMediaItemInfo_Value(item, "C_BEATATTACHMODE", 0)
  reaper.SetCurrentBPM(project, bpm, false)
  reaper.Undo_EndBlock("Detect beat BPM with TempoCNN and set project tempo", -1)
  reaper.UpdateArrange()
  return bpm
end

-- Presentation ----------------------------------------------------------------

-- Merges what the engine heard with what the file claims, and words the
-- result. The returned bpm is what the prompt offers; everything else is the
-- caption that says where it came from and what else it could be.
local function describe(result, hint_bpm, hint_origin, playrate)
  local engine_bpm = result.bpm * playrate
  local exact = result.origin == "loop"
  -- Whole bars in the item pin the tempo exactly, so that is worth saying in
  -- place of a confidence the exact answer did not come from.
  local engine_note = result.loop_beats
    and string.format("loop %d долей", result.loop_beats)
    or string.format("conf %.2f", result.confidence)
  local notes = {}
  local chosen

  if hint_bpm then
    local hinted = hint_bpm * playrate
    -- A number written into the file is exact where the engine only estimates,
    -- so when the two agree the written one is the better answer -- and it is
    -- the one that can carry a .5. When they disagree the engine wins: a tag
    -- copied by DJ software is most often wrong about the metric level, which
    -- is exactly what half/double in the caption is for.
    if math.abs(hinted - engine_bpm) <= engine_bpm * 0.02 then
      chosen, exact = hinted, true
      notes[#notes + 1] = hint_origin .. " ≈ " .. format_bpm(engine_bpm)
    else
      chosen = engine_bpm
      notes[#notes + 1] = engine_note
      notes[#notes + 1] = hint_origin .. " " .. format_bpm(hinted) .. " ≠"
    end
  else
    chosen = engine_bpm
    notes[#notes + 1] = engine_note
  end

  local bpm = quantize_bpm(chosen, exact)

  -- The network's own runner-up, when it has one, says more about the metric
  -- level than arithmetic does.
  for _, candidate in ipairs(parse_candidates(result.candidates)) do
    local candidate_bpm = candidate.bpm * playrate
    if candidate.mass >= CANDIDATE_FLOOR
      and math.abs(candidate_bpm - bpm) > bpm * 0.04 then
      notes[#notes + 1] = string.format("ещё %s (%.2f)",
        format_bpm(quantize_bpm(candidate_bpm, false)), candidate.mass)
      break
    end
  end

  if (result.agreement or 1) < AGREEMENT_FLOOR then
    notes[#notes + 1] = "темп непостоянный"
  end
  if result.cached then
    notes[#notes + 1] = "кэш"
  end

  return bpm, string.format("%s BPM · half %s / double %s · %s",
    format_bpm(bpm), format_bpm(bpm / 2), format_bpm(bpm * 2),
    table.concat(notes, ", "))
end

-- Confirmation ----------------------------------------------------------------

-- Returns a choice table, or nil plus an error message. Both nil means cancel.
local function confirm(bpm, title)
  local ok, value = reaper.GetUserInputs(title, 1, "BPM:,extrawidth=100",
    format_bpm(bpm))
  if not ok then return nil end

  local typed = tonumber(((value or ""):gsub(",", ".")):match("^%s*(.-)%s*$"))
  if not typed or typed < MIN_ACCEPTED_BPM or typed > MAX_ACCEPTED_BPM then
    return nil, string.format(
      "Введите BPM числом от %d до %d.", MIN_ACCEPTED_BPM, MAX_ACCEPTED_BPM
    )
  end
  return { bpm = round_typed_bpm(typed) }
end

local function confirm_and_apply(project, item, bpm, title)
  if not reaper.ValidatePtr2(project, item, "MediaItem*") then
    fail("Анализ завершён, но исходный item или проект уже закрыт.")
    return
  end

  local choice, input_error = confirm(bpm, title)
  if input_error then
    fail(input_error)
    return
  end
  if not choice then return end

  apply_result(project, item, choice.bpm)
end

-- Entry point -----------------------------------------------------------------

local function main()
  local project = reaper.EnumProjects(-1, "")

  if analysis_in_progress() then
    fail("Анализ уже идёт. Дождитесь результата.")
    return
  end

  local item, take, target_error = resolve_target(project)
  if not item then
    fail(target_error)
    return
  end

  local source, nested = root_source(take)
  if not source then
    fail("Не удалось получить media source выбранного take.")
    return
  end

  local playrate = take_playrate(take)
  local region = analysis_region(item, take, source, nested)
  local hinted_bpm, hint_origin = exact_bpm_hint(source)

  local audio_path = reaper.GetMediaSourceFileName(source)
  if not audio_path or audio_path == "" or not reaper.file_exists(audio_path) then
    -- Nothing to analyse, but a number written into the take is still a number.
    if hinted_bpm then
      local bpm = quantize_bpm(hinted_bpm * playrate, true)
      confirm_and_apply(project, item, bpm, string.format(
        "%s BPM · half %s / double %s · %s", format_bpm(bpm),
        format_bpm(bpm / 2), format_bpm(bpm * 2), hint_origin))
      return
    end
    fail("Движок может анализировать только audio source, сохранённый как файл.")
    return
  end

  local signature = cache_signature(audio_path, region)
  local cached = cache_lookup(signature)
  if cached then
    local bpm, title = describe(cached, hinted_bpm, hint_origin, playrate)
    confirm_and_apply(project, item, bpm, title)
    return
  end

  async_engine(audio_path, region, function(result, analysis_error)
    if not result then
      fail(analysis_error)
      return
    end
    cache_store(signature, result)
    local bpm, title = describe(result, hinted_bpm, hint_origin, playrate)
    confirm_and_apply(project, item, bpm, title)
  end)
end

main()
