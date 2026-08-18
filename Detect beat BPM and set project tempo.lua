-- @description Detect beat BPM with TempoCNN and set project tempo
-- @version 2.0.0
-- @author reaperplug
-- @about
--   Uses TempoCNN (deeptemp_k16).
--   Select an audio item (or park the edit cursor on one), then run the script.
--   The detected BPM is shown for confirmation before the project tempo changes.

local ANALYSIS_TIMEOUT_MS = 180000

local MIN_ACCEPTED_BPM = 20
local MAX_ACCEPTED_BPM = 960

local ENGINE_FILENAME = "tempocnn_engine.exe"
local LAUNCHER_FILENAME = "launch_engine.ps1"
local RUNNER_FILENAME = "run_engine.ps1"
local MODEL_FILENAME = "deeptemp_k16.onnx"
local RESULT_PREFIX = "bpm_result_"

local EXT_SECTION = "reaperplug_engine_bpm"
local EXT_RUNNING_KEY = "analysis_started_at"

local SEPARATOR = package.config:sub(1, 1)

local function fail(message)
  reaper.MB(message, "TempoCNN BPM Detector", 0)
  return nil
end

-- Whole numbers only: a fractional project tempo is not what this action is for.
local function round_bpm(bpm)
  return math.floor(bpm + 0.5)
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

local function root_source(take)
  local source = reaper.GetMediaItemTake_Source(take)
  while source do
    local parent = reaper.GetMediaSourceParent(source)
    if not parent then break end
    source = parent
  end
  return source
end

-- Exact hints -----------------------------------------------------------------

local function parse_bpm(text)
  if not text then return nil end
  text = text:gsub(",", ".")
  local value = tonumber(text:match("%d+%.?%d*"))
  if value and value >= MIN_ACCEPTED_BPM and value <= 400 then return value end
  return nil
end

local function exact_bpm_hint(take, source)
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
    if name:sub(1, #RESULT_PREFIX) == RESULT_PREFIX then
      stale[#stale + 1] = name
    end
    index = index + 1
  end
  -- Collected first: deleting during enumeration would shift the indices.
  for _, name in ipairs(stale) do
    os.remove(engine_directory .. SEPARATOR .. name)
  end
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
    confidence = tonumber(output:match("confidence:%s*([%d%.%-]+)")) or 0,
  }
end

local function read_text_file(path)
  local file = io.open(path, "rb")
  if not file then return nil end
  local content = file:read("*a")
  file:close()
  return content
end

-- Analysis state --------------------------------------------------------------

local function set_analysis_running(running)
  if running then
    reaper.SetExtState(EXT_SECTION, EXT_RUNNING_KEY, tostring(os.time()), false)
  else
    reaper.DeleteExtState(EXT_SECTION, EXT_RUNNING_KEY, false)
  end
end

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

local function async_engine(audio_path, on_complete)
  local paths, paths_error = engine_paths()
  if not paths then
    fail(paths_error)
    return false
  end

  remove_stale_results(paths.directory)

  local unique = reaper.genGuid():gsub("[^%w]", "")
  local result_path = paths.directory .. SEPARATOR .. RESULT_PREFIX .. unique .. ".txt"
  local windows_directory = os.getenv("WINDIR") or "C:\\Windows"
  local powershell_path = windows_directory
    .. "\\System32\\WindowsPowerShell\\v1.0\\powershell.exe"
  if not reaper.file_exists(powershell_path) then
    fail("Не найден системный Windows PowerShell:\n" .. powershell_path)
    return false
  end

  -- The system path contains no spaces. Leaving the executable unquoted avoids
  -- cmd.exe's special first-quote parsing used internally by os.execute.
  local launch_command = powershell_path
    .. " -NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File "
    .. quote_argument(paths.launcher) .. " "
    .. quote_argument(paths.runner) .. " "
    .. quote_argument(paths.engine) .. " "
    .. quote_argument(audio_path) .. " "
    .. quote_argument(result_path)

  local execute_result, execute_reason, execute_code = os.execute(launch_command)
  local launch_ok = execute_result == true or execute_result == 0
  if not launch_ok then
    fail("Не удалось запустить движок в фоне. Код: "
      .. tostring(execute_code or execute_result)
      .. "; причина: " .. tostring(execute_reason or "неизвестна"))
    return false
  end

  set_analysis_running(true)
  local started = reaper.time_precise()

  local function poll()
    if reaper.file_exists(result_path) then
      local output = read_text_file(result_path)
      os.remove(result_path)
      set_analysis_running(false)
      on_complete(parse_engine_output(output))
      return
    end

    if (reaper.time_precise() - started) * 1000 > ANALYSIS_TIMEOUT_MS then
      set_analysis_running(false)
      fail("Движок не завершил анализ за отведённое время.")
      return
    end

    reaper.defer(poll)
  end

  reaper.defer(poll)
  return true
end

-- Tempo -----------------------------------------------------------------------

local function take_playrate(take)
  local playrate = reaper.GetMediaItemTakeInfo_Value(take, "D_PLAYRATE")
  if not playrate or playrate <= 0 then return 1 end
  return playrate
end

local function apply_result(project, item, bpm)
  bpm = round_bpm(bpm)
  reaper.Undo_BeginBlock()
  -- Project tempo must change without stretching the reference beat.
  reaper.SetMediaItemInfo_Value(item, "C_BEATATTACHMODE", 0)
  reaper.SetCurrentBPM(project, bpm, false)
  reaper.Undo_EndBlock("Detect beat BPM with TempoCNN and set project tempo", -1)
  reaper.UpdateArrange()
  return bpm
end

-- Confirmation ----------------------------------------------------------------

-- Returns a choice table, or nil plus an error message. Both nil means cancel.
local function confirm(detected_bpm, origin)
  local rounded = round_bpm(detected_bpm)
  -- Metrical level is the tracker's weakest guess, so both neighbours are
  -- spelled out and either one is a single keystroke away.
  local title = string.format("Detected %d BPM (half %d / double %d, %s)",
    rounded, round_bpm(detected_bpm / 2), round_bpm(detected_bpm * 2), origin)
  local ok, value = reaper.GetUserInputs(title, 1, "BPM:,extrawidth=100", tostring(rounded))
  if not ok then return nil end

  local bpm = tonumber((value or ""):match("^%s*(.-)%s*$"))
  if not bpm or bpm < MIN_ACCEPTED_BPM or bpm > MAX_ACCEPTED_BPM then
    return nil, string.format(
      "Введите BPM числом от %d до %d.", MIN_ACCEPTED_BPM, MAX_ACCEPTED_BPM
    )
  end
  return { bpm = bpm }
end

local function confirm_and_apply(project, item, bpm, origin)
  if not reaper.ValidatePtr2(project, item, "MediaItem*") then
    fail("Анализ завершён, но исходный item или проект уже закрыт.")
    return
  end

  local choice, input_error = confirm(bpm, origin)
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

  local source = root_source(take)
  if not source then
    fail("Не удалось получить media source выбранного take.")
    return
  end

  local playrate = take_playrate(take)

  local hinted_bpm, hint_origin = exact_bpm_hint(take, source)
  if hinted_bpm then
    -- An exact hint is trusted as written; only playrate can move it.
    confirm_and_apply(project, item, hinted_bpm * playrate, hint_origin)
    return
  end

  local audio_path = reaper.GetMediaSourceFileName(source)
  if not audio_path or audio_path == "" or not reaper.file_exists(audio_path) then
    fail("Движок может анализировать только audio source, сохранённый как файл.")
    return
  end

  async_engine(audio_path, function(result, analysis_error)
    if not result then
      fail(analysis_error)
      return
    end
    local effective_bpm = result.bpm * playrate
    local origin = string.format("conf %.2f", result.confidence)
    confirm_and_apply(project, item, effective_bpm, origin)
  end)
end

main()
