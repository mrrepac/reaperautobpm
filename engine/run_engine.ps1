param(
    [Parameter(Mandatory = $true)][string]$EnginePath,
    [Parameter(Mandatory = $true)][string]$JobPath,
    [Parameter(Mandatory = $true)][string]$ResultPath
)

$ErrorActionPreference = 'Stop'
$workingPath = "$ResultPath.running"
$exitCode = 125
# No BOM: the Lua side reads the result as plain bytes.
$utf8 = New-Object System.Text.UTF8Encoding($false)
# Latin-1 maps every byte to the char of the same value, so the engine's output
# survives the redirect untouched and can be decoded once it is all here.
$byteSafe = [System.Text.Encoding]::GetEncoding(28591)

# The engine is a PyInstaller build: its interpreter runs with an isolated
# config that ignores PYTHONIOENCODING, so it writes stdio in the system ANSI
# codepage. A source-run engine writes UTF-8. Try the strict UTF-8 reading
# first and fall back to ANSI, which covers both without guessing wrong.
function ConvertTo-EngineText([string]$Raw) {
    $bytes = $byteSafe.GetBytes($Raw)
    $strictUtf8 = New-Object System.Text.UTF8Encoding($false, $true)
    try { return $strictUtf8.GetString($bytes) }
    catch {
        $ansi = [System.Text.Encoding]::GetEncoding(
            [System.Globalization.CultureInfo]::CurrentCulture.TextInfo.ANSICodePage)
        return $ansi.GetString($bytes)
    }
}

# Written before anything can fail so the polling script can tell "still
# working" apart from "never started".
[System.IO.File]::WriteAllText($workingPath, "started`r`n", $utf8)

try {
    if (-not (Test-Path -LiteralPath $JobPath)) {
        throw "Job file is missing: $JobPath"
    }

    # The job file carries the audio path and the region to analyse, and the
    # engine reads it itself: a non-Latin path does not survive the ANSI hops
    # on the way here, so it never travels as an argument.
    $startInfo = New-Object System.Diagnostics.ProcessStartInfo
    $startInfo.FileName = $EnginePath
    $startInfo.Arguments = '--job "' + $JobPath.Replace('"', '\"') + '"'
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $true
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true
    # Without this the output is decoded as the console codepage, so any
    # non-Latin path in an error message comes back as mojibake.
    $startInfo.StandardOutputEncoding = $byteSafe
    $startInfo.StandardErrorEncoding = $byteSafe
    # Honoured only by a source-run engine; harmless for the frozen build.
    $startInfo.EnvironmentVariables['PYTHONIOENCODING'] = 'utf-8'
    $startInfo.EnvironmentVariables['PYTHONUTF8'] = '1'

    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $startInfo
    if (-not $process.Start()) {
        throw 'ProcessStartInfo could not start the engine.'
    }
    $stdoutTask = $process.StandardOutput.ReadToEndAsync()
    $stderrTask = $process.StandardError.ReadToEndAsync()

    if (-not $process.WaitForExit(180000)) {
        $process.Kill()
        $process.WaitForExit()
        $exitCode = 124
    }
    else {
        $exitCode = [int]$process.ExitCode
    }

    $stdout = ConvertTo-EngineText $stdoutTask.GetAwaiter().GetResult()
    $stderr = ConvertTo-EngineText $stderrTask.GetAwaiter().GetResult()
    $content = $stdout + $stderr + "`r`n__ENGINE_EXIT__=$exitCode`r`n"
}
catch {
    $content = "Engine runner error: $($_.Exception.Message)`r`n__ENGINE_EXIT__=125`r`n"
}

[System.IO.File]::WriteAllText($workingPath, $content, $utf8)
Move-Item -LiteralPath $workingPath -Destination $ResultPath -Force
