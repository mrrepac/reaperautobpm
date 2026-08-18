param(
    [Parameter(Mandatory = $true)][string]$EnginePath,
    [Parameter(Mandatory = $true)][string]$AudioPath,
    [Parameter(Mandatory = $true)][string]$ResultPath
)

$ErrorActionPreference = 'Stop'
$workingPath = "$ResultPath.running"
$exitCode = 125

try {
    $startInfo = New-Object System.Diagnostics.ProcessStartInfo
    $startInfo.FileName = $EnginePath
    $startInfo.Arguments = '"' + $AudioPath.Replace('"', '\"') + '"'
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $true
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true

    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $startInfo
    if (-not $process.Start()) {
        throw 'ProcessStartInfo could not start Essentia.'
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

    $stdout = $stdoutTask.GetAwaiter().GetResult()
    $stderr = $stderrTask.GetAwaiter().GetResult()
    $content = $stdout + $stderr + "`r`n__ESSENTIA_EXIT__=$exitCode`r`n"
}
catch {
    $content = "Essentia runner error: $($_.Exception.Message)`r`n__ESSENTIA_EXIT__=125`r`n"
}

[System.IO.File]::WriteAllText($workingPath, $content, [System.Text.Encoding]::UTF8)
Move-Item -LiteralPath $workingPath -Destination $ResultPath -Force
