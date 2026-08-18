param(
    [Parameter(Mandatory = $true)][string]$RunnerPath,
    [Parameter(Mandatory = $true)][string]$EnginePath,
    [Parameter(Mandatory = $true)][string]$AudioPath,
    [Parameter(Mandatory = $true)][string]$ResultPath
)

$ErrorActionPreference = 'Stop'

function Quote-ProcessArgument([string]$Value) {
    return '"' + $Value.Replace('"', '\"') + '"'
}

$runnerArguments = @(
    '-NoProfile'
    '-NonInteractive'
    '-ExecutionPolicy Bypass'
    '-File ' + (Quote-ProcessArgument $RunnerPath)
    (Quote-ProcessArgument $EnginePath)
    (Quote-ProcessArgument $AudioPath)
    (Quote-ProcessArgument $ResultPath)
) -join ' '

Start-Process `
    -FilePath "$PSHOME\powershell.exe" `
    -ArgumentList $runnerArguments `
    -WindowStyle Hidden | Out-Null
