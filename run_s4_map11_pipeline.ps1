param(
    [string]$Output = 'E:\Bomberman\runs\s4-map11-cnn-20261031-v1',
    [int]$Workers = 5
)
$ErrorActionPreference = 'Stop'
$Python = 'E:\Tools\python.exe'
$Runner = Join-Path $PSScriptRoot 'run_s4_map11_distill.py'
$Control = 'E:\Bomberman\runs\s4-map11-cnn-20261031-v1-control'
New-Item -ItemType Directory -Path $Control -ErrorAction Stop | Out-Null
function Write-State([string]$Stage, [string]$Status, [string]$Message = '') {
    $state = @{ pid = $PID; stage = $Stage; status = $Status; message = $Message;
                output = $Output; updated = (Get-Date).ToString('o') }
    $temporary = Join-Path $Control 'state.tmp.json'
    $state | ConvertTo-Json | Set-Content -LiteralPath $temporary -Encoding UTF8
    Move-Item -LiteralPath $temporary -Destination (Join-Path $Control 'state.json') -Force
}
function Invoke-Stage([string]$Stage) {
    Write-State $Stage 'running'
    $log = Join-Path $Control "$Stage.log"
    & $Python $Runner $Stage --output $Output --workers $Workers *> $log
    if ($LASTEXITCODE -ne 0) { throw "Stage $Stage failed with exit code $LASTEXITCODE; see $log" }
}
try {
    Set-Location -LiteralPath $PSScriptRoot
    Write-State 'init' 'running'
    $initLog = Join-Path $Control 'init.log'
    & $Python $Runner init --output $Output *> $initLog
    if ($LASTEXITCODE -ne 0) { throw "Initialization failed; see $initLog" }
    Invoke-Stage 'pilot'
    $pilot = Get-Content -LiteralPath (Join-Path $Output 'pilot.json') -Raw | ConvertFrom-Json
    if (-not $pilot.proceed_to_collect) {
        Write-State 'pilot' 'pilot_stop' 'Pilot did not meet preregistered readiness conditions.'
        return
    }
    Invoke-Stage 'collect'
    Invoke-Stage 'train'
    Invoke-Stage 'validate'
    $selected = Get-Content -LiteralPath (Join-Path $Output 'selected.json') -Raw | ConvertFrom-Json
    if (-not $selected.proceed_to_test) {
        Write-State 'validate' 'validation_stop' 'No eligible candidate improved on the validation baseline.'
        return
    }
    Invoke-Stage 'test'
    Write-State 'test' 'complete'
} catch {
    Write-State 'error' 'failed' $_.Exception.Message
    Write-Error $_
}
