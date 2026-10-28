param([switch]$ContinueAfterCollect)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath 'E:\Bomberman'
$python = 'E:\Tools\python.exe'
$runner = 'E:\Bomberman\run_sd10_distill.py'
$output = 'E:\Bomberman\runs\sd10-20261026'
$control = 'E:\Bomberman\runs\sd10-20261026-control'
$statePath = Join-Path $control 'state.json'
$script:stage = 'preflight'

function Write-State([string]$status, [string]$message) {
    [ordered]@{
        status = $status
        stage = $script:stage
        message = $message
        updated_at = (Get-Date).ToString('o')
        launcher_pid = $PID
        python = $python
        output = $output
    } | ConvertTo-Json | Set-Content -LiteralPath $statePath -Encoding UTF8
}

function Invoke-Stage([string]$name, [string[]]$stageArgs) {
    $script:stage = $name
    Write-State 'running' "Started $name"
    Write-Host "`n[$(Get-Date -Format o)] $name" -ForegroundColor Cyan
    # Windows PowerShell represents native stderr as ErrorRecord objects.
    $ErrorActionPreference = 'Continue'
    & $python @stageArgs 2>&1 | ForEach-Object { $_.ToString() } |
        Tee-Object -FilePath (Join-Path $control "$name.log")
    $stageExitCode = $LASTEXITCODE
    $ErrorActionPreference = 'Stop'
    if ($stageExitCode -ne 0) {
        throw "$name exited with code $stageExitCode. See $control\$name.log"
    }
    Write-State 'running' "Finished $name"
}

try {
    if ($ContinueAfterCollect) {
        if (-not (Test-Path -LiteralPath (Join-Path $output 'collection_summary.json'))) {
            throw 'Collection summary is missing; collection must finish before training.'
        }
        foreach ($marker in @('train', 'selected.json', 'test_opened.json')) {
            if (Test-Path -LiteralPath (Join-Path $output $marker)) {
                throw "Continuation refused because $marker already exists."
            }
        }
        Write-State 'running' 'Continuing SD10 after completed collection'
    } else {
        if (Test-Path -LiteralPath $output) {
            throw "Output already exists: $output. Refusing to start a duplicate run."
        }
        Write-State 'running' 'Preparing SD10'
        Invoke-Stage 'install-tqdm' @('-m', 'pip', 'install', '--no-input', 'tqdm')
        Invoke-Stage 'checkpoints' @('-c', "import torch; import run_sd10_distill as s; s._check_checkpoints(('S2', 'G', 'S3', 'F')); print('Checkpoints S2/G/S3/F: OK'); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')")
        Invoke-Stage 'unit-tests' @('-m', 'unittest', 'test_run_sd10', '-q')
        Invoke-Stage 'collect' @('-u', $runner, 'collect', '--output', $output, '--workers', '5')
    }
    Invoke-Stage 'train' @('-u', $runner, 'train', '--output', $output)
    Invoke-Stage 'validate' @('-u', $runner, 'validate', '--output', $output, '--workers', '5')
    $selection = Get-Content -LiteralPath (Join-Path $output 'selected.json') -Raw | ConvertFrom-Json
    if ($selection.proceed_to_test -eq $true) {
        Invoke-Stage 'test' @('-u', $runner, 'test', '--output', $output, '--workers', '5')
        $summary = Get-Content -LiteralPath (Join-Path $output 'summary.json') -Raw | ConvertFrom-Json
        Write-State 'complete' ("SD10 finished. Gate pass: " + $summary.gate.pass)
    } else {
        Write-State 'validation_stop' 'No candidate qualified to open the test seed. Keep S3.'
        Write-Host 'VALIDATION_STOP: keeping S3; test seed stays unopened.' -ForegroundColor Yellow
    }
} catch {
    Write-State 'failed' $_.Exception.Message
    Write-Host $_.Exception.Message -ForegroundColor Red
}
