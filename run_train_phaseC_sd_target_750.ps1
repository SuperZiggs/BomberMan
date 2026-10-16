param(
    [switch]$EvaluateOnly
)

$ErrorActionPreference = "Stop"

function ConvertTo-StrictFiniteNumber {
    param(
        [object]$Value,
        [string]$Field,
        [int]$GameIndex = -1
    )

    if ($null -eq $Value) {
        throw "Evaluation value is null for $Field"
    }
    $NumericTypes = @(
        [System.TypeCode]::Byte,
        [System.TypeCode]::SByte,
        [System.TypeCode]::Int16,
        [System.TypeCode]::UInt16,
        [System.TypeCode]::Int32,
        [System.TypeCode]::UInt32,
        [System.TypeCode]::Int64,
        [System.TypeCode]::UInt64,
        [System.TypeCode]::Single,
        [System.TypeCode]::Double,
        [System.TypeCode]::Decimal
    )
    $TypeCode = [System.Type]::GetTypeCode($Value.GetType())
    if ($NumericTypes -notcontains $TypeCode) {
        $Location = if ($GameIndex -ge 0) { " at game index $GameIndex" } else { "" }
        throw "Evaluation field $Field$Location must be numeric"
    }
    $Number = [double]$Value
    if ([double]::IsNaN($Number) -or [double]::IsInfinity($Number)) {
        throw "Evaluation field $Field must be finite"
    }
    return $Number
}

function Get-ValidatedGateMetrics {
    param(
        [object]$Artifact,
        [int]$ExpectedCount,
        [int]$MaximumSteps
    )

    $BinaryFields = @(
        "win",
        "active_win",
        "sudden_death_win",
        "passive_win",
        "suicide",
        "sudden_death_death",
        "draw"
    )
    $NumericFields = @("kills", "bombs", "items", "steps")
    $Sums = @{}
    foreach ($Field in $BinaryFields) {
        $Sums[$Field] = 0.0
    }

    for ($Index = 0; $Index -lt $ExpectedCount; $Index++) {
        $Game = $Artifact.games[$Index]
        foreach ($Field in $BinaryFields + $NumericFields) {
            if ($Game.PSObject.Properties.Name -notcontains $Field) {
                throw "Evaluation game $Index missing required outcome field: $Field"
            }
            $Number = ConvertTo-StrictFiniteNumber -Value $Game.$Field -Field $Field -GameIndex $Index
            if ($Number -lt 0.0) {
                throw "Evaluation game $Index has negative $Field"
            }
            if ($BinaryFields -contains $Field) {
                if ($Number -ne 0.0 -and $Number -ne 1.0) {
                    throw "Evaluation game $Index field $Field must be 0 or 1"
                }
                $Sums[$Field] += $Number
            }
            elseif ($Field -eq "steps" -and $Number -gt $MaximumSteps) {
                throw "Evaluation game $Index steps exceeds max_steps"
            }
        }

        $ClassifiedWins = [double]$Game.active_win + [double]$Game.sudden_death_win + [double]$Game.passive_win
        if ($ClassifiedWins -ne [double]$Game.win) {
            throw "Evaluation game $Index has inconsistent win classification"
        }
        if ([double]$Game.win + [double]$Game.draw -gt 1.0) {
            throw "Evaluation game $Index cannot be both a win and a draw"
        }
    }

    $MetricSources = [ordered]@{
        win_rate = "win"
        active_win_rate = "active_win"
        draw_rate = "draw"
        sudden_death_death_rate = "sudden_death_death"
        suicide_rate = "suicide"
    }
    $Computed = [ordered]@{}
    foreach ($Entry in $MetricSources.GetEnumerator()) {
        if ($Artifact.PSObject.Properties.Name -notcontains $Entry.Key) {
            throw "Evaluation output missing required metric: $($Entry.Key)"
        }
        $ComputedValue = $Sums[$Entry.Value] / $ExpectedCount * 100.0
        $ReportedValue = ConvertTo-StrictFiniteNumber -Value $Artifact.($Entry.Key) -Field $Entry.Key
        if ([math]::Abs($ReportedValue - $ComputedValue) -gt 0.000000001) {
            throw "Evaluation metric $($Entry.Key) does not match game outcomes"
        }
        $Computed[$Entry.Key] = $ComputedValue
    }

    return [pscustomobject]$Computed
}

$Python = Join-Path $PSScriptRoot "venv\Scripts\python.exe"
$Checkpoint = Join-Path $PSScriptRoot "runs\phaseC-sd-attack-750\best_phaseC.pt"
$OutputDirectory = Join-Path $PSScriptRoot "runs\phaseC-sd-target-750"
$BestCheckpoint = Join-Path $OutputDirectory "best_phaseC.pt"
$EvaluationOutput = Join-Path $OutputDirectory "evaluation-200.json"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python interpreter not found: $Python"
}
if ($EvaluateOnly) {
    if (-not (Test-Path -LiteralPath $OutputDirectory -PathType Container)) {
        throw "Evaluation-only output directory not found: $OutputDirectory"
    }
    if (-not (Test-Path -LiteralPath $BestCheckpoint -PathType Leaf)) {
        throw "Best checkpoint not found for evaluation-only mode: $BestCheckpoint"
    }
}
else {
    if (-not (Test-Path -LiteralPath $Checkpoint -PathType Leaf)) {
        throw "Warm-start checkpoint not found: $Checkpoint"
    }
    if (Test-Path -LiteralPath $OutputDirectory) {
        throw "Output directory already exists: $OutputDirectory"
    }

    Write-Output "[Stage 1/2] Training Phase C-SD target model"
    & $Python (Join-Path $PSScriptRoot "train_actor_critic.py") `
        --phase C `
        --episodes 750 `
        --board_size 9 `
        --num_agents 2 `
        --max_steps 500 `
        --lr 0.000025 `
        --entropy_coef 0.0075 `
        --wall_destroy_bonus 0.15 `
        --attack_lookahead 2 `
        --min_pressure_gain 0.15 `
        --trap_sequence_depth 2 `
        --sequence_pressure_gain 0.35 `
        --ring_safety_coef 0.6 `
        --late_game_start 350 `
        --late_game_attack_multiplier 1.75 `
        --escape_loss_coef 0.5 `
        --combat_loss_coef 0.9 `
        --navigation_loss_coef 0.4 `
        --seed 2026101503 `
        --resume_from $Checkpoint `
        --validation_every 250 `
        --validation_episodes 200 `
        --validation_seed 30000 `
        --sudden_death_probability 0.75 `
        --sudden_death_start 475 `
        --sudden_death_interval 25 `
        --sudden_death_warning_steps 40 `
        --target_win_rate 70 `
        --target_active_win_rate 55 `
        --target_draw_rate 15 `
        --target_sudden_death_death_rate 10 `
        --max_suicide_rate 1.5 `
        --checkpoint_dir $OutputDirectory

    if ($LASTEXITCODE -ne 0) {
        throw "Training failed with exit code $LASTEXITCODE"
    }
    if (-not (Test-Path -LiteralPath $BestCheckpoint -PathType Leaf)) {
        throw "Best checkpoint not found after training: $BestCheckpoint"
    }
}

Write-Output "[Stage 2/2] Evaluating selected best checkpoint"
Write-Output "Do not close this window until evaluation-200.json is confirmed."
& $Python (Join-Path $PSScriptRoot "evaluate.py") `
    --mode seeded `
    --ac_model $BestCheckpoint `
    --board_size 9 `
    --num_agents 2 `
    --num_episodes 200 `
    --max_steps 500 `
    --seed 31000 `
    --device cpu `
    --sudden_death `
    --sudden_death_start 475 `
    --sudden_death_interval 25 `
    --sudden_death_warning_steps 40 `
    --resume_partial `
    --partial_every 10 `
    --output $EvaluationOutput

if ($LASTEXITCODE -ne 0) {
    throw "Evaluation failed with exit code $LASTEXITCODE"
}
if (-not (Test-Path -LiteralPath $EvaluationOutput)) {
    throw "Evaluation output not found: $EvaluationOutput"
}

$Metrics = Get-Content -LiteralPath $EvaluationOutput -Raw | ConvertFrom-Json
$ExpectedFields = @{
    seed = 31000
    board_size = 9
    num_agents = 2
    max_steps = 500
    sudden_death = $true
    sudden_death_start = 475
    sudden_death_interval = 25
    sudden_death_warning_steps = 40
    num_episodes = 200
}
foreach ($Entry in $ExpectedFields.GetEnumerator()) {
    if ($Metrics.PSObject.Properties.Name -notcontains $Entry.Key) {
        throw "Evaluation output missing required field: $($Entry.Key)"
    }
    if ($Metrics.($Entry.Key) -ne $Entry.Value) {
        throw "Evaluation output mismatch for $($Entry.Key): expected $($Entry.Value), got $($Metrics.($Entry.Key))"
    }
}

if ($Metrics.PSObject.Properties.Name -notcontains "games" -or $null -eq $Metrics.games) {
    throw "Evaluation output missing games"
}
if ($Metrics.games.Count -ne 200) {
    throw "Evaluation output must contain exactly 200 games"
}
for ($Index = 0; $Index -lt 200; $Index++) {
    $ExpectedSeed = 31000 + [math]::Floor($Index / 2)
    $ExpectedAid = $Index % 2
    $Game = $Metrics.games[$Index]
    if ($Game.seed -ne $ExpectedSeed -or $Game.aid -ne $ExpectedAid) {
        throw "Evaluation games are incomplete or out of order at index $Index"
    }
}

$RequiredMetrics = @(
    "win_rate",
    "active_win_rate",
    "draw_rate",
    "sudden_death_death_rate",
    "suicide_rate"
)
foreach ($Field in $RequiredMetrics) {
    if ($Metrics.PSObject.Properties.Name -notcontains $Field) {
        throw "Evaluation output missing required metric: $Field"
    }
}
$GateMetrics = Get-ValidatedGateMetrics -Artifact $Metrics -ExpectedCount 200 -MaximumSteps 500

Write-Output "Confirmed evaluation artifact: $EvaluationOutput"
$Gates = @(
    [pscustomobject]@{ Name = "WR"; Passed = $GateMetrics.win_rate -ge 70.0; Value = $GateMetrics.win_rate; Target = ">= 70.0" },
    [pscustomobject]@{ Name = "ActiveWR"; Passed = $GateMetrics.active_win_rate -ge 55.0; Value = $GateMetrics.active_win_rate; Target = ">= 55.0" },
    [pscustomobject]@{ Name = "Draw"; Passed = $GateMetrics.draw_rate -le 15.0; Value = $GateMetrics.draw_rate; Target = "<= 15.0" },
    [pscustomobject]@{ Name = "SD death"; Passed = $GateMetrics.sudden_death_death_rate -le 10.0; Value = $GateMetrics.sudden_death_death_rate; Target = "<= 10.0" },
    [pscustomobject]@{ Name = "Suicide"; Passed = $GateMetrics.suicide_rate -le 1.5; Value = $GateMetrics.suicide_rate; Target = "<= 1.5" }
)

Write-Output ""
Write-Output "Phase C-SD target evaluation gates"
foreach ($Gate in $Gates) {
    $Status = if ($Gate.Passed) { "PASS" } else { "FAIL" }
    Write-Output ("[{0}] {1}: {2}% (target {3}%)" -f $Status, $Gate.Name, $Gate.Value, $Gate.Target)
}

if ($Gates.Passed -notcontains $false) {
    Write-Output "[PASS] Phase C-SD target checkpoint passed all gates."
}
else {
    Write-Output "[FAIL] Phase C-SD target checkpoint did not pass all gates."
}

exit 0
