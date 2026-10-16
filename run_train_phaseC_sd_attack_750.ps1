$ErrorActionPreference = "Stop"

$Python = Join-Path $PSScriptRoot "venv\Scripts\python.exe"
$Checkpoint = Join-Path $PSScriptRoot "runs\phaseC-sd-1500-e\best_phaseC.pt"
$OutputDirectory = Join-Path $PSScriptRoot "runs\phaseC-sd-attack-750"
$BestCheckpoint = Join-Path $OutputDirectory "best_phaseC.pt"
$EvaluationOutput = Join-Path $OutputDirectory "evaluation-200.json"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python interpreter not found: $Python"
}
if (-not (Test-Path -LiteralPath $Checkpoint)) {
    throw "Warm-start checkpoint not found: $Checkpoint"
}
if (Test-Path -LiteralPath $OutputDirectory) {
    throw "Output directory already exists: $OutputDirectory"
}

& $Python (Join-Path $PSScriptRoot "train_actor_critic.py") `
    --phase C `
    --episodes 750 `
    --board_size 9 `
    --num_agents 2 `
    --max_steps 500 `
    --lr 0.00005 `
    --entropy_coef 0.01 `
    --wall_destroy_bonus 0.15 `
    --attack_lookahead 2 `
    --min_pressure_gain 0.15 `
    --late_game_start 400 `
    --late_game_attack_multiplier 1.5 `
    --escape_loss_coef 0.5 `
    --combat_loss_coef 0.75 `
    --navigation_loss_coef 0.35 `
    --seed 2026101502 `
    --resume_from $Checkpoint `
    --validation_every 250 `
    --validation_episodes 100 `
    --validation_seed 28000 `
    --sudden_death_probability 0.75 `
    --sudden_death_start 475 `
    --sudden_death_interval 25 `
    --sudden_death_warning_steps 40 `
    --checkpoint_dir $OutputDirectory

if ($LASTEXITCODE -ne 0) {
    throw "Training failed with exit code $LASTEXITCODE"
}
if (-not (Test-Path -LiteralPath $BestCheckpoint)) {
    throw "Best checkpoint not found after training: $BestCheckpoint"
}

& $Python (Join-Path $PSScriptRoot "evaluate.py") `
    --mode seeded `
    --ac_model $BestCheckpoint `
    --board_size 9 `
    --num_agents 2 `
    --num_episodes 200 `
    --max_steps 500 `
    --seed 29000 `
    --device cpu `
    --sudden_death `
    --sudden_death_start 475 `
    --sudden_death_interval 25 `
    --sudden_death_warning_steps 40 `
    --output $EvaluationOutput

if ($LASTEXITCODE -ne 0) {
    throw "Evaluation failed with exit code $LASTEXITCODE"
}
if (-not (Test-Path -LiteralPath $EvaluationOutput)) {
    throw "Evaluation output not found: $EvaluationOutput"
}

$Metrics = Get-Content -LiteralPath $EvaluationOutput -Raw | ConvertFrom-Json
$Gates = @(
    [pscustomobject]@{ Name = "active_win_rate"; Passed = $Metrics.active_win_rate -ge 38.0; Value = $Metrics.active_win_rate; Target = ">= 38.0" },
    [pscustomobject]@{ Name = "win_rate"; Passed = $Metrics.win_rate -ge 60.5; Value = $Metrics.win_rate; Target = ">= 60.5" },
    [pscustomobject]@{ Name = "draw_rate"; Passed = $Metrics.draw_rate -le 23.5; Value = $Metrics.draw_rate; Target = "<= 23.5" },
    [pscustomobject]@{ Name = "suicide_rate"; Passed = $Metrics.suicide_rate -le 1.5; Value = $Metrics.suicide_rate; Target = "<= 1.5" },
    [pscustomobject]@{ Name = "sudden_death_death_rate"; Passed = $Metrics.sudden_death_death_rate -le 18.5; Value = $Metrics.sudden_death_death_rate; Target = "<= 18.5" }
)

Write-Output ""
Write-Output "Phase C-SD attack evaluation gates"
foreach ($Gate in $Gates) {
    $Status = if ($Gate.Passed) { "PASS" } else { "FAIL" }
    Write-Output ("[{0}] {1}: {2}% (target {3}%)" -f $Status, $Gate.Name, $Gate.Value, $Gate.Target)
}

if ($Gates.Passed -notcontains $false) {
    Write-Output "[PASS] Phase C-SD attack checkpoint passed all gates."
}
else {
    Write-Output "[FAIL] Phase C-SD attack checkpoint did not pass all gates."
}

exit 0
