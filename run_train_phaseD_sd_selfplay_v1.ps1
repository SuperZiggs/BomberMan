$ErrorActionPreference = "Stop"

$Python = Join-Path $PSScriptRoot "venv\Scripts\python.exe"
$FallbackCheckpoint = Join-Path $PSScriptRoot "runs\phaseC-sd-best-fallback\best_phaseC.pt"
$OutputDirectory = Join-Path $PSScriptRoot "runs\phaseD-sd-selfplay-v1"
$BestCheckpoint = Join-Path $OutputDirectory "best_phaseD.pt"
$BaselineEvaluation = Join-Path $OutputDirectory "baseline-phaseC-evaluation-200.json"
$CandidateEvaluation = Join-Path $OutputDirectory "evaluation-200.json"
$SnapshotEvaluation = Join-Path $OutputDirectory "snapshot-pool-evaluation.json"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python interpreter not found: $Python"
}
if (-not (Test-Path -LiteralPath $FallbackCheckpoint)) {
    throw "Frozen Phase C fallback checkpoint not found: $FallbackCheckpoint"
}
if (Test-Path -LiteralPath $OutputDirectory) {
    throw "Output directory already exists: $OutputDirectory"
}

& $Python (Join-Path $PSScriptRoot "train_actor_critic.py") `
    --phase D `
    --episodes 2000 `
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
    --seed 2026101601 `
    --resume_from $FallbackCheckpoint `
    --snapshot_every 200 `
    --validation_every 250 `
    --validation_episodes 200 `
    --validation_seed 32000 `
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
    throw "Phase D training failed with exit code $LASTEXITCODE"
}
if (-not (Test-Path -LiteralPath $BestCheckpoint)) {
    throw "Best Phase D checkpoint not found after training: $BestCheckpoint"
}

foreach ($Evaluation in @(
    @{ Model = $FallbackCheckpoint; Output = $BaselineEvaluation; Label = "Phase C fallback baseline" },
    @{ Model = $BestCheckpoint; Output = $CandidateEvaluation; Label = "Phase D best" }
)) {
    & $Python (Join-Path $PSScriptRoot "evaluate.py") `
        --mode seeded `
        --ac_model $Evaluation.Model `
        --board_size 9 `
        --num_agents 2 `
        --num_episodes 200 `
        --max_steps 500 `
        --seed 33000 `
        --device cpu `
        --sudden_death `
        --sudden_death_start 475 `
        --sudden_death_interval 25 `
        --sudden_death_warning_steps 40 `
        --output $Evaluation.Output
    if ($LASTEXITCODE -ne 0) {
        throw "$($Evaluation.Label) evaluation failed with exit code $LASTEXITCODE"
    }
    if (-not (Test-Path -LiteralPath $Evaluation.Output)) {
        throw "$($Evaluation.Label) evaluation output not found: $($Evaluation.Output)"
    }
}

& $Python (Join-Path $PSScriptRoot "evaluate_phase_d_snapshot_pool.py") `
    --ac_model $BestCheckpoint `
    --snapshot_dir $OutputDirectory `
    --output $SnapshotEvaluation `
    --board_size 9 `
    --num_agents 2 `
    --games_per_snapshot 40 `
    --max_snapshots 5 `
    --seed 34000 `
    --max_steps 500 `
    --device cpu `
    --sudden_death `
    --sudden_death_start 475 `
    --sudden_death_interval 25 `
    --sudden_death_warning_steps 40
if ($LASTEXITCODE -ne 0) {
    throw "Snapshot-pool diagnostic failed with exit code $LASTEXITCODE"
}
if (-not (Test-Path -LiteralPath $SnapshotEvaluation)) {
    throw "Snapshot-pool evaluation output not found: $SnapshotEvaluation"
}

$Metrics = Get-Content -LiteralPath $CandidateEvaluation -Raw | ConvertFrom-Json
$Gates = @(
    [pscustomobject]@{ Name = "WR"; Passed = $Metrics.win_rate -ge 70.0; Value = $Metrics.win_rate; Target = ">= 70.0" },
    [pscustomobject]@{ Name = "ActiveWR"; Passed = $Metrics.active_win_rate -ge 55.0; Value = $Metrics.active_win_rate; Target = ">= 55.0" },
    [pscustomobject]@{ Name = "Draw"; Passed = $Metrics.draw_rate -le 15.0; Value = $Metrics.draw_rate; Target = "<= 15.0" },
    [pscustomobject]@{ Name = "Suicide"; Passed = $Metrics.suicide_rate -le 1.5; Value = $Metrics.suicide_rate; Target = "<= 1.5" },
    [pscustomobject]@{ Name = "SD death"; Passed = $Metrics.sudden_death_death_rate -le 10.0; Value = $Metrics.sudden_death_death_rate; Target = "<= 10.0" }
)

Write-Output ""
Write-Output "Phase D best checkpoint gates (200 seeded games vs RuleBot)"
foreach ($Gate in $Gates) {
    $Status = if ($Gate.Passed) { "PASS" } else { "FAIL" }
    Write-Output ("[{0}] {1}: {2}% (target {3}%)" -f $Status, $Gate.Name, $Gate.Value, $Gate.Target)
}
exit 0
