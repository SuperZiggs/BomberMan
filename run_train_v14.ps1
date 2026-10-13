$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

$checkpointDirectory = Join-Path $PSScriptRoot "checkpoints_v14"
New-Item -ItemType Directory -Path $checkpointDirectory -Force | Out-Null

Write-Host "Starting Phase C v14 non-combat cadence training..." -ForegroundColor Cyan
Write-Host "Target: 2.0 +/- 0.3 bombs, rolling WR >= 70%." -ForegroundColor Cyan
Write-Host "Non-combat win penalty: 0 cycles=-10, 1 cycle=-5, 2+ cycles=0." -ForegroundColor Cyan

& .\venv\Scripts\python.exe -u train_actor_critic.py `
    --phase C `
    --episodes 2000 `
    --board_size 9 `
    --num_agents 2 `
    --checkpoint_dir checkpoints_v14 `
    --resume_from checkpoints_v13\best_phaseC_target2.pt `
    --sudden_death `
    --wall_destroy_bonus 0.02 `
    --tactical_bomb_bonus 0.05 `
    --safe_escape_bonus 0.15 `
    --offensive_safe_escape_bonus 0.40 `
    --opponent_kill_bonus 1.00 `
    --generic_cycle_budget 2 `
    --target_bombs 2.0 `
    --target_bomb_tolerance 0.3 `
    --target_min_win_rate 70.0 `
    --snapshot_every 200 `
    --log_every 50

$trainingExitCode = $LASTEXITCODE
Write-Host "Training finished with exit code $trainingExitCode." -ForegroundColor $(if ($trainingExitCode -eq 0) { "Green" } else { "Red" })
Write-Host "Leave this window open so the terminal output remains available for review." -ForegroundColor Yellow
