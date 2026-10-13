$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

$checkpointDirectory = Join-Path $PSScriptRoot "checkpoints_v16"
New-Item -ItemType Directory -Path $checkpointDirectory -Force | Out-Null

Write-Host "Starting Phase C v16 outcome-gated combat training..." -ForegroundColor Cyan
Write-Host "Pressure/offensive rewards are paid only after bomb outcome when the owner is alive and safe." -ForegroundColor Cyan
Write-Host "Target: WR >= 70%, kills >= 0.025, rolling self deaths <= 0.05." -ForegroundColor Cyan

& .\venv\Scripts\python.exe -u train_actor_critic.py `
    --phase C `
    --episodes 2000 `
    --board_size 9 `
    --num_agents 2 `
    --max_steps 300 `
    --checkpoint_dir checkpoints_v16 `
    --resume_from checkpoints_v15\best_phaseC_combat.pt `
    --sudden_death `
    --sudden_death_start 200 `
    --sudden_death_interval 20 `
    --wall_destroy_bonus 0.02 `
    --tactical_bomb_bonus 0.05 `
    --safe_escape_bonus 0.15 `
    --offensive_safe_escape_bonus 0.40 `
    --pressure_bonus 0.15 `
    --opponent_kill_bonus 4.00 `
    --noncombat_win_fixed_penalty 8.00 `
    --combat_min_win_rate 70.0 `
    --combat_safe_min_kills 0.025 `
    --combat_safe_max_self_deaths 0.05 `
    --generic_cycle_budget 2 `
    --target_min_win_rate 101.0 `
    --snapshot_every 200 `
    --log_every 50

$trainingExitCode = $LASTEXITCODE
Write-Host "Training finished with exit code $trainingExitCode." -ForegroundColor $(if ($trainingExitCode -eq 0) { "Green" } else { "Red" })
Write-Host "Leave this window open so the terminal output remains available for review." -ForegroundColor Yellow
