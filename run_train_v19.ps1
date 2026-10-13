$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

$checkpointDirectory = Join-Path $PSScriptRoot "checkpoints_v19"
New-Item -ItemType Directory -Path $checkpointDirectory -Force | Out-Null

Write-Host "Starting Phase C v19 opponent/sudden-death safety training..." -ForegroundColor Cyan
Write-Host "Reward and max steps are unchanged; threatened agents must escape and closing rings are temporal walls." -ForegroundColor Cyan
Write-Host "Target: WR >= 70%, bombs >= 7, kills >= 0.08, opponent deaths <= 40/400, sudden-death deaths <= 30/400." -ForegroundColor Cyan

& .\venv\Scripts\python.exe -u train_actor_critic.py `
    --phase C `
    --episodes 2000 `
    --board_size 9 `
    --num_agents 2 `
    --max_steps 300 `
    --checkpoint_dir checkpoints_v19 `
    --resume_from checkpoints_v18\best_phaseC.pt `
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
