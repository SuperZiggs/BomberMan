$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

$checkpointDirectory = Join-Path $PSScriptRoot "checkpoints_v15"
New-Item -ItemType Directory -Path $checkpointDirectory -Force | Out-Null

Write-Host "Starting Phase C v15 combat-first training..." -ForegroundColor Cyan
Write-Host "Max steps stays at 300; Sudden Death stays aligned with evaluation." -ForegroundColor Cyan
Write-Host "Non-combat wins are capped at +2; safe pressure and direct kills are favored." -ForegroundColor Cyan

& .\venv\Scripts\python.exe -u train_actor_critic.py `
    --phase C `
    --episodes 3000 `
    --board_size 9 `
    --num_agents 2 `
    --max_steps 300 `
    --checkpoint_dir checkpoints_v15 `
    --resume_from checkpoints_v14\snapshot_phaseC_ep001200.pt `
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
    --generic_cycle_budget 2 `
    --target_min_win_rate 101.0 `
    --snapshot_every 200 `
    --log_every 50

$trainingExitCode = $LASTEXITCODE
Write-Host "Training finished with exit code $trainingExitCode." -ForegroundColor $(if ($trainingExitCode -eq 0) { "Green" } else { "Red" })
Write-Host "Leave this window open so the terminal output remains available for review." -ForegroundColor Yellow
