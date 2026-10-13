$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

$checkpointDirectory = Join-Path $PSScriptRoot "checkpoints_v12"
New-Item -ItemType Directory -Path $checkpointDirectory -Force | Out-Null

Write-Host "Starting Phase C v12 training..." -ForegroundColor Cyan

& .\venv\Scripts\python.exe -u train_actor_critic.py `
    --phase C `
    --episodes 2000 `
    --board_size 9 `
    --num_agents 2 `
    --checkpoint_dir checkpoints_v12 `
    --resume_from checkpoints_v10\best_phaseC.pt `
    --sudden_death `
    --wall_destroy_bonus 0.02 `
    --tactical_bomb_bonus 0.05 `
    --safe_escape_bonus 0.15 `
    --offensive_safe_escape_bonus 0.40 `
    --opponent_kill_bonus 1.00 `
    --generic_cycle_budget 2 `
    --log_every 50

$trainingExitCode = $LASTEXITCODE
Write-Host "Training finished with exit code $trainingExitCode." -ForegroundColor $(if ($trainingExitCode -eq 0) { "Green" } else { "Red" })
Write-Host "Leave this window open so the terminal output remains available for review." -ForegroundColor Yellow
