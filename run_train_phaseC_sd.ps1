$ErrorActionPreference = "Stop"

$Python = Join-Path $PSScriptRoot "venv\Scripts\python.exe"
$Checkpoint = Join-Path $PSScriptRoot "runs\pressure-C-next\final_phaseC.pt"
$OutputDirectory = Join-Path $PSScriptRoot "runs\phaseC-sd-pilot"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python interpreter not found: $Python"
}
if (-not (Test-Path -LiteralPath $Checkpoint)) {
    throw "Warm-start checkpoint not found: $Checkpoint"
}

& $Python (Join-Path $PSScriptRoot "train_actor_critic.py") `
    --phase C `
    --episodes 500 `
    --board_size 9 `
    --num_agents 2 `
    --max_steps 500 `
    --lr 0.0001 `
    --wall_destroy_bonus 0.15 `
    --seed 2026101401 `
    --resume_from $Checkpoint `
    --validation_every 250 `
    --validation_episodes 100 `
    --validation_seed 20000 `
    --sudden_death_probability 0.75 `
    --sudden_death_start 350 `
    --sudden_death_interval 20 `
    --sudden_death_warning_steps 40 `
    --checkpoint_dir $OutputDirectory

exit $LASTEXITCODE
