$ErrorActionPreference = "Stop"

$Python = Join-Path $PSScriptRoot "venv\Scripts\python.exe"
$Checkpoint = Join-Path $PSScriptRoot "runs\pressure-C-next\final_phaseC.pt"
$OutputDirectory = Join-Path $PSScriptRoot "runs\phaseC-sd-pilot-v2"

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
    --episodes 300 `
    --board_size 9 `
    --num_agents 2 `
    --max_steps 500 `
    --lr 0.0001 `
    --wall_destroy_bonus 0.15 `
    --seed 2026101402 `
    --resume_from $Checkpoint `
    --validation_every 150 `
    --validation_episodes 100 `
    --validation_seed 23000 `
    --sudden_death_probability 0.5 `
    --sudden_death_start 375 `
    --sudden_death_interval 25 `
    --sudden_death_warning_steps 40 `
    --checkpoint_dir $OutputDirectory

exit $LASTEXITCODE
