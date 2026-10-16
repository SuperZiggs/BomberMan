"""Test-only helpers: compare the merged trainer/evaluator with pre-merge source snapshots.

The merge of codex/phasec-sd-best-fallback made the pressure/sequence teacher and its loss
weights the default and added sudden-death metric keys. LEGACY_TRAIN selects the pre-merge
recipe; strip_sd drops the added keys so old and new outputs compare field by field.
"""
import train_actor_critic as trainer

LEGACY_TRAIN = dict(teacher_version='v1', lr=3e-4, entropy_coef=0.015, escape_loss_coef=0.5,
                    combat_loss_coef=0.5, navigation_loss_coef=0.2, late_game_attack_multiplier=1.0)
SD_KEYS = frozenset({
    'sudden_death', 'sudden_death_start', 'sudden_death_interval', 'sudden_death_warning_steps',
    'sudden_death_enabled', 'sudden_death_death', 'sudden_death_win', 'sudden_death_deaths',
    'sudden_death_wins', 'sudden_death_win_rate', 'sudden_death_death_rate', 'passive_win',
    'passive_wins', 'passive_win_rate', 'active_wins', 'draws', 'ring_safety_loss'})


def legacy_train(**kwargs):
    return trainer.train(**{**LEGACY_TRAIN, **kwargs})


def strip_sd(value):
    if isinstance(value, dict):
        return {k: strip_sd(v) for k, v in value.items() if k not in SD_KEYS}
    if isinstance(value, (list, tuple)):
        return type(value)(strip_sd(v) for v in value)
    return value
