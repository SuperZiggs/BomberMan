# Graph Report - BomberMan  (2026-10-16)

## Corpus Check
- Large corpus: 1736 files · ~22,970,979 words. Semantic extraction will be expensive (many Claude tokens). Consider running on a subfolder.

## Summary
- 1096 nodes · 3575 edges · 41 communities (32 shown, 9 thin omitted)
- Extraction: 91% EXTRACTED · 9% INFERRED · 0% AMBIGUOUS · INFERRED: 308 edges (avg confidence: 0.84)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- audit_t2.py / run_combat_move_pilot.py
- benchmark_t3.py / run_t3_curriculum.py
- train_actor_critic.py / torch
- numpy / Bomb
- CurriculumChecks / run()
- visualize.py / test_visualize.py
- test_combat.py / rule_based_bot.py
- TinyNet / tape()
- Plan and Tracker Bomberman / Improvements Log
- arena() / CombatChecks
- BombermanFFAEnv / .reset()
- get_action_mask() / test_audit_t2.py
- train() / ActorCriticNet
- gae_advantages() / .observe()
- test_tactical_teacher.py / get_two_bomb_plan()
- run_sudden_death_experiment. / evaluate_case()
- pilot_two_bomb.py / make_ac_fn()
- SnapshotOpponent / replay_game()
- StateChecks / .continuation()
- result() / PilotChecks
- time / compare_branches.py
- evaluate.py / make_rule_based_fn()
- AdvantageChecks / normalize_policy_advantages(
- .observe() / D0TrainingChecks
- compare_new_checkpoints.py / native_evaluation()
- .run_mocked() / ProtocolChecks
- ProtocolChecks / .run_mocked()
- certify_combat_move() / CombatMoveChecks
- reference_inputs() / CompleteRunChecks
- Comparison Loc_update b00eca / Comparison Loc_update
- SuddenDeathEnv / .test_off_is_exact_current_e
- evaluate_seeded() / observe_contact()
- .run_mocked() / ProtocolChecks
- RewardChecks / .parts()
- T2 teacher and CNN combo aud / T3.2 combat curriculum with 
- RewardIsolationChecks / .test_seeded_evaluation_reus
- _eval_vs_rulebot() / RuleBasedOpponent
- Stage A rule-based bot and s / Stage B Q-learning linear FA
- two_bomb full run / Research directions doc
- _combat_reward() / .test_passive_win_has_no_bon
- .test_observation_and_rollou / AuditChecks

## God Nodes (most connected - your core abstractions)
1. `BombermanFFAEnv` - 83 edges
2. `get_action_mask()` - 80 edges
3. `train()` - 76 edges
4. `arena()` - 63 edges
5. `CombatChecks` - 54 edges
6. `put_bomb()` - 50 edges
7. `Bomb` - 46 edges
8. `dump()` - 43 edges
9. `digest()` - 39 edges
10. `make_env()` - 32 edges

## Surprising Connections (you probably didn't know these)
- `main()` --indirect_call--> `make_ac_fn()`  [INFERRED]
  _comparison/compare_branches.py → evaluate.py
- `main()` --calls--> `BombermanFFAEnv`  [EXTRACTED]
  _comparison/compare_branches.py → bomberman_env.py
- `main()` --calls--> `evaluate_seeded()`  [EXTRACTED]
  _comparison/compare_branches.py → evaluate.py
- `main()` --calls--> `ActorCriticNet`  [EXTRACTED]
  _comparison/compare_branches.py → train_actor_critic.py
- `main()` --calls--> `smart_load_weights()`  [EXTRACTED]
  _comparison/compare_branches.py → train_actor_critic.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Loc_update reproduced risks** — _comparison_comparison_loc_update_b00eca8_ckpt_misload, _comparison_comparison_loc_update_b00eca8_kill_bug, _comparison_comparison_loc_update_b00eca8_forecast_bug, _comparison_comparison_loc_update_b00eca8_rolling_train [EXTRACTED 1.00]
- **T3 benchmark to curriculum pipeline** — benchmark_t3, run_t3_curriculum, audit_t2 [INFERRED 0.65]
- **T4 optimizer/reward pilots (all gate FAIL, keep B)** — advantage_guide_norm, gae_pilot_guide_pilot, ppo_pilot_guide_d1, r0_guide_r0 [INFERRED 0.85]

## Communities (41 total, 9 thin omitted)

### Community 0 - "audit_t2.py / run_combat_move_pilot.py"
Cohesion: 0.05
Nodes (56): describe(), distribution(), observe(), play_game(), plot_results(), run(), validation_parity(), diagnostic_row() (+48 more)

### Community 1 - "benchmark_t3.py / run_t3_curriculum.py"
Cohesion: 0.06
Nodes (54): collect_candidates(), fraction(), inspect_state(), main(), plan_key(), selection(), summarize_states(), viable() (+46 more)

### Community 2 - "train_actor_critic.py / torch"
Cohesion: 0.08
Nodes (13): advantage_diagnostics, T4-A1 Advantage Guide, Actor advantage normalization (gate FAIL, keep B), _run(), GAESmokeGuards, D0SmokeGuards, same(), save_state() (+5 more)

### Community 3 - "numpy / Bomb"
Cohesion: 0.06
Nodes (18): Bomb, certify_fixtures(), make_fixture(), masked_surviving_path(), parity_replay(), surviving_path(), ablation_bomb_shyness(), plot_curves() (+10 more)

### Community 4 - "CurriculumChecks / run()"
Cohesion: 0.06
Nodes (12): Curriculum, evaluate_models(), full_pair_ready(), matched_budget_diagnostic(), run(), ScriptedOpponent, train_arm(), progress() (+4 more)

### Community 5 - "visualize.py / test_visualize.py"
Cohesion: 0.07
Nodes (9): check(), App, BoardView, capture_state(), GamePane, load_agent(), rolling_metrics(), TrainPane (+1 more)

### Community 6 - "test_combat.py / rule_based_bot.py"
Cohesion: 0.09
Nodes (15): _adjacent_to_opponent(), _adjacent_to_wood(), _bfs_safe_cell(), _bfs_toward_item(), _bfs_toward_opponent(), _bfs_toward_wood(), _bomb_escape_pressure(), _build_danger_map() (+7 more)

### Community 7 - "TinyNet / tape()"
Cohesion: 0.10
Nodes (12): scripted(), RolloutChecks, tape(), TinyNet, D1MathChecks, step(), D1TrainingChecks, adam() (+4 more)

### Community 8 - "Plan and Tracker Bomberman / Improvements Log"
Cohesion: 0.06
Nodes (39): T2 audit helpers.py, Escape certificate on clone, Combat MOVE Pilot Guide, local_combat_move attack teacher (gate FAIL), Control best400 confirmation (gate FAIL, keep B), Control Confirmation Guide, T3 approach audit / counter-bomb fixtures, T4-B0 GAE Correctness Guide (+31 more)

### Community 9 - "arena() / CombatChecks"
Cohesion: 0.14
Nodes (3): arena(), CombatChecks, put_bomb()

### Community 10 - "BombermanFFAEnv / .reset()"
Cohesion: 0.14
Nodes (6): Agent, BombermanFFAEnv, replay_seen_counterbombs(), eval_random_baseline(), run_match(), evaluate_rule_based()

### Community 11 - "get_action_mask() / test_audit_t2.py"
Cohesion: 0.16
Nodes (7): observed_teacher(), PlanTracker, rate(), summarize(), AuditChecks, combo_arena(), get_action_mask()

### Community 12 - "train() / ActorCriticNet"
Cohesion: 0.09
Nodes (9): after_episode(), observe_selection(), winning_step(), ActorCriticNet, load_opponent_pool(), save_snapshot(), _tensor_stats(), train() (+1 more)

### Community 13 - "gae_advantages() / .observe()"
Cohesion: 0.14
Nodes (5): GAEMathChecks, GAETrainingChecks, inputs(), mc_returns(), gae_advantages()

### Community 14 - "test_tactical_teacher.py / get_two_bomb_plan()"
Cohesion: 0.14
Nodes (6): _reachable_cells(), get_two_bomb_plan(), evaluate(), _learner_blockers(), _placement_routes(), TacticalChecks

### Community 15 - "run_sudden_death_experiment. / evaluate_case()"
Cohesion: 0.11
Nodes (13): comparisons(), digest(), dump(), evaluate_case(), cnn_mask(), __init__(), observed_match(), reset() (+5 more)

### Community 16 - "pilot_two_bomb.py / make_ac_fn()"
Cohesion: 0.17
Nodes (9): make_ac_fn(), complete_validation(), final_evaluation(), paired_active_bootstrap(), preflight(), run_pilot(), save_json(), digest() (+1 more)

### Community 17 - "SnapshotOpponent / replay_game()"
Cohesion: 0.15
Nodes (8): replay_game(), obs_to_tensor(), _print_report(), _render_demo_match(), _round_robin_eval(), SnapshotOpponent, _spawn_fairness_eval(), validate_candidate()

### Community 18 - "StateChecks / .continuation()"
Cohesion: 0.14
Nodes (3): StateChecks, episode(), cut_train()

### Community 19 - "result() / PilotChecks"
Cohesion: 0.17
Nodes (5): PilotChecks, evaluate(), ready_arm(), result(), evaluate()

### Community 20 - "time / compare_branches.py"
Cohesion: 0.19
Nodes (10): main(), main(), arena(), engine_probes(), load_module(), load_remote(), main(), put_bomb() (+2 more)

### Community 21 - "evaluate.py / make_rule_based_fn()"
Cohesion: 0.16
Nodes (6): eval_rulebots(), make_random_fn(), fn(), make_rule_based_fn(), RandomAgent, tournament()

### Community 22 - "AdvantageChecks / normalize_policy_advantages("
Cohesion: 0.16
Nodes (3): AdvantageChecks, normalize_policy_advantages(), obs_batch_to_tensor()

### Community 23 - ".observe() / D0TrainingChecks"
Cohesion: 0.16
Nodes (3): D0TrainingChecks, execute(), same()

### Community 24 - "compare_new_checkpoints.py / native_evaluation()"
Cohesion: 0.25
Nodes (9): check_observer(), common_policy(), fn(), load_net(), native_evaluation(), ObservedLocEnv, run_case(), summarize() (+1 more)

### Community 27 - "certify_combat_move() / CombatMoveChecks"
Cohesion: 0.21
Nodes (3): certify_combat_move(), finish(), CombatMoveChecks

### Community 29 - "Comparison Loc_update b00eca / Comparison Loc_update"
Cohesion: 0.18
Nodes (11): Checkpoint silent misload risk, Common benchmark same rules and opponents, Comparison Loc_update b00eca8, Forecast misses blast through destroyed wall, Loc_update engine double-counted kills, Rolling train insufficient for checkpoint selection, Bomb outcome diagnostics, Position choice and chained bombs (+3 more)

### Community 31 - "evaluate_seeded() / observe_contact()"
Cohesion: 0.20
Nodes (6): evaluate_seeded(), observed(), summarize(), observe_contact(), summarize_contact(), on_step()

### Community 34 - "T2 teacher and CNN combo aud / T3.2 combat curriculum with "
Cohesion: 0.22
Nodes (9): T2 guide, Future masks and physical plan clone test, Plan churn measurement, T2 teacher and CNN combo audit, T3.2 curriculum guide, full600 paired run, T3.2 combat curriculum with control, T3.1 guide (+1 more)

### Community 35 - "RewardIsolationChecks / .test_seeded_evaluation_reus"
Cohesion: 0.29
Nodes (5): match(), policy(), evaluate(), RewardIsolationChecks, observed()

### Community 37 - "Stage A rule-based bot and s / Stage B Q-learning linear FA"
Cohesion: 0.60
Nodes (5): Experiment results, Gate checks, Stage A rule-based bot and sanity check, Stage B Q-learning linear FA, Stage C Actor-Critic 1v1 vs rule bot

### Community 38 - "two_bomb full run / Research directions doc"
Cohesion: 0.67
Nodes (3): Research directions doc, Next research ordering after pilot, two_bomb full run

## Knowledge Gaps
- **29 isolated node(s):** `advantage_diagnostics`, `Escape certificate on clone`, `Control best400 confirmation (gate FAIL, keep B)`, `Two masks bootstrap_allowed / continue_segment`, `Phase C Actor-Critic CNN vs RuleBot` (+24 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 317 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **9 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `train()` connect `train() / ActorCriticNet` to `audit_t2.py / run_combat_move_pilot.py`, `benchmark_t3.py / run_t3_curriculum.py`, `train_actor_critic.py / torch`, `numpy / Bomb`, `CurriculumChecks / run()`, `visualize.py / test_visualize.py`, `test_combat.py / rule_based_bot.py`, `TinyNet / tape()`, `arena() / CombatChecks`, `BombermanFFAEnv / .reset()`, `get_action_mask() / test_audit_t2.py`, `gae_advantages() / .observe()`, `test_tactical_teacher.py / get_two_bomb_plan()`, `pilot_two_bomb.py / make_ac_fn()`, `SnapshotOpponent / replay_game()`, `evaluate.py / make_rule_based_fn()`, `AdvantageChecks / normalize_policy_advantages(`, `.observe() / D0TrainingChecks`, `certify_combat_move() / CombatMoveChecks`, `evaluate_seeded() / observe_contact()`, `_eval_vs_rulebot() / RuleBasedOpponent`?**
  _High betweenness centrality (0.115) - this node is a cross-community bridge._
- **Are the 24 inferred relationships involving `BombermanFFAEnv` (e.g. with `play_game()` and `replay_game()`) actually correct?**
  _`BombermanFFAEnv` has 24 INFERRED edges - model-reasoned connections that need verification._
- **What connects `advantage_diagnostics`, `Escape certificate on clone`, `Control best400 confirmation (gate FAIL, keep B)` to the rest of the system?**
  _29 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `audit_t2.py / run_combat_move_pilot.py` be split into smaller, more focused modules?**
  _Cohesion score 0.05307346326836582 - nodes in this community are weakly interconnected._
- **Why does `BombermanFFAEnv` connect `BombermanFFAEnv / .reset()` to `audit_t2.py / run_combat_move_pilot.py`, `benchmark_t3.py / run_t3_curriculum.py`, `train_actor_critic.py / torch`, `numpy / Bomb`, `_eval_vs_rulebot() / RuleBasedOpponent`, `visualize.py / test_visualize.py`, `test_combat.py / rule_based_bot.py`, `arena() / CombatChecks`, `get_action_mask() / test_audit_t2.py`, `train() / ActorCriticNet`, `pilot_two_bomb.py / make_ac_fn()`, `SnapshotOpponent / replay_game()`, `time / compare_branches.py`, `evaluate.py / make_rule_based_fn()`, `SuddenDeathEnv / .test_off_is_exact_current_e`, `evaluate_seeded() / observe_contact()`?**
  _High betweenness centrality (0.112) - this node is a cross-community bridge._
- **Are the 3 inferred relationships involving `get_action_mask()` (e.g. with `main()` and `.test_certificate_rejects_a_future_mask_block_and_invalid_state()`) actually correct?**
  _`get_action_mask()` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Should `benchmark_t3.py / run_t3_curriculum.py` be split into smaller, more focused modules?**
  _Cohesion score 0.05695970695970696 - nodes in this community are weakly interconnected._