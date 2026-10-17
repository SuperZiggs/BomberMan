# D1 DONE / REVIEWED

Output: D:/fpt/subject/BomberMan/runs/ppo-full-20261015-220313
Both unified training session34324 and review/package sessions have exited0. No training process remains from this task.

P4 resumed from episode36/Adam144, preserving full state and original deadlines. U4 ten completed files stayed byte-identical. Both arms finished200episodes/800Adam; best200. Pilot elapsed106.89min including pause.

Final400/model, boards37000–37199 both corners:
- B Active37.50%, suicide0.75%, draw53.50%.
- U4 Active37.00%, suicide1.00%, draw54.25%.
- P4 Active32.00%, suicide1.00%, draw58.75%.
- P4-B=-5.50pp CI95[-11.25;0.50]; P4-U4=-5.00pp CI95[-11.00;1.00]. Both CIs include0; no statistically established difference from0. All three replacement gates failed; keep B.
- Both arms4epochs throughout, no KL early-stop. Ratio-outside-range fraction U4=0.0442%, P4=0.0507%; max postKL~0.005. D2 early-stop trigger absent.

Independent review PASSED: source/B/checkpoint/locked-model hashes, original raw ZIP59files CRC/bytes,384skill tapes/8794frames, full-state/context/history/Adam/scheduler/RNG presence, validation selection, raw aggregate/corner metrics and paired bootstrap.

Reports, guide/tracker/README/RESULTS/IMPROVEMENTS updated. Original raw-send ZIP unchanged. Reviewed ZIP89files CRC/byte verified; package_review.json records hashes. decision_report.md has been prepared for opening in Codex.

No new experiment or model replacement. One train seed; no fresh AC-one-update training control. Do not rerun the completed trial or retune on test37000. Future experiment needs separate authorization/protocol and unused seeds.
