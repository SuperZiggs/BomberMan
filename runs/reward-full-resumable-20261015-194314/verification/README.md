# Evidence for the completed R1 rerun

The raw `-send.zip` remains unchanged. This review archive adds independent analysis, report, run console, resume proof and document snapshots.

- `all-tests.log`: 209 regression tests. Two additional deadline/evaluation guards and one writer-lock test bring coverage to 212 distinct test methods. The 49-test inference/visualization check overlaps those methods.
- `test_training_state.py` in `source/` includes uninterrupted versus paused/resumed CPU and CUDA tests in new Python processes, including Adam, scheduler, model, action tape and RNG equality.
- CPU runner smoke paused half-wall at episode 10, then resumed it in a new process through episode 20; the completed control files were unchanged. CUDA runner smoke completed 20 episodes per arm.
- Smoke JSON records source changes made after that smoke finished. They are historical checks of their archived sources, not claims that the smoke source was identical to the final frozen trial. Later loader/lock guards were tested separately.
- `analysis_results.json` verifies the actual full-run states at episode 200 against final/best weights, 200 Adam steps, scheduler, histories, committed log context, runtime/source provenance and all RNG groups.
- Trial completed fresh, with `resumed=false`; no simulated interruption was introduced into the full experiment.

Resume is Phase C only and requires the same configuration, source, runtime and device. The runner retains its original deadline. This completed trial needs no resume. Use `best_phaseC.pt`/`final_phaseC.pt` for inference; `training_state.pt` for training recovery.

Independent review can be rerun from the repository root with:

```powershell
& 'D:\anaconda\envs\torch\python.exe' -X utf8 .\runs\reward-full-resumable-20261015-194314\review_results.py .\runs\reward-full-resumable-20261015-194314
```
