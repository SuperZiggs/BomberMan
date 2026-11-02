# S4-11 CNN distillation

Experiment specification: `docs/superpowers/specs/2026-10-31-s4-map11-cnn-distillation-design.md`.
The launcher creates a new output and runs pilot → collect → train → validate → test, stopping
at the registered pilot or validation guards. Test is opened at most once. A stale or foreign
lock requires diagnosis; do not delete it while any associated process may still be alive.
Collection resume is only for missing, verified shards, after confirming all old processes have stopped:

```cmd
E:\Tools\python.exe run_s4_map11_distill.py collect --output E:\Bomberman\runs\s4-map11-cnn-20261031-v1 --workers 5 --resume
```

An interrupted pilot, train, validation or test is not automatically replayed. In particular,
never rerun test when `test_opened.json` exists. Full results and a model can only be assessed
after validation and the pre-registered one-time test; smoke data are not research results.

The completed run's frozen source bundle is under `runs/s4-map11-cnn-20261031-v1/source/`.
The repository root also includes the user's pre-existing GUI changes, so root `visualize.py`
and `test_visualize.py` differ from the registered snapshot. Root-based provenance checks will
reject this difference; use the frozen source bundle and recorded hashes for provenance. Paths in
the registration and selection JSON describe the original Windows machine, not a portable checkout.
