# Workspace Notes

This directory is the active Python package inside the larger git repository at
`/home/shawheen/research_project`.

## What Belongs Here

- Simulator and model source code.
- Experiment scripts that import package modules directly.
- Small documentation files describing active experiments.

## What Is Generated

- `datasets/*.npz` - cached simulated trajectories.
- `checkpoints/**` - model weights, summaries, plots, and videos.
- `__pycache__/**` and `*.pyc` - Python bytecode caches.
- `*.mp4`, `*.png`, `*.pt` - visual/model artifacts.

## Current Experiment Families

- `simple_tcno_1p*` - time-conditioned one-particle collision models.
- `eval_simple_tcno_1p*` - rollout evaluations and visual diagnostics for TCNO
  checkpoints.
- `*_exact_*` - runs using exact one-particle wall reflection.
- `*_gaussian*`, `*_window*`, `*_spike*` - event-target variants.
- `history_*` and `rnn_residual*` - autoregressive/history baselines.

## Cleanup Policy

Source files are intentionally left in place because many scripts use direct
imports and relative paths. Keep larger generated artifacts under `datasets/`
or `checkpoints/`; do not mix fresh outputs into the package root.

If you want a deeper cleanup later, good next candidates are:

- untrack committed `__pycache__` files;
- move exploratory notebooks into a `notebooks/` directory;
- move top-level experiment launchers into a `scripts/` directory and update
  imports/paths in one pass;
- add a root-level README that points here as the active package workspace.
