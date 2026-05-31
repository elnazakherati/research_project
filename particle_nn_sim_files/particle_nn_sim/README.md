# particle_nn_sim

Research workspace for learning neural simulators of simple 2D particle motion.

The current active thread is a one-particle-in-a-box collision problem: generate
synthetic trajectories with exact wall reflections, train time-conditioned neural
models to predict state at queried times, and evaluate long rollouts against the
ground-truth simulator.

## Core Modules

- `simulator.py` - 2D circular-particle simulator with wall and pair collisions.
- `one_particle_data.py` - one-particle initial condition sampling and dataset
  construction.
- `data.py` - older two-particle dataset helpers.
- `time_conditioned_collision_model.py` - time-conditioned model that predicts
  `[x, y, vx, vy]` and optionally wall-event logits from initial state plus time.
- `models.py` - MLP, interaction-network, and residual-MLP baselines.
- `train.py` - shared standardization, dataset, and train/eval helpers.
- `one_particle_rollout.py` and `rollout_eval.py` - rollout and visualization
  utilities.

## Main Scripts

- `generate_tcno_stratified_dataset.py` - build cached one-particle NPZ datasets.
- `train_simple_tcno_1p.py` - train the simple time-conditioned one-particle
  collision model from a cached dataset.
- `run_eval_time_conditioned_collision_1p.py` - evaluate trained TCNO checkpoints
  on rollout metrics and visual diagnostics.
- `run_one_particle_rnn_residual_pipeline.py` - train/evaluate an RNN residual
  baseline.
- `run_eval_trainset_1p.py` and `run_eval_history_trainset_1p.py` - evaluate
  older one-particle baselines.
- `run_gt_clamp_vs_exact_1p.py` and `run_gt_perturbation_1p.py` - ground-truth
  simulator diagnostics.
- `plot_training_curves.py` and `plot_tcno_speed_diagnostics.py` - plotting
  helpers for saved runs.

## Generated Data And Outputs

- `datasets/` contains generated `.npz` trajectory datasets.
- `checkpoints/` contains model checkpoints, metrics, plots, and rollout videos.
- `__pycache__/`, `.pt`, `.png`, and `.mp4` files are generated artifacts.

The repository currently contains some tracked generated artifacts from earlier
experiments. Avoid moving or deleting them casually unless you want that cleanup
to appear as explicit git changes.

## Typical Workflow

Generate a dataset:

```bash
python generate_tcno_stratified_dataset.py --out datasets/simple_tcno_stratified_1p.npz
```

Train a time-conditioned one-particle model:

```bash
python train_simple_tcno_1p.py --dataset datasets/simple_tcno_stratified_1p.npz
```

Evaluate a trained checkpoint:

```bash
python run_eval_time_conditioned_collision_1p.py \
  --checkpoint checkpoints/simple_tcno_1p/model_tc_collision_1p.pt
```
