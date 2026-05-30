# particle_nn_sim

Research code for learning neural simulators of 2D particle motion.  The older
part of the repo targets two-particle elastic collisions; the current active
line of work is a one-particle box problem with exact wall reflections.

Most current code lives in `particle_nn_sim/`. The current one-particle
experiment artifacts are stored under `particle_nn_sim/datasets/`,
`particle_nn_sim/checkpoints/`, `particle_nn_sim/results/`, and
`particle_nn_sim/wandb/`. Some older generated artifacts also exist at the
repository top level.

## What Is Being Modeled

The main active task is:

- sample one-particle initial conditions in a square box;
- roll them out with exact wall reflections using `ParticleSim2D`;
- train neural models to predict particle state at queried times;
- evaluate long-horizon rollout quality against the ground-truth simulator.

The one-particle state is usually `[x, y, vx, vy]`. The best-performing recent
models for long horizons predict only position `[x, y]`; velocity is either not
predicted or handled by a separate diagnostic/auxiliary head.

## Repository Layout

- `particle_nn_sim/simulator.py`: 2D particle simulator.
- `particle_nn_sim/one_particle_data.py`: one-particle initial-condition and
  trajectory generation helpers.
- `particle_nn_sim/data.py`: older two-particle data helpers.
- `particle_nn_sim/models.py`: older MLP, interaction-network, and residual-MLP
  baselines.
- `particle_nn_sim/time_conditioned_collision_model.py`: current
  time-conditioned model used by the TCNO experiments.
- `particle_nn_sim/train.py`: shared standardization and training utilities.
- `particle_nn_sim/generate_tcno_stratified_dataset.py`: builds cached
  stratified one-particle datasets.
- `particle_nn_sim/train_simple_tcno_1p.py`: trains the current
  time-conditioned one-particle model from a cached NPZ dataset.
- `particle_nn_sim/run_eval_time_conditioned_collision_1p.py`: evaluates TCNO
  checkpoints on rollout metrics and visual diagnostics.
- `particle_nn_sim/run_one_particle_rnn_residual_pipeline.py`: GRU/LSTM residual
  baseline for autoregressive one-step residual prediction.
- `particle_nn_sim/checkpoints/`: model checkpoints and evaluation summaries.
- `particle_nn_sim/results/`: overlay and long-horizon visual summaries.

## Model Families

### TCNO: time-conditioned neural operator

Implemented in `particle_nn_sim/time_conditioned_collision_model.py` and trained
with `particle_nn_sim/train_simple_tcno_1p.py`.

The model takes an initial state `s0 = [x0, y0, vx0, vy0]` and a query time `t`,
then predicts the state at that time. It uses a shared MLP trunk, configurable
time encoding, and optional output heads:

- `output_mode=state`: predicts `[x, y, vx, vy]`.
- `output_mode=position`: predicts `[x, y]` only.
- `use_event_head=true`: also predicts wall-collision/event logits.
- `use_endpoint_velocity_head=true`: adds an auxiliary endpoint velocity head.
- `enforce_t0_anchor=true`: reparameterizes the output as `s0 + t * f(s0,t)`,
  so the prediction is exact at `t=0`.

Important TCNO experiment families:

- `A`, `C`, `D`, `E`, `F`, `G`: early full-state TCNOs comparing raw,
  Fourier, low-frequency Fourier, and event-target choices.
- `H`, `I`, `J`: full-state TCNOs without useful event supervision, including
  no-event-head variants and deeper raw-time trunks.
- `K`: position-only loss variants that still output state or add sparse
  velocity supervision.
- `L`: clean position-only TCNOs. These are the best models for position
  prediction and long-horizon visual rollouts.
- `M`: position TCNO with an auxiliary endpoint velocity head. Better when
  endpoint velocity estimates matter, but not the best pure-position model.

### Autoregressive residual RNN

Implemented in `particle_nn_sim/run_one_particle_rnn_residual_pipeline.py`.

This GRU/LSTM baseline predicts one-step residuals relative to free flight. It
uses short sequence windows, optional collision rebalancing, and then rolls out
autoregressively. It is useful as a recurrent baseline, but the strongest saved
results in this workspace are from the direct time-conditioned TCNO models.

### Older two-particle baselines

Implemented in `particle_nn_sim/models.py` and older scripts:

- `MLP`: direct feed-forward baseline.
- `InteractionNet2`: tiny two-particle interaction network.
- `ResMLP`: residual MLP for dynamics corrections.

These are kept for historical comparison and for the older two-particle setup.
They are not the current best models for the one-particle exact-wall problem.

## Best Saved Models

The most useful checkpoints currently in the workspace are:

| Use case | Checkpoint | Why it matters |
| --- | --- | --- |
| Best 10,000-step position model | `particle_nn_sim/checkpoints/simple_tcno_1p_exact_L_base_10000steps_10x_raw_gaussian_noeventhead_w256_d5_poshead_b65536_wandb_target1e-6_gpu5/model_tc_collision_1p.pt` | Best long-horizon L-base run. Position-only TCNO trained on the 10k-step 10x dataset. Best validation position MSE: `5.06e-4`; corrected 10k overlay mean position error: about `0.00719`. |
| Best 2,000-step full-test model | `particle_nn_sim/checkpoints/simple_tcno_1p_exact_L_2000steps_10x_raw_gaussian_noeventhead_w256_d5_poshead_b65536_wandb_target1e-6/model_tc_collision_1p.pt` | Strong full-test result on the 2k-step 10x dataset. Full test position MSE with anchor-fix eval: `7.80e-7`; overlay mean position error: about `9e-4` to `1e-3`. |
| Best 1,000-step 10x position model | `particle_nn_sim/checkpoints/simple_tcno_1p_exact_L10x_raw_gaussian_noeventhead_w256_d5_poshead_b65536_wandb_target1e-6_retry/model_tc_collision_1p.pt` | Strong 1k-step position-only TCNO. Best validation position MSE: `6.92e-6`; 20-episode test position MSE: `9.38e-6`. |
| Best endpoint-velocity diagnostic model | `particle_nn_sim/checkpoints/simple_tcno_1p_exact_M_endpointvel_1000steps_10x_raw_gaussian_noeventhead_w256_d5_anchorvel_b65536_target1e-6/model_tc_collision_1p.pt` | Adds an auxiliary endpoint velocity head. Best validation position MSE: `1.17e-4`; endpoint velocity MSE in a 3-episode diagnostic: about `5.02e-5`. |
| Best older full-state TCNO | `particle_nn_sim/checkpoints/simple_tcno_1p_exact_J_raw_gaussian_noeventhead_w256_d5_full_b65536/model_tc_collision_1p.pt` | Best of the deeper full-state no-event-head runs. 20-episode 1k rollout: state MSE `0.00152`, position MSE `1.68e-4`, velocity MSE `0.00287`. |

In short: use the **L-family position-only models** for position accuracy and
long rollouts. Use **M** only when endpoint velocity is part of the question.
Use **J/H/I/F/G** for historical full-state comparisons.

## Current Best Training Recipe

The L-family recipe is the cleanest current approach:

1. Generate a fixed-speed, stratified, exact-wall dataset.
2. Train a raw-time TCNO with width `256`, depth `5`.
3. Remove the event head.
4. Use `output_mode=position` and `state_loss_mode=position`.
5. Use large batches (`65536`) and a plateau LR scheduler.
6. For very large datasets, sample a fixed number of train points per epoch
   instead of shuffling the entire materialized training set.

The successful 10k L-base run used:

```bash
cd particle_nn_sim

python train_simple_tcno_1p.py \
  --dataset datasets/simple_tcno_stratified_1p_10000steps_10x.npz \
  --out-dir checkpoints/simple_tcno_1p_exact_L_base_10000steps_10x_raw_gaussian_noeventhead_w256_d5_poshead_b65536_wandb_target1e-6_gpu5 \
  --epochs 20000 \
  --batch-size 65536 \
  --samples-per-epoch 8388608 \
  --eval-batch-size 65536 \
  --lr 1e-3 \
  --weight-decay 1e-6 \
  --output-mode position \
  --state-loss-mode position \
  --use-event-head false \
  --event-loss-weight 0.0 \
  --trunk-width 256 \
  --trunk-depth 5 \
  --time-encoding-mode raw \
  --num-frequencies 4 \
  --lr-scheduler plateau \
  --lr-plateau-factor 0.5 \
  --lr-plateau-patience 25 \
  --lr-plateau-min 1e-6 \
  --target-val-mse 1e-6 \
  --print-every 25
```

For that run, the dataset had `16384 * 10001 = 163856384` materialized train
samples. A full `torch.randperm` over all samples OOM'd, so the successful run
used `--samples-per-epoch 8388608`, which is `2^23`, or roughly `512`
timepoints per training episode per epoch.

## Dataset Generation

Example 10k-step 10x dataset:

```bash
cd particle_nn_sim

python generate_tcno_stratified_dataset.py \
  --out datasets/simple_tcno_stratified_1p_10000steps_10x.npz \
  --steps 10000 \
  --episodes-per-bucket 160 \
  --fixed-speed 0.5
```

This produces:

- `20480` episodes;
- `10000` steps per episode plus the initial frame;
- `dt = 0.01`;
- train/val/test split of `16384/2048/2048`.

## Evaluation

Evaluate a checkpoint:

```bash
cd particle_nn_sim

python run_eval_time_conditioned_collision_1p.py \
  --checkpoint checkpoints/simple_tcno_1p_exact_L10x_raw_gaussian_noeventhead_w256_d5_poshead_b65536_wandb_target1e-6_retry/model_tc_collision_1p.pt \
  --dataset datasets/simple_tcno_stratified_1p_1000steps_10x.npz \
  --split test \
  --num-episodes 20 \
  --rollout-steps 1000
```

For position-only models, prefer evaluation summaries and overlays that use the
current anchor-fix path. Older 10k overlays before the correction can look much
worse even for the same checkpoint.

## Notes And Gotchas

- Validation MSE and long-rollout error are not interchangeable. For model
  selection, compare runs on the same dataset length and same evaluator.
- Full-state models can have reasonable state MSE but worse position rollout
  behavior than position-only models.
- The event head did not help the best one-particle position models. The recent
  L-family checkpoints remove it.
- Position-only checkpoints naturally report `NaN` for velocity and event
  metrics.
- Many generated artifacts are already present in this workspace. Avoid deleting
  or moving `datasets/`, `checkpoints/`, `results/`, or `wandb/` unless the
  cleanup is intentional.
