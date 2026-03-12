#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
PIPELINE_PY="${PIPELINE_PY:-/home/lnazaa/research_project/particle_nn_sim_files/run_one_particle_pipeline.py}"
EVAL_PY="${EVAL_PY:-/home/lnazaa/research_project/particle_nn_sim_files/run_one_particle_eval_only.py}"

OUT_ROOT="${OUT_ROOT:-/home/lnazaa/checkpoints/seed_sweep}"
TRAIN_SEEDS="${TRAIN_SEEDS:-0 1 2 3 4}"
EVAL_SEEDS="${EVAL_SEEDS:-123 456 789}"

NUM_ROLLOUTS="${NUM_ROLLOUTS:-100}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-2000}"
SPEED_MAX="${SPEED_MAX:-0.7}"
DIV_THRESHOLD="${DIV_THRESHOLD:-0.3}"

# Base training config (matches your current best recipe)
EPISODES="${EPISODES:-1000}"
STEPS="${STEPS:-500}"
EPOCHS="${EPOCHS:-60}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
LR="${LR:-0.0015}"
HORIZON="${HORIZON:-20}"
COLLISION_WEIGHT="${COLLISION_WEIGHT:-2}"
TARGET_COLLISION_FRAC="${TARGET_COLLISION_FRAC:-0.15}"
TRAIN_ROLLOUT_STEPS="${TRAIN_ROLLOUT_STEPS:-500}"
USE_WANDB="${USE_WANDB:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-particle-nn-sim}"

MODE="${1:-all}"  # train | eval | rank | all

run_train() {
  mkdir -p "${OUT_ROOT}"
  for s in ${TRAIN_SEEDS}; do
    echo "=== TRAIN seed ${s} ==="
    ${PYTHON_BIN} "${PIPELINE_PY}" \
      --episodes "${EPISODES}" \
      --steps "${STEPS}" \
      --epochs "${EPOCHS}" \
      --batch-size "${BATCH_SIZE}" \
      --lr "${LR}" \
      --multistep-horizon "${HORIZON}" \
      --collision-weight "${COLLISION_WEIGHT}" \
      --target-collision-frac "${TARGET_COLLISION_FRAC}" \
      --rollout-steps "${TRAIN_ROLLOUT_STEPS}" \
      --divergence-threshold "${DIV_THRESHOLD}" \
      --use-wandb "${USE_WANDB}" \
      --wandb-project "${WANDB_PROJECT}" \
      --wandb-run-name "one_particle_seed_sweep_s${s}" \
      --seed "${s}" \
      --out-dir "${OUT_ROOT}/s${s}"
  done
}

run_eval() {
  for s in ${TRAIN_SEEDS}; do
    for es in ${EVAL_SEEDS}; do
      echo "=== EVAL model seed s${s} on eval seed ${es} ==="
      ${PYTHON_BIN} "${EVAL_PY}" \
        --ckpt "${OUT_ROOT}/s${s}/model_1p_resmlp.pt" \
        --num-rollouts "${NUM_ROLLOUTS}" \
        --rollout-steps "${ROLLOUT_STEPS}" \
        --speed-max "${SPEED_MAX}" \
        --divergence-threshold "${DIV_THRESHOLD}" \
        --no-render true \
        --seed "${es}" \
        --out-dir "${OUT_ROOT}/s${s}/eval_seed${es}"
    done
  done
}

run_rank() {
  ${PYTHON_BIN} - <<'PY'
import glob
import json
import re
from collections import defaultdict

paths = glob.glob('/home/lnazaa/checkpoints/seed_sweep/s*/eval_seed*/eval_summary.json')
rows = defaultdict(list)

for p in paths:
    m = re.search(r'/seed_sweep/(s\d+)/eval_seed\d+/eval_summary\.json$', p)
    if not m:
        continue
    sid = m.group(1)
    d = json.load(open(p, 'r', encoding='utf-8'))
    rows[sid].append(d)

if not rows:
    print('No eval_summary.json files found. Run eval first.')
    raise SystemExit(1)

scored = []
for sid, ds in rows.items():
    n = len(ds)
    scored.append({
        'seed': sid,
        'n_eval_runs': n,
        'avg_ttf_p10': sum(x['ttf_p10'] for x in ds) / n,
        'avg_ttf_median': sum(x['ttf_median'] for x in ds) / n,
        'avg_ttf_mean': sum(x['ttf_mean'] for x in ds) / n,
        'avg_divergence_rate': sum(x['divergence_rate'] for x in ds) / n,
        'avg_final_err_p90': sum(x['final_err_p90'] for x in ds) / n,
    })

scored.sort(
    key=lambda r: (r['avg_ttf_p10'], r['avg_ttf_median'], -r['avg_divergence_rate']),
    reverse=True,
)

print('Ranking (best first):')
for r in scored:
    print(
        f"{r['seed']} | "
        f"n={r['n_eval_runs']} | "
        f"ttf_p10={r['avg_ttf_p10']:.2f} | "
        f"ttf_median={r['avg_ttf_median']:.2f} | "
        f"ttf_mean={r['avg_ttf_mean']:.2f} | "
        f"div_rate={r['avg_divergence_rate']:.3f} | "
        f"final_err_p90={r['avg_final_err_p90']:.3f}"
    )
PY
}

case "${MODE}" in
  train)
    run_train
    ;;
  eval)
    run_eval
    ;;
  rank)
    run_rank
    ;;
  all)
    run_train
    run_eval
    run_rank
    ;;
  *)
    echo "Usage: $0 [train|eval|rank|all]"
    exit 2
    ;;
esac
