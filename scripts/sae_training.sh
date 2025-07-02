#!/usr/bin/env bash
set -euo pipefail

MODEL_CHECKPOINT="https://splat-nlp.nyc3.cdn.digitaloceanspaces.com/dataset_v2/model_ultra.pth"
DATA_CSV="https://splat-nlp.nyc3.cdn.digitaloceanspaces.com/dataset_v2/tokenized_data.csv"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
SAVE_DIR="sae_runs/run_${TIMESTAMP}"
mkdir -p "${SAVE_DIR}"

VOCAB_PATH="https://splat-nlp.nyc3.cdn.digitaloceanspaces.com/dataset_v2/vocab.json"
WEAPON_VOCAB_PATH="https://splat-nlp.nyc3.cdn.digitaloceanspaces.com/dataset_v2/weapon_vocab.json"

PRIMARY_EMBEDDING_DIM=32
PRIMARY_HIDDEN_DIM=512
PRIMARY_NUM_LAYERS=3
PRIMARY_NUM_HEADS=8
PRIMARY_NUM_INDUCING=32

EXPANSION_FACTOR=6
LR=1e-4
L1_COEFF=1e-4
TARGET_USAGE=7e-3
USAGE_COEFF=0.0
GRADIENT_CLIP_VAL=1.0
KL_WARMUP_STEPS=0
KL_PERIOD_STEPS=60000
KL_FLOOR=0.0
L1_WARMUP_STEPS=0
L1_START=1e-4
DEAD_NEURON_THRESHOLD=1e-6

EPOCHS=3
BUFFER_SIZE=100000
SAE_BATCH_SIZE=1024
STEPS_BEFORE_TRAIN=50000
SAE_TRAIN_STEPS=4
PRIMARY_DATA_FRACTION=0.005

RESAMPLE_WEIGHT=0.01
RESAMPLE_BIAS=-1.0
RESAMPLE_STEPS=(7000 14000 21000 28000)

DEVICE="cuda"
NUM_WORKERS=16
VERBOSE_FLAG="--verbose"

# --- Weights & Biases Config ---
# Set to your W&B username or team name. If you are already logged in, you can skip this.
# WANDB_ENTITY="your-wandb-entity"
WANDB_PROJECT="splatnlp-sae"
# Set to --wandb-log to enable, or "" to disable
WANDB_LOG_FLAG="--wandb-log"

CMD=(
  poetry run python3 -m splatnlp.monosemantic_sae.cli

  --model-checkpoint "${MODEL_CHECKPOINT}"
  --data-csv         "${DATA_CSV}"
  --save-dir         "${SAVE_DIR}"
  --vocab-path       "${VOCAB_PATH}"
  --weapon-vocab-path "${WEAPON_VOCAB_PATH}"

  --primary-embedding-dim "${PRIMARY_EMBEDDING_DIM}"
  --primary-hidden-dim     "${PRIMARY_HIDDEN_DIM}"
  --primary-num-layers     "${PRIMARY_NUM_LAYERS}"
  --primary-num-heads      "${PRIMARY_NUM_HEADS}"
  --primary-num-inducing   "${PRIMARY_NUM_INDUCING}"

  --epochs                 "${EPOCHS}"
  --expansion-factor       "${EXPANSION_FACTOR}"
  --lr                     "${LR}"
  --l1-coeff               "${L1_COEFF}"
  --target-usage           "${TARGET_USAGE}"
  --usage-coeff            "${USAGE_COEFF}"
  --gradient-clip-val      "${GRADIENT_CLIP_VAL}"
  --kl-warmup-steps        "${KL_WARMUP_STEPS}"
  --kl-period-steps        "${KL_PERIOD_STEPS}"
  --kl-floor               "${KL_FLOOR}"
  --l1-warmup-steps        "${L1_WARMUP_STEPS}"
  --l1-start               "${L1_START}"
  --dead-neuron-threshold  "${DEAD_NEURON_THRESHOLD}"

  --buffer-size            "${BUFFER_SIZE}"
  --sae-batch-size         "${SAE_BATCH_SIZE}"
  --steps-before-train     "${STEPS_BEFORE_TRAIN}"
  --sae-train-steps        "${SAE_TRAIN_STEPS}"
  --primary-data-fraction  "${PRIMARY_DATA_FRACTION}"

  --resample-weight        "${RESAMPLE_WEIGHT}"
  --resample-bias          "${RESAMPLE_BIAS}"
  --resample-steps         "${RESAMPLE_STEPS[@]}"

  --device                 "${DEVICE}"
  --num-workers            "${NUM_WORKERS}"
)

if [[ -n "${VERBOSE_FLAG}" ]]; then
  CMD+=("${VERBOSE_FLAG}")
fi

if [[ -n "${WANDB_LOG_FLAG}" ]]; then
  CMD+=("${WANDB_LOG_FLAG}")
  CMD+=(--wandb-project "${WANDB_PROJECT}")
  if [[ -n "${WANDB_ENTITY:-}" ]]; then
    CMD+=(--wandb-entity "${WANDB_ENTITY}")
  fi
fi

echo "Launching SAE training - output will be written to: ${SAVE_DIR}"
printf '» %q ' "${CMD[@]}"; echo
"${CMD[@]}"

echo "Training finished ✅  (artifacts in ${SAVE_DIR})"
