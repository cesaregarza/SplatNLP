#!/usr/bin/env bash
set -euo pipefail

# Basic training wrapper for SetCompletionModel (supports optional DDP).

DATA_PATH="${DATA_PATH:-https://splat-nlp.nyc3.cdn.digitaloceanspaces.com/dataset_v2/tokenized_data.csv}"
VOCAB_PATH="${VOCAB_PATH:-https://splat-nlp.nyc3.cdn.digitaloceanspaces.com/dataset_v2/vocab.json}"
WEAPON_VOCAB_PATH="${WEAPON_VOCAB_PATH:-https://splat-nlp.nyc3.cdn.digitaloceanspaces.com/dataset_v2/weapon_vocab.json}"
OUTPUT_DIR="${OUTPUT_DIR:-tmp_results/basic_training}"

EMBEDDING_DIM="${EMBEDDING_DIM:-32}"
HIDDEN_DIM="${HIDDEN_DIM:-512}"
NUM_LAYERS="${NUM_LAYERS:-3}"
NUM_HEADS="${NUM_HEADS:-8}"
NUM_INDUCING_POINTS="${NUM_INDUCING_POINTS:-32}"
USE_LAYER_NORM="${USE_LAYER_NORM:---use-layer-norm}"
DROPOUT="${DROPOUT:-0.0}"

MAX_ROWS="${MAX_ROWS:-20000}"
FRACTION="${FRACTION:-1.0}"
NUM_EPOCHS="${NUM_EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
NUM_WORKERS="${NUM_WORKERS:-2}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
CLIP_GRAD_NORM="${CLIP_GRAD_NORM:-1.0}"
SCHEDULER_FACTOR="${SCHEDULER_FACTOR:-0.1}"
SCHEDULER_PATIENCE="${SCHEDULER_PATIENCE:-2}"
PATIENCE="${PATIENCE:-3}"
NUM_MASKS_PER_SET="${NUM_MASKS_PER_SET:-2}"
SKEW_FACTOR="${SKEW_FACTOR:-1.2}"
INCLUDE_NULL="${INCLUDE_NULL:-}"
METRIC_UPDATE_INTERVAL="${METRIC_UPDATE_INTERVAL:-50}"
DEVICE="${DEVICE:-}"

# DDP settings (set USE_DDP=1 to enable).
USE_DDP="${USE_DDP:-}"
DDP_NPROC_PER_NODE="${DDP_NPROC_PER_NODE:-4}"
DDP_NNODES="${DDP_NNODES:-1}"
DDP_NODE_RANK="${DDP_NODE_RANK:-0}"

CMD=(
  --data-path "${DATA_PATH}"
  --vocab-path "${VOCAB_PATH}"
  --weapon-vocab-path "${WEAPON_VOCAB_PATH}"
  --output-dir "${OUTPUT_DIR}"
  --embedding-dim "${EMBEDDING_DIM}"
  --hidden-dim "${HIDDEN_DIM}"
  --num-layers "${NUM_LAYERS}"
  --num-heads "${NUM_HEADS}"
  --num-inducing-points "${NUM_INDUCING_POINTS}"
  ${USE_LAYER_NORM}
  --dropout "${DROPOUT}"
  --max-rows "${MAX_ROWS}"
  --fraction "${FRACTION}"
  --num-epochs "${NUM_EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --learning-rate "${LEARNING_RATE}"
  --weight-decay "${WEIGHT_DECAY}"
  --clip-grad-norm "${CLIP_GRAD_NORM}"
  --scheduler-factor "${SCHEDULER_FACTOR}"
  --scheduler-patience "${SCHEDULER_PATIENCE}"
  --patience "${PATIENCE}"
  --num-masks-per-set "${NUM_MASKS_PER_SET}"
  --skew-factor "${SKEW_FACTOR}"
  --metric-update-interval "${METRIC_UPDATE_INTERVAL}"
)

if [[ -n "${INCLUDE_NULL}" ]]; then
  CMD+=("${INCLUDE_NULL}")
fi

if [[ -n "${DEVICE}" ]]; then
  CMD+=(--device "${DEVICE}")
fi

if [[ -n "${USE_DDP}" ]]; then
  CMD+=(--distributed)
  LAUNCH=(
    poetry run torchrun
    --nproc_per_node "${DDP_NPROC_PER_NODE}"
    --nnodes "${DDP_NNODES}"
    --node_rank "${DDP_NODE_RANK}"
    src/splatnlp/model/basic_training_cli.py
  )
else
  LAUNCH=(poetry run python -m splatnlp.model.basic_training_cli)
fi

echo "Launching basic training"
printf '» %q ' "${LAUNCH[@]}" "${CMD[@]}"; echo
"${LAUNCH[@]}" "${CMD[@]}"
