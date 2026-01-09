#!/usr/bin/env bash
set -euo pipefail

# Wrapper for model_embeddings training-style extraction (masked contexts).

DATA_PATH="${DATA_PATH:-https://splat-nlp.nyc3.cdn.digitaloceanspaces.com/dataset_v2/tokenized_data.csv}"
VOCAB_PATH="${VOCAB_PATH:-https://splat-nlp.nyc3.cdn.digitaloceanspaces.com/dataset_v2/vocab.json}"
WEAPON_VOCAB_PATH="${WEAPON_VOCAB_PATH:-https://splat-nlp.nyc3.cdn.digitaloceanspaces.com/dataset_v2/weapon_vocab.json}"
MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-https://splat-nlp.nyc3.cdn.digitaloceanspaces.com/dataset_v2/model_ultra.pth}"
MODEL_PARAMS="${MODEL_PARAMS:-}"
OUTPUT_DIR="${OUTPUT_DIR:-tmp_results/model_embeddings}"

EMBEDDING_DIM="${EMBEDDING_DIM:-32}"
HIDDEN_DIM="${HIDDEN_DIM:-512}"
NUM_LAYERS="${NUM_LAYERS:-3}"
NUM_HEADS="${NUM_HEADS:-8}"
NUM_INDUCING_POINTS="${NUM_INDUCING_POINTS:-32}"
USE_LAYER_NORM="${USE_LAYER_NORM:---use-layer-norm}"
DROPOUT="${DROPOUT:-0.0}"

NUM_MASKS_PER_SET="${NUM_MASKS_PER_SET:-5}"
SKEW_FACTOR="${SKEW_FACTOR:-1.2}"
INCLUDE_NULL="${INCLUDE_NULL:-}"
LIMIT="${LIMIT:-}"

BATCH_SIZE="${BATCH_SIZE:-512}"
NUM_WORKERS="${NUM_WORKERS:-0}"
SHUFFLE="${SHUFFLE:-}"
DEVICE="${DEVICE:-}"
LOG_INTERVAL="${LOG_INTERVAL:-50}"
NO_NORMALIZE="${NO_NORMALIZE:-}"

# DDP settings (set USE_DDP=1 to enable).
USE_DDP="${USE_DDP:-}"
DDP_NPROC_PER_NODE="${DDP_NPROC_PER_NODE:-4}"
DDP_NNODES="${DDP_NNODES:-1}"
DDP_NODE_RANK="${DDP_NODE_RANK:-0}"

# Weights & Biases (set WANDB_LOG=1 to enable).
WANDB_LOG="${WANDB_LOG:-}"
WANDB_PROJECT="${WANDB_PROJECT:-splatnlp-embeddings}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-}"
WANDB_TAGS="${WANDB_TAGS:-}"

CMD=(
  --data-path "${DATA_PATH}"
  --vocab-path "${VOCAB_PATH}"
  --weapon-vocab-path "${WEAPON_VOCAB_PATH}"
  --model-checkpoint "${MODEL_CHECKPOINT}"
  --output-dir "${OUTPUT_DIR}"
  --embedding-dim "${EMBEDDING_DIM}"
  --hidden-dim "${HIDDEN_DIM}"
  --num-layers "${NUM_LAYERS}"
  --num-heads "${NUM_HEADS}"
  --num-inducing-points "${NUM_INDUCING_POINTS}"
  ${USE_LAYER_NORM}
  --dropout "${DROPOUT}"
  --num-masks-per-set "${NUM_MASKS_PER_SET}"
  --skew-factor "${SKEW_FACTOR}"
  --batch-size "${BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --log-interval "${LOG_INTERVAL}"
)

if [[ -n "${MODEL_PARAMS}" ]]; then
  CMD+=(--model-params "${MODEL_PARAMS}")
fi

if [[ -n "${INCLUDE_NULL}" ]]; then
  CMD+=("${INCLUDE_NULL}")
fi

if [[ -n "${LIMIT}" ]]; then
  CMD+=(--limit "${LIMIT}")
fi

if [[ -n "${SHUFFLE}" ]]; then
  CMD+=(--shuffle)
fi

if [[ -n "${DEVICE}" ]]; then
  CMD+=(--device "${DEVICE}")
fi

if [[ -n "${NO_NORMALIZE}" ]]; then
  CMD+=(--no-normalize)
fi

if [[ -n "${WANDB_LOG}" ]]; then
  CMD+=(--wandb-log)
  CMD+=(--wandb-project "${WANDB_PROJECT}")
  if [[ -n "${WANDB_ENTITY}" ]]; then
    CMD+=(--wandb-entity "${WANDB_ENTITY}")
  fi
  if [[ -n "${WANDB_RUN_NAME}" ]]; then
    CMD+=(--wandb-run-name "${WANDB_RUN_NAME}")
  fi
  if [[ -n "${WANDB_TAGS}" ]]; then
    # shellcheck disable=SC2206
    CMD+=(--wandb-tags ${WANDB_TAGS})
  fi
fi

if [[ -n "${USE_DDP}" ]]; then
  CMD+=(--distributed)
  LAUNCH=(
    poetry run torchrun
    --nproc_per_node "${DDP_NPROC_PER_NODE}"
    --nnodes "${DDP_NNODES}"
    --node_rank "${DDP_NODE_RANK}"
    src/splatnlp/model_embeddings/harness_cli.py
  )
else
  LAUNCH=(poetry run python -m splatnlp.model_embeddings.harness_cli)
fi

echo "Launching model_embeddings extraction"
printf '» %q ' "${LAUNCH[@]}" "${CMD[@]}"; echo
"${LAUNCH[@]}" "${CMD[@]}"
