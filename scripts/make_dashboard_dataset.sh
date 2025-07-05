#!/usr/bin/env bash
set -euo pipefail

# --- Configuration ---
SAE_CONFIG_FILE="https://splat-nlp.nyc3.cdn.digitaloceanspaces.com/dataset_v2/sae_config_ultra.json"

# --- Model and Data Paths ---
PRIMARY_MODEL="https://splat-nlp.nyc3.cdn.digitaloceanspaces.com/dataset_v2/model_ultra.pth"
SAE_MODEL="https://splat-nlp.nyc3.cdn.digitaloceanspaces.com/dataset_v2/sae_model_ultra.pth"
DATASET="https://splat-nlp.nyc3.cdn.digitaloceanspaces.com/dataset_v2/tokenized_data.csv"

# --- Vocabularies ---
ABILITY_VOCAB="https://splat-nlp.nyc3.cdn.digitaloceanspaces.com/dataset_v2/vocab.json"
WEAPON_VOCAB="https://splat-nlp.nyc3.cdn.digitaloceanspaces.com/dataset_v2/weapon_vocab.json"

# --- Output Configuration ---
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="mnt/e/activations_ultra/dataset_${TIMESTAMP}"
mkdir -p "${OUTPUT_DIR}"

# --- Dataset Configuration ---
DATA_FRACTION=1.0
NUM_MASKS_PER_SET=5
SKEW_FACTOR=1.2

# --- Processing Configuration ---
DEVICE="cuda"
CHUNK_SIZE=5000
ACTIVATION_DTYPE="fp16"
WORKERS=0  # Keep at 0 for streaming inference

# --- Optional Features ---
EMIT_TOKEN_COMBOS=false
TOP_K=30
COMPUTE_CORRELATIONS=false
DEBUG_SAVE_INPUTS=false

# --- Logging ---
LOG_LEVEL="INFO"
VERBOSE_FLAG="--verbose"

CMD=(
  poetry run python3 -m splatnlp.dashboard.commands.make_dashboard_dataset

  --primary-model         "${PRIMARY_MODEL}"
  --sae-model            "${SAE_MODEL}"
  --ability-vocab        "${ABILITY_VOCAB}"
  --weapon-vocab         "${WEAPON_VOCAB}"
  --dataset              "${DATASET}"
  --output-dir           "${OUTPUT_DIR}"

  --sae-config            "${SAE_CONFIG_FILE}"

  --data-fraction          "${DATA_FRACTION}"
  --num-masks-per-set      "${NUM_MASKS_PER_SET}"
  --skew-factor            "${SKEW_FACTOR}"

  --device                 "${DEVICE}"
  --chunk-size             "${CHUNK_SIZE}"
  --activation-dtype       "${ACTIVATION_DTYPE}"
  --workers                "${WORKERS}"

  --top-k                  "${TOP_K}"
  --log-level              "${LOG_LEVEL}"
)

# --- Add optional flags ---
if [[ "${EMIT_TOKEN_COMBOS}" == "true" ]]; then
  CMD+=(--emit-token-combos)
fi

if [[ "${COMPUTE_CORRELATIONS}" == "true" ]]; then
  CMD+=(--compute-correlations)
fi

if [[ "${DEBUG_SAVE_INPUTS}" == "true" ]]; then
  CMD+=(--debug-save-inputs)
fi

if [[ -n "${VERBOSE_FLAG}" ]]; then
  CMD+=("${VERBOSE_FLAG}")
fi

echo ""
echo "🚀 Launching dashboard dataset generation"
echo "   Output directory: ${OUTPUT_DIR}"
echo "   SAE model: ${SAE_MODEL}"
echo "   SAE config: ${SAE_CONFIG_FILE}"
echo ""

printf '» %q ' "${CMD[@]}"; echo
echo ""

"${CMD[@]}"

echo ""
echo "✅ Dashboard dataset generation finished"
echo "   Output directory: ${OUTPUT_DIR}"
echo ""
echo "📁 Generated files:"
echo "   • analysis_df.ipc - Main dataset with example metadata"
echo "   • idf.ipc - Inverse document frequency for tokens"
echo "   • neuron_XXXXX/ - Per-feature activation data"
echo "   • logit_influences.jsonl - Feature influence on vocabulary"
echo "   • metadata.json - Generation metadata and configuration"
echo ""
echo "🔗 To use with the dashboard, point fs_database.py to: ${OUTPUT_DIR}" 