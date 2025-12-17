# SplatNLP — SplatGPT + SAE interpretability for Splatoon 3 gear builds

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cesaregarza/SplatNLP/blob/main/notebooks/colab_demo.ipynb)

This repository contains the code for SplatNLP, an end-to-end ML + mech-interp
project for predicting and analyzing optimal Splatoon 3 gear loadouts.

**Quick links**

- Reviewer guide: `docs/START_HERE.md`
- Project one-pager: `docs/PROJECT_ONEPAGER.md`
- Runnable demo: `notebooks/colab_demo.ipynb`
- Canonical blog posts: `cegarza.com` (see below)

## Overview

Optimizing gear loadouts in Splatoon 3 presents a unique challenge due to intricate ability stacking mechanics, weapon-specific synergies, context-dependent effectiveness and noisy real-world data. Traditional ML approaches often struggle with the set-based nature of gear and the complex interactions involved.

This project tackles this challenge through an end-to-end machine learning pipeline. It explores multiple approaches, including representing gear sets using **Doc2Vec embeddings** for analysis and clustering (see the `embeddings` module) and developing the core **`SetCompletionModel` (`SplatGPT`)**, a novel architecture designed specifically for set-based prediction tasks.

**Core Model (`SplatGPT`):** The primary model (`SetCompletionModel`) leverages principles from Set Transformers and GPT-2, incorporating unique attention mechanisms to process gear sets effectively while considering weapon context.

The model is approximately **83 million parameters** in size and is available in two variants:

- **Full:** Trained on a single H100 GPU for 62 hours, using 5 subset variants
  per data point (each subset created via randomized masking).
- **Ultra:** Trained on four B200 GPUs for 35 hours, utilizing 20 subset variants
  per data point. This is the checkpoint used by the Colab demo (along with an
  Ultra SAE for interpretability/probing).

Both variants are referenced under `saved_models/` for offline inspection when
you have local model artifacts available (see the quickstart below).

**Sparse Autoencoder (SAE):** The SAE is trained on the activations of the
primary model to provide a sparse feature representation for interpretability
and analysis (see `src/splatnlp/monosemantic_sae/`). The `SetCompletionHook`
supports a passthrough/probe mode (capture feature activations without changing
model outputs) and a reconstruction mode (insert the SAE reconstruction, which
can change discrete beam-search outcomes). This is based on Anthropic's work:
[Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features/index.html)

---

**Start Here (Reviewer Guide)**

If you’re reviewing this repo, start with:

- `docs/START_HERE.md`
- `notebooks/colab_demo.ipynb` (Colab-friendly demo)
  - https://colab.research.google.com/github/cesaregarza/SplatNLP/blob/main/notebooks/colab_demo.ipynb

---

**Blog Series (Canonical)**

For a comprehensive deep-dive into the problem definition, the novel model
architecture (`SplatGPT`), methodology, data processing techniques, results and
insights, please read the canonical posts on `cegarza.com`:

[SplatGPT: Set-Based Deep Learning for Splatoon 3 Gear Completion](https://cegarza.com/introducing-splatgpt/)

Note: the `docs/splatgpt-blog-part-*.txt` / `docs/splatgpt-blog-part-*-draft.md`
files are LLM-friendly extracts/drafts for tooling; they are not the canonical
published/publishable versions.

---
## Key Features

* **End-to-End Pipeline:** Covers data acquisition from stat.ink, sophisticated preprocessing, model training, evaluation, and API serving.
* **Novel Architecture (`SetCompletionModel`):** Implements a custom model inspired by Set Transformer and GPT-2 principles, featuring attention mechanisms like Induced Set Attention and Pooling Multihead Attention to handle set-based inputs effectively (see `src/splatnlp/model/models.py`).
* **Model Variants:** Two versions of the `SetCompletionModel` available:
  - **Full (83M params):** Extensively tested with complete SAE interpretability.
  - **Ultra (83M params):** Experimental, leveraging significantly more diverse subset variants per data point for richer context, SAE interpretability in progress.
* **Embedding-Based Analysis:** Provides tools for training Doc2Vec models on gear sets, performing TF-IDF analysis, clustering builds using UMAP and DBSCAN, and visualizing embeddings (`src/splatnlp/embeddings/`).
* **Advanced Preprocessing:** Includes domain-specific logic for ability bucketing based on Ability Point (AP) thresholds, tokenization, handling game patches, and targeted sampling to bias towards optimal configurations. Uses PyArrow for memory efficiency during partitioning (see `src/splatnlp/preprocessing/`).
* **Interpretability via Sparse Autoencoders (SAEs):** Incorporates training of SAEs on the *activations* of the primary model for feature analysis and interpretability, following recent research trends (see `src/splatnlp/monosemantic_sae/`).
* **API Serving:** Provides a FastAPI application (`src/splatnlp/serve/`) to serve the trained `SetCompletionModel` for real-time gear set completion predictions.
* **Command-Line Tools:** Offers CLIs for orchestrating preprocessing (`src/splatnlp/preprocessing/pipeline.py`), training the main model (`src/splatnlp/model/cli.py`), training SAEs (`src/splatnlp/monosemantic_sae/sae_training/cli.py`), and running embedding experiments (`src/splatnlp/embeddings/cli.py`).
* **Visualization Utilities:** Contains tools for dimensionality reduction (t-SNE via `embeddings` CLI) and fetching weapon images/abbreviations (`src/splatnlp/viz/`) to support analysis.
* **Hyperparameter Optimization:** Includes utilities for grid search (`src/splatnlp/model/grid_search.py`).

## Project Structure

```
SplatNLP/
├── src/
│   └── splatnlp/
│       ├── embeddings/             # Embeddings, TF-IDF analysis, clustering
│       ├── model/                  # SplatGPT model, training, evaluation, CLI
│       ├── monosemantic_sae/       # Sparse Autoencoder tools and training
│       ├── preprocessing/          # Data preprocessing, sampling, tokenization
│       ├── serve/                  # API serving via FastAPI
│       └── viz/                    # Visualization utilities
├── data/                           # Raw and processed data storage
├── models/                         # Trained model artifacts
├── LICENSE                         # GNU GPL-3.0 License
├── poetry.lock                     # Dependency lock file
├── pyproject.toml                  # Project dependencies and configuration
└── README.md                       # Project overview
```
## Quickstart (local CPU demo)

This demo expects a local copy of pretrained artifacts under
`saved_models/dataset_v2` (model checkpoint + `vocab.json` +
`weapon_vocab.json` + `model_params.json`). Those large artifacts are not
tracked in git. You can download them from the configured artifact host via:

`poetry run python -m splatnlp.utils.download_artifacts --dataset-dir dataset_v2`

You can also override the host/path using `DO_SPACES_ML_ENDPOINT` and
`DO_SPACES_ML_DIR`.

1. Install dependencies (dev extras include formatting/testing):\
   `poetry install --with dev`\
   (For the Doc2Vec/embeddings workflow, also install `gensim` via
   `poetry install --with dev,embeddings`.)
2. (Optional) Download artifacts if you don’t already have them:

   ```bash
   poetry run python -m splatnlp.utils.download_artifacts \
     --base-url https://splat-nlp.nyc3.cdn.digitaloceanspaces.com \
     --dataset-dir dataset_v2
   ```

   To also download Ultra + Ultra SAE artifacts (and optional Ultra feature
   labels, if hosted), add `--include-ultra-sae`. To fetch labels only, add
   `--include-ultra-labels`.

3. Run a one-off inference:

   ```bash
   poetry run python - <<'PY'
   import json, torch
   from pathlib import Path
   from splatnlp.model.models import SetCompletionModel
   from splatnlp.serve.tokenize import tokenize_build

   base = Path("saved_models/dataset_v2")
   if not base.exists():
       base = Path("saved_models/dataset_v0_2")
   params = json.loads(base.joinpath("model_params.json").read_text())
   vocab = json.loads(base.joinpath("vocab.json").read_text())
   weapon_vocab = json.loads(base.joinpath("weapon_vocab.json").read_text())

   model = SetCompletionModel(**params)
   model.load_state_dict(torch.load(base / "model.pth", map_location="cpu"))
   model.eval()

   tokens = tokenize_build({"ink_saver_main": 6, "run_speed_up": 12, "intensify_action": 10})
   weapon_id = "weapon_id_1001"
   input_tokens = torch.tensor([[vocab[t] for t in tokens]])
   input_weapons = torch.tensor([[weapon_vocab[weapon_id]]])
   key_padding_mask = input_tokens == params["pad_token_id"]

   with torch.no_grad():
       preds = torch.sigmoid(
           model(input_tokens, input_weapons, key_padding_mask=key_padding_mask)
       ).squeeze()

   skip = {vocab["<PAD>"], vocab["<NULL>"]}
   inv_vocab = {v: k for k, v in vocab.items()}
   scores = [(i, float(p)) for i, p in enumerate(preds.tolist()) if i not in skip]
   top5 = sorted(scores, key=lambda x: x[1], reverse=True)[:5]
   print("tokens:", tokens)
   print("top-5:", [(inv_vocab[i], p) for i, p in top5])
   PY
   ```

4. (Optional) Run the test suite:\
   `poetry run pytest -q`

## Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/cesaregarza/SplatNLP](https://github.com/cesaregarza/SplatNLP)
    cd SplatNLP
    ```
2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Linux/macOS:
    source venv/bin/activate
    # On Windows:
    # venv\Scripts\activate
    ```
3.  **Install dependencies using Poetry (includes dev/test tools):**
    ```bash
    pip install poetry # If you don't have poetry installed
    poetry install --with dev
    # Optional (Doc2Vec/embeddings):
    poetry install --with dev,embeddings
    ```
4.  **(Optional) Set environment variables for serving:**
    The API server (`src/splatnlp/serve/app.py`) loads model artifacts from URLs specified by environment variables. See `src/splatnlp/serve/load_model.py` for details (e.g., `VOCAB_URL`, `MODEL_URL`, `PARAMS_URL`, `WEAPON_VOCAB_URL`, `INFO_URL` or `DO_SPACES_ML_ENDPOINT`/`DO_SPACES_ML_DIR`).

## Development workflow

- Format: `poetry run isort .` then `poetry run black .` (line length 80)
- Test: `poetry run pytest -q` (uses fixtures under `test_data/`)

## Usage Examples

*(Note: Adjust paths and parameters according to your setup. Data paths often expect TSV/CSV files where the `ability_tags` column contains JSON-parseable lists of integers/strings depending on the context.)*

**1. Run the Preprocessing Pipeline:**
*(Downloads data from stat.ink, processes it including ability bucketing and sampling, and saves partitioned output. Final output is `weapon_partitioned.csv` in the base path.)*
```bash
# Example: Persist raw downloads, process, save result to data/weapon_partitioned.csv
python -m splatnlp.preprocessing.pipeline --base_path data/ --persist
```

**2. Train/Use Doc2Vec Embeddings:**

```bash
# Example: Train a Doc2Vec model with clustering
python -m splatnlp.embeddings.cli \
    --data_path path/to/your/tokenized_data.csv \
    --vocab_path path/to/your/vocab.json \
    --weapon_vocab_path path/to/your/weapon_vocab.json \
    --output_dir ./embeddings_output \
    --vector_size 100 \
    --window 25 \
    --min_count 1 \
    --epochs 10 \
    --dm 0 \
    --workers 16 \
    --n_components 2 \
    --perplexity 30.0 \
    --n_iter 1000 \
    --eps 0.5 \
    --min_samples 5 \
    --umap_neighbors 15 \
    --umap_min_dist 0.1 \
    --random_state 42 \
    --verbose True
> **Note:** The pretrained `doc2vec.model` is not included in the repository. Download it from the project releases and place it in `./embeddings_output/` before running inference or clustering.


# Example: Run inference with trained model
python -m splatnlp.embeddings.cli \
    --mode infer \
    --model_path ./embeddings_output/doc2vec.model \
    # ... (same paths as above) ...

# Example: Run dimensionality reduction
python -m splatnlp.embeddings.cli \
    --mode reduce \
    --model_path ./embeddings_output/doc2vec.model \
    # ... (same paths as above) ...

# Example: Run clustering
python -m splatnlp.embeddings.cli \
    --mode cluster \
    --model_path ./embeddings_output/doc2vec.model \
    # ... (same paths as above) ...
```

**3. Train the Main `SetCompletionModel` (`SplatGPT`):**

```bash
# Example: Train using a tokenized CSV file
python -m splatnlp.model.cli \
    --data_path path/to/your/tokenized_data.csv \
    --vocab_path path/to/your/vocab.json \
    --weapon_vocab_path path/to/your/weapon_vocab.json \
    --output_dir ./trained_splatgpt_model \
    --embedding_dim 32 \
    --hidden_dim 512 \
    --num_layers 3 \
    --use_layer_norm True \
    --dropout 0.3 \
    --learning_rate 0.0001 \
    --fraction 1 \
    --verbose True \
    --num_epochs 20 \
    --batch_size 1024 \
    --use_mixed_precision True \
    --num_workers 16 \
    --metric_update_interval 1000
```

**4. Train a Sparse Autoencoder (SAE) on Model Activations:**

> NOTE: This is currently experimental and may require significant tuning.

```bash
# Example: Train an SAE on masked_mean activations from a pretrained SetCompletionModel
python -m scripts.train_sae \
    --model-checkpoint path/to/your/model.pth \
    --data-csv path/to/your/tokenized_data.csv \
    --save-dir ./trained_sae \
    --vocab-path path/to/your/vocab.json \
    --weapon-vocab-path path/to/your/weapon_vocab.json \
    --primary-embedding-dim 32 \
    --primary-hidden-dim 512 \
    --primary-num-layers 3 \
    --primary-num-heads 8 \
    --primary-num-inducing 32 \
    --epochs 10 \
    --expansion-factor 4 \
    --lr 1e-4 \
    --l1-coeff 1e-4 \
    --target-usage 7e-3 \
    --usage-coeff 0.0 \
    --gradient-clip-val 1.0 \
    --buffer-size 100000 \
    --sae-batch-size 1024 \
    --steps-before-train 50000 \
    --sae-train-steps 4 \
    --primary-data-fraction 0.005 \
    --resample-weight 0.01 \
    --resample-bias -1.0 \
    --dead-neuron-threshold 1e-6 \
    --kl-floor 0.0 \
    --resample-steps 7000 14000 21000 28000 \
    --device cuda \
    --num-workers 16 \
    --verbose
```

**5. Run the API Server (Serves `SetCompletionModel`):**
*(Requires model artifacts accessible via URLs configured through environment variables)*

> NOTE: THIS HAS NO SECURITY MEASURES, IT IS DESIGNED TO BE USED IN A LOCAL ENVIRONMENT OR SILOED OFF IN A CONTAINERIZED ENVIRONMENT WITH STRICT NETWORKING POLICIES. DO NOT DEPLOY THIS IN A PRODUCTION ENVIRONMENT WITHOUT ADDING THE APPROPRIATE SECURITY MEASURES. Artifacts are fetched over HTTP and deserialized with `torch.load` (pickle), so only point the env vars at trusted, integrity-checked endpoints.

```bash
# Ensure environment variables for model URLs are set (see src/splatnlp/serve/load_model.py)
# Example: uvicorn module.path:app --host <host> --port <port>
uvicorn splatnlp.serve.app:app --host 0.0.0.0 --port 9000 --reload
```

**6. Query the API Endpoint:**

```bash
# Example: Get predictions for a partial build (Splattershot Pro - ID 310)
# Note: Provide AP values as integers (e.g., 1 main = 10, 1 sub = 3)
curl -X POST "http://localhost:9000/infer" \
     -H "Content-Type: application/json" \
     -d '{
          "target": {
            "ink_saver_main": 6,
            "run_speed_up": 12,
            "intensify_action": 10
          },
          "weapon_id": 310
        }'

# Example: Get baseline build for a weapon (using NULL token logic)
curl -X POST "http://localhost:9000/infer" \
     -H "Content-Type: application/json" \
     -d '{
          "target": {},
          "weapon_id": 310
        }'
# Expected response structure (values are examples):
# {
#   "predictions": [
#     ["ability_tag_1", 0.95],
#     ["ability_tag_2", 0.88],
#     ...
#   ],
#   "splatgpt_info": { ... model metadata ... },
#   "api_version": "0.2.0",
#   "inference_time": 0.05
# }

# Example Test GET request (uses hardcoded input)
curl "http://localhost:9000/infer"
```

## Architecture

(A diagram and detailed explanation of the `SetCompletionModel` architecture can be found in the blog post linked above.)

## License

This project is licensed under the GNU General Public License v3.0 (GPL-3.0).
See the [LICENSE](LICENSE) file for details.
