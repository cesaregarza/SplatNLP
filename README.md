# SplatNLP: End-to-End ML Pipeline for Splatoon 3 Gear Set Optimization and Analysis

This is SplatNLP, a machine learning project for predicting optimal gear loadouts in Splatoon 3.

## Overview

Optimizing gear loadouts in Splatoon 3 is tricky. You've got ability stacking mechanics, weapon-specific synergies, context-dependent effectiveness, and noisy real-world data to deal with. Standard ML approaches don't work well because gear sets are inherently set-based with complex interactions.

This project tackles the problem with an end-to-end ML pipeline. It includes Doc2Vec embeddings for analysis and clustering (see the `embeddings` module), but the main focus is the **SetCompletionModel** (nicknamed **SplatGPT**), a custom architecture built specifically for set-based prediction tasks.

**Core Model (SplatGPT):** The SetCompletionModel combines ideas from Set Transformers and GPT-2, using custom attention mechanisms to process gear sets while accounting for weapon context.

The model has about **83 million parameters** and comes in two variants:

- **Full:** Trained on a single H100 GPU for 62 hours, using 5 subset variants per data point (each subset created via randomized masking). This variant has been thoroughly explored and includes a fully-trained monosemantic sparse autoencoder (SAE) with labeled neurons for interpretability.
- **Ultra:** Trained on four B200 GPUs for 35 hours, using 20 subset variants per data point. Still being explored; the monosemantic SAE is being trained.

A Google Colab notebook for inference and analysis is coming soon.

**Sparse Autoencoder (SAE):** The SAE trains on activations from the primary model to produce sparse, monosemantic representations of gear sets. This makes the model's predictions interpretable and allows for feature analysis (see the `monosemantic_sae` module). It includes a `SetCompletionHook` for hooking into the primary model and modifying activations during inference (model steering). Based on Anthropic's work: [Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features/index.html)

---

**Blog Post**

For a detailed writeup on the problem definition, model architecture, methodology, data processing, results, and insights, check out the blog post:

[SplatGPT: Set-Based Deep Learning for Splatoon 3 Gear Completion](https://cegarza.com/introducing-splatgpt/)

---
## Key Features

* **End-to-End Pipeline:** Data acquisition from stat.ink, preprocessing, model training, evaluation, and API serving.
* **Custom Architecture (SetCompletionModel):** A model inspired by Set Transformer and GPT-2, with attention mechanisms like Induced Set Attention and Pooling Multihead Attention for handling set-based inputs (see `src/splatnlp/model/models.py`).
* **Model Variants:** Two versions available:
  - **Full (83M params):** Well-tested with complete SAE interpretability.
  - **Ultra (83M params):** Experimental, uses more diverse subset variants per data point. SAE interpretability in progress.
* **Embedding-Based Analysis:** Tools for training Doc2Vec models on gear sets, TF-IDF analysis, clustering with UMAP and DBSCAN, and embedding visualization (`src/splatnlp/embeddings/`).
* **Advanced Preprocessing:** Domain-specific logic for ability bucketing based on Ability Point (AP) thresholds, tokenization, patch handling, and targeted sampling toward optimal configurations. Uses PyArrow for memory efficiency (see `src/splatnlp/preprocessing/`).
* **Interpretability via Sparse Autoencoders (SAEs):** SAE training on model activations for feature analysis and interpretability (see `src/splatnlp/monosemantic_sae/`).
* **API Serving:** FastAPI application (`src/splatnlp/serve/`) for serving the trained SetCompletionModel.
* **Command-Line Tools:** CLIs for preprocessing (`src/splatnlp/preprocessing/pipeline.py`), main model training (`src/splatnlp/model/cli.py`), SAE training (`src/splatnlp/monosemantic_sae/sae_training/cli.py`), and embedding experiments (`src/splatnlp/embeddings/cli.py`).
* **Visualization Utilities:** Tools for dimensionality reduction (t-SNE via `embeddings` CLI) and fetching weapon images/abbreviations (`src/splatnlp/viz/`).
* **Hyperparameter Optimization:** Grid search utilities (`src/splatnlp/model/grid_search.py`).

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
3.  **Install dependencies using Poetry:**
    ```bash
    pip install poetry # If you don't have poetry installed
    poetry install
    ```
4.  **(Optional) Set environment variables for serving:**
    The API server (`src/splatnlp/serve/app.py`) loads model artifacts from URLs specified by environment variables. See `src/splatnlp/serve/load_model.py` for details (e.g., `VOCAB_URL`, `MODEL_URL`, `PARAMS_URL`, `WEAPON_VOCAB_URL`, `INFO_URL` or `DO_SPACES_ML_ENDPOINT`/`DO_SPACES_ML_DIR`).

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

> NOTE: THIS HAS NO SECURITY MEASURES, IT IS DESIGNED TO BE USED IN A LOCAL ENVIRONMENT OR SILOED OFF IN A CONTAINERIZED ENVIRONMENT WITH STRICT NETWORKING POLICIES. DO NOT DEPLOY THIS IN A PRODUCTION ENVIRONMENT WITHOUT ADDING THE APPROPRIATE SECURITY MEASURES.

```bash
# Ensure environment variables for model URLs are set (see src/splatnlp/serve/load_model.py)
# Example: uvicorn module.path:app --host <host> --port <port>
uvicorn splatnlp.serve.app:app --host 0.0.0.0 --port 9000 --reload
```

**6. Query the API Endpoint:**

```bash
# Get predictions for a partial build (Splattershot Pro, ID 310)
# AP values are integers (1 main = 10, 1 sub = 3)
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

# Get baseline build for a weapon (uses NULL token logic)
curl -X POST "http://localhost:9000/infer" \
     -H "Content-Type: application/json" \
     -d '{
          "target": {},
          "weapon_id": 310
        }'
# Response structure:
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

# Test GET request (uses hardcoded input)
curl "http://localhost:9000/infer"
```

## Architecture

See the blog post linked above for a diagram and detailed explanation of the SetCompletionModel architecture.

## License

GNU General Public License v3.0 (GPL-3.0). See the [LICENSE](LICENSE) file.
