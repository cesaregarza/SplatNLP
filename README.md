# SplatNLP: End-to-End ML Pipeline for Splatoon 3 Gear Set Optimization and Analysis

This repository contains the code for SplatNLP, a project focused on applying machine learning techniques to understand and predict optimal gear loadouts in the complex environment of Nintendo's Splatoon 3.

## Overview

Optimizing gear loadouts in Splatoon 3 presents a unique challenge due to intricate ability stacking mechanics, weapon-specific synergies, context-dependent effectiveness, and noisy real-world data. Traditional ML approaches often struggle with the set-based nature of gear and the complex interactions involved.

This project tackles this challenge through an end-to-end machine learning pipeline. It explores representing gear sets mathematically and predicting optimal combinations based on player data.

**Iterative Development:** The project began by exploring traditional NLP techniques (Doc2Vec embeddings - see the `embeddings` module) to validate the feasibility of representing gear sets and identifying meaningful relationships between weapon configurations. The promising results from this initial analysis motivated the development of the core `SetCompletionModel` (`SplatGPT`), a novel architecture designed specifically for this set-based prediction task.

**Core Model (`SplatGPT`):** The primary model (`SetCompletionModel`) leverages principles from Set Transformers and GPT-2, incorporating a unique cross-attention mechanism to process gear sets effectively while considering weapon context.

---

**Blog Post Deep Dive:**

For a comprehensive deep-dive into the problem definition, the novel model architecture (`SplatGPT`), methodology, data processing techniques, results, and insights, please read the accompanying blog post:

[SplatGPT: Set-Based Deep Learning for Splatoon 3 Gear Completion](https://cegarza.com/introducing-splatgpt/)

---

## Key Features

* **End-to-End Pipeline:** Covers data acquisition from stat.ink, sophisticated preprocessing, model training, evaluation, and API serving.
* **Novel Architecture (`SetCompletionModel`):** Implements a custom model inspired by Set Transformer and GPT-2 principles, featuring a unique cross-attention mechanism to handle set-based inputs effectively (see `src/splatnlp/model/models.py`).
* **Advanced Preprocessing:** Includes domain-specific logic for ability bucketing, tokenization, handling game patches, and targeted sampling to bias towards optimal configurations (see `src/splatnlp/preprocessing/`).
* **Interpretability via Sparse Autoencoders (SAEs):** Incorporates training of SAEs on model activations for feature analysis and interpretability, following recent research trends (see `src/splatnlp/monosemanticity_train/`).
* **API Serving:** Provides a FastAPI application (`src/splatnlp/serve/`) to serve the trained model for real-time gear set completion predictions.
* **Command-Line Tools:** Offers CLIs for orchestrating preprocessing, training the main model, and training SAEs.
* **Initial NLP Exploration:** Includes the initial Doc2Vec embedding and clustering experiments (`src/splatnlp/embeddings/`) that validated the core approach.
* **Visualization Utilities:** Contains tools for dimensionality reduction (t-SNE) and fetching weapon images (`src/splatnlp/viz/`).

## Project Structure

```
SplatNLP/
├── src/splatnlp/
│   ├── embeddings/         # Initial Doc2Vec embedding generation, clustering, analysis
│   ├── model/              # Core SetCompletionModel definition, training loop, evaluation, CLI
│   ├── monosemanticity_train/ # Sparse Autoencoder (SAE) models, training loop, CLI for interpretability
│   ├── preprocessing/      # Data pulling, transformation, tokenization, dataset creation pipeline
│   ├── serve/              # FastAPI application for serving the model via API
│   └── viz/                # Utilities for visualizing embeddings and weapon data
├── data/                 # (Example) Directory for storing raw/processed data
├── models/               # (Example) Directory for storing trained models/vocab
└── requirements.txt      # Project dependencies
```

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/cesaregarza/SplatNLP
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
3.  **Install dependencies:**
    ```bash
    poetry install
    ```
4.  **(Optional) Set environment variables for serving:**
    The API server (`src/splatnlp/serve/app.py`) loads model artifacts from URLs specified by environment variables. See `src/splatnlp/serve/load_model.py` for details (e.g., `VOCAB_URL`, `MODEL_URL`, `PARAMS_URL`, `WEAPON_VOCAB_URL`, `INFO_URL` or `DO_SPACES_ML_ENDPOINT`/`DO_SPACES_ML_DIR`).

## Usage Examples

*(Note: Adjust paths and parameters according to your setup.)*

**1. Run the Preprocessing Pipeline:**
*(Downloads data from stat.ink, processes it, and saves partitioned output)*
```bash
# Example: Persist raw downloads, save processed CSV to data/weapon_partitioned.csv
python src/splatnlp/preprocessing/pipeline.py --base_path data/ --persist
```
*(The final output suitable for the model training CLI might need further consolidation/tokenization based on the CLI's expectations - refer to the CLI code)*

**2. Train the Main `SetCompletionModel`:**
```bash
# Example: Train using a tokenized TSV file
python src/splatnlp/model/cli.py \
    --data_path path/to/your/tokenized_data.tsv \
    --vocab_path path/to/your/vocab.json \
    --weapon_vocab_path path/to/your/weapon_vocab.json \
    --output_dir ./trained_model \
    --num_epochs 20 \
    --batch_size 128 \
    --learning_rate 0.0005 \
    --embedding_dim 64 \
    --hidden_dim 512 \
    --num_layers 3 \
    --dropout 0.2 \
    --verbose True
```

**3. Train a Sparse Autoencoder (SAE):**
```bash
# Example: Train an SAE on activations from a pretrained SetCompletionModel
python src/splatnlp/monosemanticity_train/cli.py \
    --data_path path/to/your/tokenized_data.tsv \
    --vocab_path path/to/your/vocab.json \
    --weapon_vocab_path path/to/your/weapon_vocab.json \
    --pretrained_model_path ./trained_model/model.pth \
    --output_dir ./trained_sae \
    --autoencoder_dim 1024 \
    --num_epochs 5 \
    --batch_size 64 \
    --learning_rate 0.0001
```

**4. Run the API Server:**
*(Requires model artifacts accessible via URLs configured through environment variables)*
> NOTE: THIS HAS NO SECURITY MEASURES, IT IS DESIGNED TO BE USED IN A LOCAL ENVIRONMENT OR SILOED OFF IN A CONTAINERIZED ENVIRONMENT WITH STRICT NETWORKING POLICIES. DO NOT DEPLOY THIS IN A PRODUCTION ENVIRONMENT WITHOUT ADDING THE APPROPRIATE SECURITY MEASURES.

```bash
# Ensure environment variables for model URLs are set
uvicorn splatnlp.serve.app:app --host 0.0.0.0 --port 9000 --reload
```

**5. Query the API Endpoint:**
```bash
# Example: Get predictions for a partial build (Splattershot Pro - ID 310)
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
```

## Architecture

(A diagram and detailed explanation of the `SetCompletionModel` architecture can be found in the blog post linked above.)

## License

This project is licensed under the GNU General Public License v3.0 (GPL-3.0).
See the [LICENSE](LICENSE) file for details.
