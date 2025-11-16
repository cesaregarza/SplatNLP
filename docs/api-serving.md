# API Serving

The project includes a FastAPI server for real-time inference. This allows external applications to get gear recommendations from the trained model.

## Architecture

```
Client Request
      ↓
FastAPI Endpoint
      ↓
Tokenization (raw AP → tokens)
      ↓
Model Inference (forward pass)
      ↓
JSON Response (probabilities)
```

## API Overview

**Framework**: FastAPI with Pydantic validation

**Endpoints**:
- `POST /infer`: Get predictions for a partial build
- `GET /infer`: Test endpoint with hardcoded input

**CORS**: Enabled for all origins (development mode)

## Request Format

```json
{
  "target": {
    "ink_saver_main": 6,
    "run_speed_up": 12,
    "intensify_action": 10
  },
  "weapon_id": 310
}
```

**target**: Dictionary of ability names to AP values. AP is in raw units (1 main = 10, 1 sub = 3). Empty dict requests a baseline build.

**weapon_id**: Numeric weapon identifier (matches stat.ink IDs).

## Response Format

```json
{
  "predictions": [
    ["swim_speed_up", 0.95],
    ["ink_saver_main", 0.88],
    ["quick_respawn", 0.72],
    ...
  ],
  "splatgpt_info": {
    "model_version": "full",
    "training_date": "2024-01-15",
    "vocab_size": 74
  },
  "api_version": "0.2.0",
  "inference_time": 0.05
}
```

**predictions**: List of (ability_token, probability) tuples. Higher probability means the model thinks this ability should be in the build. Sorted by token ID, not probability.

**splatgpt_info**: Metadata about the model (version, training info).

**inference_time**: Seconds taken for the forward pass. Useful for monitoring latency.

## Model Loading

At startup, the server loads model artifacts from URLs:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    vocab, weapon_vocab, pad_token_id, model_info, model = load_from_env()
    yield
```

The `lifespan` context manager ensures the model is loaded before serving requests.

**Required Environment Variables**:
```bash
VOCAB_URL=https://storage.example.com/vocab.json
WEAPON_VOCAB_URL=https://storage.example.com/weapon_vocab.json
MODEL_URL=https://storage.example.com/model.pth
PARAMS_URL=https://storage.example.com/model_params.json
INFO_URL=https://storage.example.com/model_info.json
```

Or use base path:
```bash
DO_SPACES_ML_ENDPOINT=https://storage.example.com
DO_SPACES_ML_DIR=splatgpt/v1
```

The server downloads all artifacts into memory at startup. No disk I/O during inference.

## Inference Pipeline

### 1. Tokenization

Raw AP values get converted to tokens:

```python
def tokenize_build(abilities: dict[str, int]) -> list[str]:
    # ink_saver_main: 6 → ["ink_saver_main_3", "ink_saver_main_6"]
    # main_only abilities → ["ability_name"]
    # Empty dict → ["<NULL>"]
```

Uses the same bucketing logic as preprocessing. Thresholds: [3, 6, 12, 15, 21, 29, 38, 51, 57].

If no abilities provided, uses a special NULL token to request baseline builds.

### 2. Forward Pass

```python
def inference(model, target, weapon_id, vocab, ...):
    input_tokens = [vocab[token] for token in target]
    input_weapons = [weapon_vocab[weapon_id]]

    with torch.no_grad():
        outputs = model(input_tokens, input_weapons, key_padding_mask)
        preds = torch.sigmoid(outputs).squeeze()

    return [(token, prob) for token, prob in zip(vocab, preds)]
```

**Key points**:
- `torch.no_grad()`: No gradient computation (faster, less memory)
- `torch.sigmoid()`: Convert logits to probabilities
- CPU inference: Model loaded on CPU for compatibility

### 3. Response Construction

Maps token indices back to readable ability names:
```python
inv_vocab = {v: k for k, v in vocab.items()}
predictions = [(inv_vocab[i], float(pred)) for i, pred in enumerate(preds)]
```

## Running the Server

```bash
uvicorn splatnlp.serve.app:app --host 0.0.0.0 --port 9000 --reload
```

**Flags**:
- `--host 0.0.0.0`: Listen on all interfaces
- `--port 9000`: Port number
- `--reload`: Auto-reload on code changes (dev mode)

For production, remove `--reload` and add workers:
```bash
uvicorn splatnlp.serve.app:app --host 0.0.0.0 --port 9000 --workers 4
```

## Example Usage

**cURL**:
```bash
curl -X POST "http://localhost:9000/infer" \
     -H "Content-Type: application/json" \
     -d '{
          "target": {
            "swim_speed_up": 12,
            "ninja_squid": 1
          },
          "weapon_id": 310
        }'
```

**Python**:
```python
import requests

response = requests.post(
    "http://localhost:9000/infer",
    json={
        "target": {"swim_speed_up": 12},
        "weapon_id": 310,
    }
)
predictions = response.json()["predictions"]
top_5 = sorted(predictions, key=lambda x: x[1], reverse=True)[:5]
```

**Baseline build** (no abilities specified):
```bash
curl -X POST "http://localhost:9000/infer" \
     -H "Content-Type: application/json" \
     -d '{"target": {}, "weapon_id": 310}'
```

## Performance Characteristics

**Latency**: ~50ms per request on CPU (model forward pass dominates)

**Memory**: ~350MB for model weights + vocab

**Throughput**: Single-threaded, 20 req/s. Multi-worker scales linearly.

**Cold start**: 5-10 seconds (downloads model artifacts from URLs)

## Security Considerations

The current implementation is designed for local/trusted environments:

- CORS allows all origins
- No authentication
- No rate limiting
- No input validation beyond type checking

**For production**:
- Add authentication (API keys, OAuth)
- Restrict CORS origins
- Implement rate limiting
- Add input sanitization
- Use HTTPS
- Monitor for abuse

## Deployment Options

**Local development**:
```bash
python -m splatnlp.serve.app
```

**Docker**:
```dockerfile
FROM python:3.11
COPY . /app
WORKDIR /app
RUN pip install poetry && poetry install --no-dev
CMD ["uvicorn", "splatnlp.serve.app:app", "--host", "0.0.0.0", "--port", "9000"]
```

**Cloud run** (serverless):
- Model loading on each cold start is slow
- Consider persistent instances or pre-warming

**Kubernetes**:
- Use readiness probes to wait for model loading
- Horizontal pod autoscaling for traffic spikes

## Monitoring

The server logs:
- Request details (target, weapon_id)
- Inference time
- Full response

Example log output:
```
2024-01-15 10:30:45 - INFO - app.py - Received inference request: target={'swim_speed_up': 12} weapon_id=310
2024-01-15 10:30:45 - INFO - inference.py - Starting inference
2024-01-15 10:30:45 - INFO - inference.py - Finished inference in 0.05s
2024-01-15 10:30:45 - INFO - app.py - Returning response: predictions=[...]
```

For production, integrate with:
- Prometheus/Grafana for metrics
- ELK stack for log aggregation
- Distributed tracing (Jaeger/Zipkin)

## Code Location

FastAPI app: `src/splatnlp/serve/app.py`
Model loading: `src/splatnlp/serve/load_model.py`
Inference logic: `src/splatnlp/serve/inference.py`
Tokenization: `src/splatnlp/serve/tokenize.py`
