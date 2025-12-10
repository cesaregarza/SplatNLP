# AGENT GUIDELINES

This repository uses **Poetry** for dependency management and testing.
When working on this project:

* Install dependencies with `poetry install --with dev`.
* Run the test suite using `poetry run pytest -q`.
* Apply code formatting with `poetry run black .` and sort imports with `poetry run isort .`.
  The configured line length for both tools is 80 characters (see `pyproject.toml`).
* The codebase targets Python 3.10+, and the CI tests run on Python 3.11.

Follow these steps before submitting changes.

## Ultra model + SAE quickstart (avoid timeouts)

Running the ultra model with the SAE is heavy; use the activation server so
experiments don’t re-load Zarr each time.

1) Start the activation server (once per session):
```bash
poetry run uvicorn splatnlp.mechinterp.server.activation_server:app \
  --host 127.0.0.1 --port 8765
```
   - It loads from `/mnt/e/activations_ultra_efficient` and caches features.
   - Leave it running; clients auto-detect it.

2) Ensure ultra checkpoints are local (already in repo):
   - Primary: `saved_models/dataset_v0_2_super/clean_slate.pth`
   - SAE: `sae_runs/run_20250704_191557/sae_model_final.pth`

3) Example: forward pass with SAE (CPU-friendly):
```python
import json, torch, torch.nn.functional as F
from splatnlp.model.models import SetCompletionModel
from splatnlp.monosemantic_sae.models import SparseAutoencoder

vocab = json.load(open('saved_models/dataset_v0_2_full/vocab.json'))
weapon_vocab = json.load(open('saved_models/dataset_v0_2_full/weapon_vocab.json'))

model = SetCompletionModel(len(vocab), len(weapon_vocab), 32, 512, len(vocab),
                           num_layers=3, num_heads=8, num_inducing_points=32,
                           use_layer_norm=True, dropout=0.0,
                           pad_token_id=vocab['<PAD>'])
model.load_state_dict(torch.load('saved_models/dataset_v0_2_super/clean_slate.pth',
                                 map_location='cpu'))
model.eval()

sae = SparseAutoencoder(512, expansion_factor=48.0,
                        l1_coefficient=0.0, target_usage=0.0, usage_coeff=0.0)
sae.load_state_dict(torch.load('sae_runs/run_20250704_191557/sae_model_final.pth',
                               map_location='cpu'))
sae.eval()

null = vocab['<NULL>']
weapon = weapon_vocab['weapon_id_8000']  # Stamper
tokens = torch.tensor([[null]])
weapons = torch.tensor([[weapon]])
mask = torch.zeros_like(tokens, dtype=torch.bool)

with torch.no_grad():
    emb = model.ability_embedding(tokens) + model.weapon_embedding(weapons).expand_as(tokens.unsqueeze(-1))
    x = model.input_proj(emb)
    for layer in model.transformer_layers:
        x = layer(x, key_padding_mask=mask)
    masked = model.masked_mean(x, key_padding_mask=mask)
    logits = model.output_layer(masked).squeeze(0)
    _, h_post = sae.encode(masked)
    decoder_norm = F.normalize(sae.decoder.weight, dim=0)
    influence = model.output_layer.weight @ decoder_norm
```

4) Faster activation fetches:
   - Use server client (auto): `ctx = load_context('ultra')` → uses server if up.
   - Endpoint supports `limit`, `sample_frac`, `include_abilities=false` to reduce payloads.

If fetches still hang, bump command timeouts once per session to let the first
Zarr load finish, then it should be warm and fast.
