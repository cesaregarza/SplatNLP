# AGENT GUIDELINES

This repository uses **Poetry** for dependency management and testing.
When working on this project:

* Install dependencies with `poetry install --with dev`.
* Run the test suite using `poetry run pytest -q`.
* Apply code formatting with `poetry run black .` and sort imports with `poetry run isort .`.
  The configured line length for both tools is 80 characters (see `pyproject.toml`).
* The codebase targets Python 3.10+, and the CI tests run on Python 3.11.

Follow these steps before submitting changes.

## Docs invariants (do not change)

- Do not change the canonical blog link in `README.md`:
  `https://cegarza.com/splatgpt-part-1/`
  (If you think it should change, ask first.)

## Ultra model + SAE quickstart (avoid timeouts)

Running the ultra model with the SAE is heavy; use the activation server so
experiments don't re-load Zarr each time.

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

## Fast beam search recipe (ultra, with SAE trace)
1) Load model + SAE on CPU (fast enough for single runs):
```bash
poetry run python - <<'PY'
import json, torch
from pathlib import Path
from splatnlp.model.models import SetCompletionModel
from splatnlp.monosemantic_sae.models import SparseAutoencoder
from splatnlp.monosemantic_sae.hooks import register_hooks
from splatnlp.utils.infer import build_predict_abilities
from splatnlp.utils.reconstruct.allocator import Allocator
from splatnlp.utils.reconstruct.beam_search import reconstruct_build

root = Path('/root/dev/SplatNLP')
vocab = json.load((root/'saved_models/dataset_v0_2_full/vocab.json').open())
weapon_vocab = json.load((root/'saved_models/dataset_v0_2_full/weapon_vocab.json').open())

model = SetCompletionModel(len(vocab), len(weapon_vocab), 32, 512, len(vocab),
                           num_layers=3, num_heads=8, num_inducing_points=32,
                           use_layer_norm=True, dropout=0.0, pad_token_id=vocab['<PAD>'])
model.load_state_dict(torch.load(root/'saved_models/dataset_v0_2_super/clean_slate.pth',
                                 map_location='cpu'))
model.eval()

sae = SparseAutoencoder(512, expansion_factor=48.0, l1_coefficient=0.0, target_usage=0.0, usage_coeff=0.0)
sae.load_state_dict(torch.load(root/'sae_runs/run_20250704_191557/sae_model_final.pth',
                               map_location='cpu'))
sae.eval()
hook, handle = register_hooks(model, sae_model=sae, bypass=False, no_change=True)

predict_fn_factory = build_predict_abilities(vocab, weapon_vocab, pad_token='<PAD>',
                                             hook=None, device=torch.device('cpu'), output_type='dict')
def predict_fn(current_tokens, weapon_id):
    probs = predict_fn_factory(model, current_tokens, weapon_id)
    acts = hook.last_h_post.detach().cpu().flatten().tolist() if hook.last_h_post is not None else None
    return probs, acts

allocator = Allocator()
builds, traces = reconstruct_build(predict_fn=predict_fn,
                                   weapon_id='weapon_id_8000',
                                   initial_context=['<NULL>'],
                                   allocator=allocator,
                                   beam_size=5, max_steps=6, top_k=1,
                                   record_traces=True)
handle.remove()

# Compact trace summary with top preds and top 10 features per step
summary = []
seen = set()
for fr in traces[0]:
    caps = sorted(fr.partial_caps.keys())
    added = sorted(set(caps) - seen); seen.update(caps)
    top_preds = [(t, round(p,4)) for t,p in sorted(fr.logits.items(), key=lambda x:x[1], reverse=True) if not t.startswith('<')][:8]
    acts = fr.activations
    top_feats = []
    if acts is not None:
        top_idx = sorted(range(len(acts)), key=lambda i: acts[i], reverse=True)[:10]
        top_feats = [(int(i), round(float(acts[i]),4)) for i in top_idx]
    summary.append({'step': fr.step, 'beam_rank': fr.beam_rank,
                    'capstones': caps, 'added_this_step': added,
                    'top_preds': top_preds, 'top_features': top_feats})

out = {'build': None if not builds else {'mains': builds[0].mains,
                                         'subs': dict(builds[0].subs),
                                         'total_ap': builds[0].total_ap,
                                         'achieved_ap': builds[0].achieved_ap},
       'trace_summary': summary}
out_path = root/'tmp_results/beam_trace.json'
out_path.write_text(json.dumps(out, indent=2))
print('Wrote', out_path)
PY
```
2) If `/root/dev/SplatNLP/outputs` points to a read-only mount, write to `tmp_results/`.
3) For faster repeated runs, start the activation server once (see above) so SAE hooks can fetch activations without reloading Zarr. Use `beam_size=5`, `max_steps=6–8` for speed.***
