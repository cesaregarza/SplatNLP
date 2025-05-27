"""Command for precomputing logit influences for SAE features."""

import argparse
import logging
import json # For loading vocab and writing JSONL output
from pathlib import Path
from typing import Dict, List, Any 

import torch
from tqdm import tqdm

from splatnlp.model.models import SetCompletionModel
from splatnlp.monosemantic_sae.models import SparseAutoencoder

logger = logging.getLogger(__name__)

def compute_logit_influences(
    sae_model: SparseAutoencoder,
    primary_model: SetCompletionModel,
    feature_id: int,
    vocab_list: List[str], # List of token names, index is token_id
    inv_vocab_map: Dict[str, int], # Map from token_name to token_id
    top_k: int = 5
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Compute top positive and negative logit influences for a feature.
    Output format includes rank and token_id.
    """
    if feature_id >= sae_model.decoder.weight.shape[0]:
        logger.warning(f"Feature ID {feature_id} out of bounds for SAE decoder. Skipping logit influences.")
        return {'positive': [], 'negative': []}

    decoder_vector = sae_model.decoder.weight[feature_id].detach().cpu()
    output_layer_weight = primary_model.output_layer.weight.detach().cpu()

    with torch.no_grad():
        logit_influences_tensor = torch.matmul(output_layer_weight, decoder_vector)

    actual_top_k = min(top_k, len(vocab_list))

    # Positive influences
    top_pos_values, top_pos_indices = torch.topk(logit_influences_tensor, actual_top_k)
    positive_influences = []
    for rank, (val, idx_tensor) in enumerate(zip(top_pos_values, top_pos_indices), 1):
        token_idx = idx_tensor.item()
        token_name = vocab_list[token_idx] if token_idx < len(vocab_list) else f"Token {token_idx}"
        positive_influences.append({
            'rank': rank,
            'token_id': token_idx,
            'token_name': token_name,
            'influence': float(val.item())
        })

    # Negative influences
    top_neg_values, top_neg_indices = torch.topk(-logit_influences_tensor, actual_top_k)
    negative_influences = []
    for rank, (val, idx_tensor) in enumerate(zip(top_neg_values, top_neg_indices), 1):
        token_idx = idx_tensor.item()
        token_name = vocab_list[token_idx] if token_idx < len(vocab_list) else f"Token {token_idx}"
        negative_influences.append({
            'rank': rank,
            'token_id': token_idx,
            'token_name': token_name,
            'influence': float(-val.item()) # Negate value back
        })
    
    return {'positive': positive_influences, 'negative': negative_influences}

# Main command function
def precompute_analytics_command(args: argparse.Namespace):
    """Precompute logit influences and save them to a JSONL file."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger.info("Starting precomputation of logit influences.")

    device = args.device if hasattr(args, 'device') and args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # --- Load Vocab ---
    logger.info("Loading vocabulary...")
    try:
        with open(args.vocab_path, 'r') as f:
            vocab_map = json.load(f) # token -> index map
        # Create list of tokens ordered by index for direct lookup
        vocab_list = [""] * len(vocab_map)
        for token, index in vocab_map.items():
            if 0 <= index < len(vocab_list):
                vocab_list[index] = token
            else:
                # This case should ideally not happen with a well-formed vocab
                logger.warning(f"Token index {index} for token '{token}' out of bounds for vocab_list size {len(vocab_list)}. Adjusting list size or ignoring token.")
                # If indices are sparse and large, this simple list creation might be inefficient or problematic.
                # Assuming vocab indices are dense and 0-based for now.
                if index >= len(vocab_list): # Dynamically extend if necessary, though not ideal
                    vocab_list.extend([""] * (index - len(vocab_list) + 1))
                    vocab_list[index] = token


        inv_vocab_map = {name: idx for idx, name in enumerate(vocab_list)} # Used if needed, but vocab_list is primary for index->name
        vocab_size = len(vocab_list)
        logger.info(f"Loaded vocab from {args.vocab_path}, size: {vocab_size}")
        if any(not token_name for token_name in vocab_list):
            logger.warning("Some token indices in vocab_list are empty strings. This might indicate issues with vocab construction if those indices are used.")

    except Exception as e:
        logger.error(f"Failed to load vocab from {args.vocab_path}: {e}. Cannot proceed.", exc_info=True)
        return

    # weapon_vocab_path is not used in this simplified script for logit influences only.
    # If it were needed for model loading (e.g. if SetCompletionModel requires weapon_vocab_size for init unconditionally)
    # it would need to be loaded. Assuming it's not strictly necessary for the parts of the model used here.
    # For safety, if SetCompletionModel requires weapon_vocab_size, let's load it.
    weapon_vocab_size = 0 # Default if not loaded
    if hasattr(args, 'weapon_vocab_path') and args.weapon_vocab_path:
        try:
            with open(args.weapon_vocab_path, 'r') as f:
                weapon_vocab_map = json.load(f)
            weapon_vocab_size = len(weapon_vocab_map)
            logger.info(f"Loaded weapon vocab from {args.weapon_vocab_path}, size: {weapon_vocab_size}")
        except Exception as e:
            logger.warning(f"Could not load weapon_vocab_path: {e}. Using default size {weapon_vocab_size}.")


    # --- Load Models ---
    logger.info("Loading models...")
    primary_model_params = {
        'vocab_size': vocab_size, 
        'weapon_vocab_size': weapon_vocab_size, # Pass even if 0, model should handle
        'embedding_dim': args.embedding_dim, 'hidden_dim': args.hidden_dim,
        'output_dim': vocab_size, 'num_layers': args.num_layers,
        'num_heads': args.num_heads, 'num_inducing_points': args.num_inducing_points,
        'use_layer_norm': True, 'dropout': 0.0,
        'pad_token_id': vocab_map.get(args.pad_token_name, 0) 
    }
    sae_model_params = {
        'input_dim': args.hidden_dim, 
        'expansion_factor': args.sae_expansion_factor
    }

    primary_model = SetCompletionModel(**primary_model_params)
    sae_model = SparseAutoencoder(**sae_model_params)
    
    try:
        primary_model.load_state_dict(torch.load(args.primary_model_checkpoint, map_location=device))
        sae_model.load_state_dict(torch.load(args.sae_model_checkpoint, map_location=device))
        primary_model.to(device).eval()
        sae_model.to(device).eval()
        logger.info("Models loaded and set to eval mode.")
    except Exception as e:
        logger.error(f"Error loading model checkpoints: {e}", exc_info=True)
        return

    num_sae_features = sae_model.decoder.weight.shape[0]
    logger.info(f"SAE model has {num_sae_features} features.")

    # --- Compute and Write Logit Influences ---
    output_file = Path(args.output_logit_influences)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Computing logit influences and writing to {output_file}...")
    with open(output_file, 'w') as f_out:
        for feature_idx in tqdm(range(num_sae_features), desc="Computing logit influences"):
            influences = compute_logit_influences(
                sae_model, primary_model, feature_idx, 
                vocab_list, inv_vocab_map, # Pass inv_vocab_map too
                args.top_k_logit_tokens
            )
            
            output_record = {
                "feature_id": feature_idx,
                "positive_influences": influences['positive'],
                "negative_influences": influences['negative']
            }
            f_out.write(json.dumps(output_record) + '\n')
    
    logger.info(f"Logit influences successfully written to {output_file}.")
    logger.info("Precomputation of logit influences finished.")


# Example CLI setup (to be integrated into a main CLI script later)
# def main_cli_example():
#     parser = argparse.ArgumentParser(description="Precompute logit influences for SAE features.")
    
#     # Model paths and vocabs
#     parser.add_argument("--primary-model-checkpoint", type=str, required=True, help="Path to primary model checkpoint (.pt).")
#     parser.add_argument("--sae-model-checkpoint", type=str, required=True, help="Path to SAE model checkpoint (.pt).")
#     parser.add_argument("--vocab-path", type=str, required=True, help="Path to vocabulary JSON file (token -> id map).")
#     # weapon_vocab_path is optional if SetCompletionModel can handle its absence or a default size.
#     parser.add_argument("--weapon-vocab-path", type=str, help="Path to weapon vocabulary JSON file (token -> id map).")
#     parser.add_argument("--pad-token-name", type=str, default="<PAD>", help="Name of the padding token in the vocab, for pad_token_id in model init.")

#     # Output
#     parser.add_argument("--output-logit-influences", type=str, required=True, help="Output JSONL file to save logit influences.")

#     # Model architecture parameters (must match the loaded checkpoints)
#     parser.add_argument("--embedding-dim", type=int, default=32, help="Primary model embedding dimension.")
#     parser.add_argument("--hidden-dim", type=int, default=512, help="Primary model hidden dimension (also SAE input dimension).")
#     parser.add_argument("--num-layers", type=int, default=3, help="Primary model number of layers.")
#     parser.add_argument("--num-heads", type=int, default=8, help="Primary model number of attention heads.")
#     parser.add_argument("--num-inducing-points", type=int, default=32, help="Primary model number of inducing points.")
#     parser.add_argument("--sae-expansion-factor", type=float, default=4.0, help="SAE expansion factor.")

#     # Analytics parameters
#     parser.add_argument("--top-k-logit-tokens", type=int, default=10, help="Number of top/bottom tokens for logit influence.")
    
#     parser.add_argument("--device", type=str, default=None, help="Device to use ('cuda', 'cpu'). Auto-detects if None.")

#     cli_args = parser.parse_args()
#     if cli_args.device is None: 
#         cli_args.device = "cuda" if torch.cuda.is_available() else "cpu"
        
#     precompute_analytics_command(cli_args)

# if __name__ == '__main__':
#     # For direct testing:
#     # Ensure you have dummy models and vocab files or provide paths to real ones.
#     # Example:
#     # test_args_list = [
#     #     "--primary-model-checkpoint", "dummy_primary.pt",
#     #     "--sae-model-checkpoint", "dummy_sae.pt",
#     #     "--vocab-path", "dummy_vocab.json",
#     #     "--weapon-vocab-path", "dummy_weapon_vocab.json",
#     #     "--output-logit-influences", "output_influences.jsonl",
#     #     # Add other params as needed, or ensure defaults are fine
#     # ]
#     # from types import SimpleNamespace
#     # # Simulate model creation and saving dummy checkpoints
#     # vocab_map_example = {"<PAD>": 0, "token_a": 1, "token_b": 2}
#     # weapon_vocab_map_example = {"weapon1": 0}
#     # with open("dummy_vocab.json", "w") as f: json.dump(vocab_map_example, f)
#     # with open("dummy_weapon_vocab.json", "w") as f: json.dump(weapon_vocab_map_example, f)
        
#     # primary_params_example = {'vocab_size': len(vocab_map_example), 'weapon_vocab_size': len(weapon_vocab_map_example), 'embedding_dim': 32, 'hidden_dim': 512, 'output_dim': len(vocab_map_example), 'num_layers': 3, 'num_heads': 8, 'num_inducing_points': 32, 'use_layer_norm': True, 'dropout': 0.0, 'pad_token_id': 0}
#     # sae_params_example = {'input_dim': 512, 'expansion_factor': 4.0}
#     # torch.save(SetCompletionModel(**primary_params_example).state_dict(), "dummy_primary.pt")
#     # torch.save(SparseAutoencoder(**sae_params_example).state_dict(), "dummy_sae.pt")
        
#     # # Create a dummy ArgumentParser to parse the list
#     # main_parser = argparse.ArgumentParser()
#     # # Add arguments to this parser matching those in main_cli_example (or the actual CLI setup)
#     # # This part needs to be done carefully to match the expected args by precompute_analytics_command
#     # # ... (add all arguments from the example CLI setup to main_parser) ...
#     # # Then parse: parsed_args = main_parser.parse_args(test_args_list)
#     # # And call: precompute_analytics_command(parsed_args)
#     pass
