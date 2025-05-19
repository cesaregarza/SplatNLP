import pytest
import torch

from splatnlp.model.models import SetCompletionModel
from splatnlp.monosemantic_sae.hooks import register_hooks
from splatnlp.monosemantic_sae.models import SparseAutoencoder


def make_models():
    torch.manual_seed(0)
    model = SetCompletionModel(
        vocab_size=5,
        weapon_vocab_size=2,
        embedding_dim=4,
        hidden_dim=4,
        output_dim=4,
        num_layers=1,
        num_heads=1,
        num_inducing_points=2,
        use_layer_norm=False,
        dropout=0.0,
        pad_token_id=0,
    )
    sae = SparseAutoencoder(input_dim=4, expansion_factor=1)
    with torch.no_grad():
        sae.encoder.weight.copy_(torch.eye(4))
        sae.encoder.bias.zero_()
        sae.decoder.weight.copy_(torch.eye(4))
        sae.decoder_bias.zero_()
    return model, sae


def run_model(model, ability_tokens, weapon_token):
    key_padding_mask = ability_tokens == 0
    with torch.no_grad():
        return model(ability_tokens, weapon_token, key_padding_mask=key_padding_mask)


def test_hook_bypass_returns_original_output():
    model, sae = make_models()
    abilities = torch.tensor([[1, 2, 3]], dtype=torch.long)
    weapon = torch.tensor([[1]], dtype=torch.long)

    baseline = run_model(model, abilities, weapon)

    hook, handle = register_hooks(model, sae, bypass=True)
    out = run_model(model, abilities, weapon)

    handle.remove()

    assert torch.allclose(out, baseline)
    assert hook.last_in is None
    assert hook.last_x_recon is None


def test_hook_edit_neuron_and_out_of_range():
    model, sae = make_models()
    abilities = torch.tensor([[1, 2, 3]], dtype=torch.long)
    weapon = torch.tensor([[1]], dtype=torch.long)

    hook, handle = register_hooks(model, sae, bypass=False, no_change=True)
    baseline = run_model(model, abilities, weapon)

    # Editing within range changes output
    hook.update_neuron(0, 2.0)
    edited = run_model(model, abilities, weapon)
    assert not torch.allclose(edited, baseline)

    # Out of range index raises
    hook.update_neuron(10, 1.0)
    with pytest.raises(IndexError):
        run_model(model, abilities, weapon)

    handle.remove()
