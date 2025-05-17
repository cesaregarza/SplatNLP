import torch
from splatnlp.model.models import SetCompletionModel


def test_forward_output_shape():
    ability_vocab_size = 5
    weapon_vocab_size = 2
    model = SetCompletionModel(
        vocab_size=ability_vocab_size,
        weapon_vocab_size=weapon_vocab_size,
        embedding_dim=8,
        hidden_dim=8,
        output_dim=ability_vocab_size,
        num_layers=1,
        num_heads=2,
        num_inducing_points=2,
        use_layer_norm=False,
        dropout=0.0,
        pad_token_id=0,
    )

    ability_tokens = torch.tensor([[1, 2, 0], [3, 4, 0]], dtype=torch.long)
    weapon_tokens = torch.tensor([[0], [1]], dtype=torch.long)
    mask = ability_tokens == 0

    output = model(ability_tokens, weapon_tokens, key_padding_mask=mask)
    assert output.shape == (2, ability_vocab_size)


def test_masked_mean_ignores_padding():
    model = SetCompletionModel(
        vocab_size=5,
        weapon_vocab_size=1,
        embedding_dim=4,
        hidden_dim=4,
        output_dim=2,
        num_layers=1,
        num_heads=2,
        num_inducing_points=2,
        use_layer_norm=False,
        dropout=0.0,
        pad_token_id=0,
    )

    x = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [100.0, 100.0]]])
    mask_none = torch.tensor([[False, False, False]])
    mask_last = torch.tensor([[False, False, True]])

    mean_all = model.masked_mean(x, mask_none)
    mean_masked = model.masked_mean(x, mask_last)

    expected = torch.tensor([[2.0, 3.0]])
    assert torch.allclose(mean_masked, expected)
    assert not torch.allclose(mean_all, expected)

