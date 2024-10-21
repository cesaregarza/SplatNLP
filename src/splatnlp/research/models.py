from typing import Optional

import torch
from torch import nn

from splatnlp.model.models import SetCompletionModel, SetTransformerLayer


class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int):
        super().__init__()
        self.encoder = nn.Linear(input_dim, embed_dim)
        self.decoder = nn.Linear(embed_dim, input_dim)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.relu(self.encoder(x))
        decoded = self.decoder(encoded)
        return decoded


class ModifiedSetCompletionModel(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        pretrained_model: SetCompletionModel,
        autoencoder_dim: int,
    ):
        super().__init__()
        self.pretrained_model = pretrained_model
        # Freeze the parameters of the pretrained model
        for param in self.pretrained_model.parameters():
            param.requires_grad = False

        self.sparse_autoencoder = SparseAutoencoder(hidden_dim, autoencoder_dim)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        autoencoder_dim = kwargs.pop("autoencoder_dim")
        pretrained_model_path = kwargs.pop("pretrained_model_path")

        pretrained_model = SetCompletionModel.load_from_checkpoint(
            pretrained_model_path, **kwargs
        )
        return cls(kwargs["hidden_dim"], pretrained_model, autoencoder_dim)

    def forward(
        self,
        ability_tokens: torch.Tensor,
        weapon_token: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Use the pretrained model up to the last transformer layer
        ability_embeddings = self.pretrained_model.ability_embedding(
            ability_tokens
        )
        weapon_embeddings = self.pretrained_model.weapon_embedding(
            weapon_token
        ).expand_as(ability_embeddings)
        embeddings = ability_embeddings + weapon_embeddings
        x = self.pretrained_model.input_proj(embeddings)

        # Apply all transformer layers except the last one
        for layer in self.pretrained_model.transformer_layers[:-1]:
            x = layer(x, key_padding_mask=key_padding_mask)

        # Apply the last transformer layer
        last_layer_output = self.pretrained_model.transformer_layers[-1](
            x, key_padding_mask=key_padding_mask
        )

        # Apply SparseAutoencoder after the last SetTransformerLayer
        autoencoder_output = self.sparse_autoencoder(last_layer_output)

        # Combine the autoencoder output with the last layer output
        x = last_layer_output + autoencoder_output

        x = self.pretrained_model.masked_mean(x, key_padding_mask)
        x = self.pretrained_model.output_layer(x)
        return x
