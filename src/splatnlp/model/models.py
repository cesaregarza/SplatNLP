from typing import Optional

import torch
import torch.nn as nn


class MultiheadAttentionBlock(nn.Module):
    """Multihead Attention Block with optional layer normalization.

    This module implements a Multihead Attention Block that can be used in
    transformer-like architectures. It includes a multihead attention layer
    followed by a feedforward network, with optional layer normalization.

    Args:
        embed_dim (int): The embedding dimension of the input.
        num_heads (int): The number of attention heads.
        use_layer_norm (bool, optional): Whether to use layer normalization.
            Defaults to False.
        dropout (float, optional): Dropout rate. Defaults to 0.0.

    Attributes:
        mha (nn.MultiheadAttention): The multihead attention layer.
        feedforward (nn.Sequential): The feedforward network.
        use_layer_norm (bool): Whether layer normalization is used.
        layer_norm1 (nn.LayerNorm): First layer normalization layer.
        layer_norm2 (nn.LayerNorm): Second layer normalization layer.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        use_layer_norm: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.use_layer_norm = use_layer_norm
        if self.use_layer_norm:
            self.layer_norm1 = nn.LayerNorm(embed_dim)
            self.layer_norm2 = nn.LayerNorm(embed_dim)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass of the MultiheadAttentionBlock.

        Args:
            query (torch.Tensor): The query tensor.
            key (torch.Tensor): The key tensor.
            key_padding_mask (Optional[torch.Tensor], optional): The key
                padding mask. Defaults to None.

        Returns:
            torch.Tensor: The output tensor after attention and feedforward
                layers.
        """
        attn_output, _ = self.mha(
            query, key, key, key_padding_mask=key_padding_mask
        )
        x = query + attn_output
        if self.use_layer_norm:
            x = self.layer_norm1(x)
        ff_output = self.feedforward(x)
        output = x + ff_output
        if self.use_layer_norm:
            output = self.layer_norm2(output)
        return output


class SelfAttentionBlock(nn.Module):
    """Self-Attention Block using MultiheadAttentionBlock.

    This module implements a Self-Attention Block by using a
    MultiheadAttentionBlock where the query and key are the same input.

    Args:
        embed_dim (int): The embedding dimension of the input.
        num_heads (int): The number of attention heads.
        use_layer_norm (bool, optional): Whether to use layer normalization.
            Defaults to False.
        dropout (float, optional): Dropout rate. Defaults to 0.0.

    Attributes:
        mab (MultiheadAttentionBlock): The underlying multihead attention block.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        use_layer_norm: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.mab = MultiheadAttentionBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            use_layer_norm=use_layer_norm,
            dropout=dropout,
        )

    def forward(
        self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass of the SelfAttentionBlock.

        Args:
            x (torch.Tensor): The input tensor.
            key_padding_mask (Optional[torch.Tensor], optional): The key
                padding mask. Defaults to None.

        Returns:
            torch.Tensor: The output tensor after self-attention.
        """
        return self.mab(x, x, key_padding_mask=key_padding_mask)


class InducedSetAttentionBlock(nn.Module):
    """Induced Set Attention Block.

    This module implements an Induced Set Attention Block, which uses inducing
    points to compute attention over a set of elements.

    Args:
        embed_dim (int): The embedding dimension of the input.
        num_heads (int): The number of attention heads.
        num_inducing_points (int): The number of inducing points.
        use_layer_norm (bool, optional): Whether to use layer normalization.
            Defaults to False.
        dropout (float, optional): Dropout rate. Defaults to 0.0.

    Attributes:
        inducing_points (nn.Parameter): Learnable inducing points.
        mab1 (MultiheadAttentionBlock): First multihead attention block.
        mab2 (MultiheadAttentionBlock): Second multihead attention block.
        layer_norm (nn.LayerNorm): Layer normalization.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_inducing_points: int,
        use_layer_norm: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.inducing_points = nn.Parameter(
            torch.randn(num_inducing_points, embed_dim)
        )
        nn.init.xavier_uniform_(self.inducing_points)
        self.mab1 = MultiheadAttentionBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            use_layer_norm=use_layer_norm,
            dropout=dropout,
        )
        self.mab2 = MultiheadAttentionBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            use_layer_norm=use_layer_norm,
            dropout=dropout,
        )
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(
        self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass of the InducedSetAttentionBlock.

        Args:
            x (torch.Tensor): The input tensor.
            key_padding_mask (Optional[torch.Tensor], optional): The key
                padding mask. Defaults to None.

        Returns:
            torch.Tensor: The output tensor after induced set attention.
        """
        batch_size = x.size(0)
        inducing_points = self.inducing_points.unsqueeze(0).expand(
            batch_size, -1, -1
        )
        h = self.mab1(inducing_points, x, key_padding_mask=key_padding_mask)
        # Since `h` does not contain padding, we don't need to pass `key_padding_mask` here
        out = self.mab2(x, h)
        out = out + x
        out = self.layer_norm(out)
        return out


class PoolingMultiheadAttention(nn.Module):
    """Pooling Multihead Attention module.

    This module implements a Pooling Multihead Attention mechanism, which uses
    learnable seed vectors to pool information from the input set.

    Args:
        embed_dim (int): The embedding dimension of the input.
        num_heads (int): The number of attention heads.
        num_seeds (int): The number of seed vectors for pooling.
        use_layer_norm (bool, optional): Whether to use layer normalization.
            Defaults to False.
        dropout (float, optional): Dropout rate. Defaults to 0.0.

    Attributes:
        seed_vectors (nn.Parameter): Learnable seed vectors for pooling.
        mab (MultiheadAttentionBlock): The multihead attention block.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_seeds: int,
        use_layer_norm: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.seed_vectors = nn.Parameter(torch.randn(num_seeds, embed_dim))
        nn.init.xavier_uniform_(self.seed_vectors)
        self.mab = MultiheadAttentionBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            use_layer_norm=use_layer_norm,
            dropout=dropout,
        )
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass of the PoolingMultiheadAttention module.

        Args:
            x (torch.Tensor): The input tensor.
            key_padding_mask (Optional[torch.Tensor], optional): The key
                padding mask. Defaults to None.

        Returns:
            torch.Tensor: The output tensor after pooling attention.
        """
        batch_size = x.size(0)
        seed_vectors = self.seed_vectors.unsqueeze(0).expand(batch_size, -1, -1)
        x = self.mab(seed_vectors, x, key_padding_mask=key_padding_mask)
        out = self.out_proj(x)
        return out


class SetTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_seeds: int,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_inducing_points: int = 32,
        use_layer_norm: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        # Input projection if input_dim != embed_dim
        if input_dim != embed_dim:
            self.input_proj = nn.Linear(input_dim, embed_dim)
        else:
            self.input_proj = nn.Identity()

        # Define encoder layers individually
        self.enc_layer1 = InducedSetAttentionBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_inducing_points=num_inducing_points,
            use_layer_norm=use_layer_norm,
            dropout=dropout,
        )
        self.enc_layer2 = InducedSetAttentionBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_inducing_points=num_inducing_points,
            use_layer_norm=use_layer_norm,
            dropout=dropout,
        )

        # Define decoder layers individually
        self.dec_layer1 = PoolingMultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_seeds=num_seeds,
            use_layer_norm=use_layer_norm,
            dropout=dropout,
        )
        self.dec_layer2 = SelfAttentionBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            use_layer_norm=use_layer_norm,
            dropout=dropout,
        )
        self.dec_layer3 = SelfAttentionBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            use_layer_norm=use_layer_norm,
            dropout=dropout,
        )

    def forward(
        self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.enc_layer1(x, key_padding_mask=key_padding_mask) + x
        x = self.enc_layer2(x, key_padding_mask=key_padding_mask) + x
        x = self.dec_layer1(x, key_padding_mask=key_padding_mask)
        x = self.dec_layer2(x) + x
        x = self.dec_layer3(x) + x
        return x


class SetTransformerLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_inducing_points: int,
        num_layers_mlp: int = 2,
        mlp_hidden_dim: int | None = None,
        use_layer_norm: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.set_transformer = SetTransformer(
            input_dim=embed_dim,
            num_seeds=num_inducing_points,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_inducing_points=num_inducing_points,
            use_layer_norm=use_layer_norm,
            dropout=dropout,
        )

        # Cross-attention layer to project back to original sequence length
        self.cross_attention = MultiheadAttentionBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            use_layer_norm=use_layer_norm,
            dropout=dropout,
        )

        # MLP with configurable number of layers
        if mlp_hidden_dim is None:
            mlp_hidden_dim = embed_dim * 4

        layers = []
        for i in range(num_layers_mlp):
            if i == 0:
                layers.append(nn.Linear(embed_dim, mlp_hidden_dim))
            elif i == num_layers_mlp - 1:
                layers.append(nn.Linear(mlp_hidden_dim, embed_dim))
            else:
                layers.append(nn.Linear(mlp_hidden_dim, mlp_hidden_dim))

            if i < num_layers_mlp - 1:  # No activation after the last layer
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))

        self.feedforward = nn.Sequential(*layers)

        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm1 = nn.LayerNorm(embed_dim)
            self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Apply SetTransformer
        set_transformer_output = self.set_transformer(
            x, key_padding_mask=key_padding_mask
        )

        # Project SetTransformer output back to input sequence length via cross-attention
        projected_output = self.cross_attention(
            query=x,
            key=set_transformer_output,
            key_padding_mask=None,
        )

        # First residual connection and layer norm
        x = x + self.dropout(projected_output)
        if self.use_layer_norm:
            x = self.layer_norm1(x)

        # Feedforward network
        ff_output = self.feedforward(x)

        # Second residual connection and layer norm
        x = x + self.dropout(ff_output)
        if self.use_layer_norm:
            x = self.layer_norm2(x)

        return x


class SetCompletionModel(nn.Module):
    """A Set Completion Model using Set Transformer architecture.

    This model takes a set of tokens and a single weapon token as input,
    processes them through a stack of Set Transformer layers, and outputs a
    fixed-size representation for set completion tasks.

    The architecture consists of:
    1. Two embedding layers: one for ability tokens and one for weapon tokens
    2. An optional input projection layer if embedding_dim != hidden_dim
    3. A stack of SetTransformerLayers with InducedSetAttentionBlocks
    4. A masked mean pooling operation to aggregate set information
    5. An output layer to produce the final representation

    Args:
        vocab_size (int): Size of the ability token vocabulary
        weapon_vocab_size (int): Size of the weapon token vocabulary
        embedding_dim (int): Dimension of the token embeddings
        hidden_dim (int): Dimension of the hidden layers in the transformer
        output_dim (int): Dimension of the output representation
        num_layers (int): Number of SetTransformerLayers
        num_heads (int): Number of attention heads in each layer
        num_inducing_points (int): Number of inducing points for
            InducedSetAttentionBlock
        use_layer_norm (bool): Whether to use layer normalization
        dropout (float): Dropout rate
        pad_token_id (int): ID of the padding token
    """

    def __init__(
        self,
        vocab_size: int,
        weapon_vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        num_layers_mlp: int = 2,
        num_heads: int = 4,
        num_inducing_points: int = 32,
        use_layer_norm: bool = False,
        dropout: float = 0.0,
        pad_token_id: int = None,
    ):
        super().__init__()
        self.ability_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_token_id,
        )
        self.weapon_embedding = nn.Embedding(
            num_embeddings=weapon_vocab_size,
            embedding_dim=embedding_dim,
        )

        if embedding_dim != hidden_dim:
            self.input_proj = nn.Linear(embedding_dim, hidden_dim)
        else:
            self.input_proj = nn.Identity()

        layers = []
        for _ in range(num_layers):
            layers.append(
                SetTransformerLayer(
                    embed_dim=hidden_dim,
                    num_heads=num_heads,
                    num_inducing_points=num_inducing_points,
                    use_layer_norm=use_layer_norm,
                    dropout=dropout,
                    num_layers_mlp=num_layers_mlp,
                )
            )
        self.transformer_layers = nn.ModuleList(layers)

        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def masked_mean(
        self, x: torch.Tensor, key_padding_mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute the masked mean of the input tensor along the sequence
            dimension.

        This method calculates the mean of the input tensor while ignoring
        padded positions specified by the key_padding_mask.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length,
                hidden_dim)
            key_padding_mask (torch.Tensor): Boolean mask indicating padded
                positions (True for pad tokens, False for non-pad tokens)

        Returns:
            torch.Tensor: Masked mean of the input tensor, shape (batch_size,
                hidden_dim)
        """
        valid_mask = (~key_padding_mask).unsqueeze(2).float()
        x = x * valid_mask
        sum_x = x.sum(dim=1)
        lengths = valid_mask.sum(dim=1)
        lengths = lengths.clamp(min=1)
        x_mean = sum_x / lengths
        return x_mean

    def forward(
        self,
        ability_tokens: torch.Tensor,
        weapon_token: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        ability_embeddings = self.ability_embedding(ability_tokens)
        weapon_embeddings = self.weapon_embedding(weapon_token).expand_as(
            ability_embeddings
        )
        embeddings = ability_embeddings + weapon_embeddings
        x = self.input_proj(embeddings)
        for layer in self.transformer_layers:
            x = layer(x, key_padding_mask=key_padding_mask)
        x = self.masked_mean(x, key_padding_mask)
        x = self.output_layer(x)
        return x

    @classmethod
    def load_from_checkpoint(
        cls, checkpoint_path: str, **model_args
    ) -> "SetCompletionModel":
        """Load the model from a .pth checkpoint file.

        Args:
            checkpoint_path (str): Path to the .pth checkpoint file
            **model_args: Additional arguments to initialize the model

        Returns:
            SetCompletionModel: Loaded model instance
        """
        model = cls(**model_args)
        model.load_state_dict(torch.load(checkpoint_path))
        return model
