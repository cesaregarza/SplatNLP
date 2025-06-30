import torch
from torch import nn
from torch.nn import functional as F


class SparseAutoencoder(nn.Module):
    """Sparse Autoencoder model implementation.

    This class defines a sparse autoencoder model that learns a compressed
    representation of input data. The model includes an encoder and decoder
    with a sparse regularization term to encourage the latent space to be
    sparse.
    """

    def __init__(
        self,
        input_dim: int,
        expansion_factor: float = 8,
        l1_coefficient: float = 1e-3,
        dead_neuron_threshold: float = 1e-6,
        dead_neuron_steps: int = 12500,
        target_usage: float = 0.05,
        usage_coeff: float = 1e-3,
    ) -> None:
        """Initialize the SparseAutoencoder model.

        Args:
            input_dim (int): Dimension of the input data.
            expansion_factor (float, optional): Factor by which the input
                dimension is expanded. Defaults to 8.
            l1_coefficient (float, optional): Coefficient for the L1
                regularization term. Defaults to 1e-3.
            dead_neuron_threshold (float, optional): Threshold for identifying
                dead neurons. Defaults to 1e-6.
            dead_neuron_steps (int, optional): Number of steps to count for
                dead neuron detection. Defaults to 12500.
            target_usage (float, optional): Target usage rate for sparse
                activations. Defaults to 0.05.
            usage_coeff (float, optional): Coefficient for the usage KL
                divergence term. Defaults to 1e-3.
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = int(input_dim * expansion_factor)
        self.l1_coefficient = l1_coefficient
        self.dead_neuron_threshold = dead_neuron_threshold
        self.dead_neuron_steps = dead_neuron_steps
        self.target_usage = target_usage
        self.usage_coeff = usage_coeff

        self.register_buffer("usage_ema", torch.zeros(self.hidden_dim))

        self.decoder_bias = nn.Parameter(torch.zeros(input_dim))

        self.encoder = nn.Linear(input_dim, self.hidden_dim)
        self.decoder = nn.Linear(self.hidden_dim, input_dim)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize the weights of the encoder and decoder."""
        nn.init.kaiming_uniform_(
            self.encoder.weight, mode="fan_in", nonlinearity="relu"
        )
        nn.init.kaiming_uniform_(
            self.decoder.weight, mode="fan_in", nonlinearity="relu"
        )
        nn.init.zeros_(self.encoder.bias)
        nn.init.zeros_(self.decoder_bias)
        with torch.no_grad():
            self.decoder.weight.data = F.normalize(
                self.decoder.weight.data, dim=0
            )

    def set_kl_coeff(self, coeff: float) -> None:
        """Update the usage/KL weight used inside `loss_fn`."""
        self.usage_coeff = float(coeff)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, input_dim]
                or [batch_size * seq_len, input_dim].

        Returns:
            tuple: A tuple containing:
                - h_pre: Pre-activation hidden layer activations of shape
                    [batch_size, hidden_dim] or
                    [batch_size * seq_len, hidden_dim].
                - h_post: Post-activation hidden layer activations of shape
                    [batch_size, hidden_dim] or
                    [batch_size * seq_len, hidden_dim].
        """
        x_centered = x - self.decoder_bias
        h_pre = self.encoder(x_centered)
        h_post = torch.clamp(F.relu(h_pre), min=0.0, max=6.0)
        return h_pre, h_post

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        """Decode the hidden activations.

        Args:
            h (torch.Tensor): Hidden activations of shape [batch_size, hidden_dim]
                or [batch_size * seq_len, hidden_dim].

        Returns:
            torch.Tensor: Decoded tensor of shape [batch_size, input_dim]
                or [batch_size * seq_len, input_dim].
        """
        with torch.no_grad():
            normalized_decoder_weights = F.normalize(self.decoder.weight, dim=0)
        return F.linear(h, normalized_decoder_weights) + self.decoder_bias

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the SparseAutoencoder.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, input_dim]
                or [batch_size * seq_len, input_dim].

        Returns:
            tuple: A tuple containing:
                - reconstruction: Reconstructed tensor of shape
                    [batch_size, input_dim] or
                    [batch_size * seq_len, input_dim].
                - hidden: Hidden layer activations of shape
                    [batch_size, hidden_dim] or
                    [batch_size * seq_len, hidden_dim].
        """
        _, h_post = self.encode(x)
        reconstruction = self.decode(h_post)
        return reconstruction, h_post

    def compute_loss(
        self,
        x: torch.Tensor,
        reconstruction: torch.Tensor,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute total loss with sparsity penalties.

        Args:
            x (torch.Tensor): Original input tensor.
            reconstruction (torch.Tensor): Reconstruction tensor from forward
                pass.
            hidden (torch.Tensor): Hidden layer activations from forward pass.

        Returns:
            tuple: (total loss, dictionary of loss components).
        """
        mse_loss = F.mse_loss(reconstruction, x, reduction="mean")

        l1_loss = hidden.abs().sum(dim=1).mean()

        if hidden.dim() == 3:
            hidden = hidden.reshape(-1, hidden.size(-1))

        usage_batch = (hidden > 0).float().mean(dim=0)

        if usage_batch.shape[0] != self.hidden_dim:
            raise RuntimeError(
                f"usage_batch has shape {usage_batch.shape}, "
                f"expected [{self.hidden_dim}]. Check your hidden shape!"
            )
        alpha = 0.99

        with torch.no_grad():
            self.usage_ema = alpha * self.usage_ema + (1 - alpha) * usage_batch

        # Compute KL divergence for usage regularization
        p = torch.clamp(self.usage_ema, min=1e-7, max=1.0 - 1e-7)
        rho = self.target_usage

        kl_usage = rho * torch.log(rho / p) + (1 - rho) * torch.log(
            (1 - rho) / (1 - p)
        )
        kl_loss = kl_usage.mean()

        total_loss = (
            mse_loss
            + self.l1_coefficient * l1_loss
            + self.usage_coeff * kl_loss
        )

        metrics = {
            "total": total_loss.item(),
            "mse": mse_loss.item(),
            "l1": l1_loss.item(),
            "kl_usage": kl_loss.item(),
        }
        return total_loss, metrics

    def training_step(
        self,
        x: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        gradient_clip_val: float | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Perform one optimization step.

        Args:
            x (torch.Tensor): Input tensor for training.
            optimizer (torch.optim.Optimizer): Optimizer instance.
            gradient_clip_val (float | None): Value to clip gradients.

        Returns:
            dict: Dictionary containing loss components.
        """
        reconstruction, hidden = self(x)
        loss, metrics = self.compute_loss(x, reconstruction, hidden)
        optimizer.zero_grad()
        loss.backward()

        self.remove_parallel_gradients()

        if gradient_clip_val is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), gradient_clip_val)

        optimizer.step()

        with torch.no_grad():
            self.decoder.weight.data = F.normalize(
                self.decoder.weight.data, dim=0
            )

        return metrics

    def remove_parallel_gradients(self) -> None:
        """Remove gradient components parallel to decoder dictionary vectors."""
        if self.decoder.weight.grad is not None:
            normalized_dict = F.normalize(self.decoder.weight.data, dim=0)
            parallel_component = (
                self.decoder.weight.grad * normalized_dict
            ).sum(0) * normalized_dict
            self.decoder.weight.grad -= parallel_component

    def get_dead_neurons(self, threshold: float = 1e-6) -> torch.Tensor:
        """Identify dead neurons based on usage statistics.

        Args:
            threshold (float): Threshold below which neurons are considered
            dead.

        Returns:
            torch.Tensor: Indices of dead neurons.
        """
        return (self.usage_ema < threshold).nonzero(as_tuple=False).flatten()
