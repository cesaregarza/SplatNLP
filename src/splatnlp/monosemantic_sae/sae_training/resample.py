import logging

import torch
import torch.nn.functional as F
from torch.optim import Optimizer

from splatnlp.monosemantic_sae.data_objects import ActivationBuffer, SAEConfig
from splatnlp.monosemantic_sae.models import SparseAutoencoder

logger = logging.getLogger(__name__)


@torch.no_grad()
def resample_dead_neurons(
    sae_model: SparseAutoencoder,
    activation_buffer: ActivationBuffer,
    optimizer: Optimizer,
    sae_config: SAEConfig,
    device: torch.device,
    resample_weight: float = 0.2,
    resample_bias: float = 0.0,
) -> int:
    """
    Resample dead neurons in the SAE model using activations from the buffer.

    Args:
        sae_model: The Sparse Autoencoder model
        activation_buffer: Buffer containing recent activations
        optimizer: The optimizer used for training
        sae_config: Configuration for the SAE
        device: Device for computations
        resample_weight: Weight for normalized resampled inputs (default: 0.2)
        resample_bias: Bias for resampled neurons (default: 0.0)

    Returns:
        int: Number of neurons that were resampled
    """
    dead_neurons = sae_model.get_dead_neurons(
        threshold=sae_config.dead_neuron_threshold
    )
    n_dead = len(dead_neurons)
    if n_dead == 0:
        return 0

    logger.info(f"Resampling {n_dead} dead neurons")
    sample_size = min(len(activation_buffer), n_dead * 10)
    if sample_size == 0:
        logger.warning("Activation buffer empty - skip resampling")
        return 0

    buffer_loader = activation_buffer.get_loader(
        batch_size=sample_size, shuffle=False
    )
    inputs_sample: torch.Tensor = next(iter(buffer_loader))[0]

    recon: torch.Tensor = sae_model(inputs_sample)[0]

    mse = F.mse_loss(
        recon.float(), inputs_sample.float(), reduction="none"
    ).mean(dim=1)

    if torch.isnan(mse).any() or mse.sum() < 1e-12:
        idx = torch.randint(0, inputs_sample.size(0), (n_dead,), device=device)
    else:
        probs = (mse / mse.sum()).cpu()
        idx = torch.multinomial(probs, n_dead, replacement=True).to(device)

    resample_inputs = inputs_sample[idx]
    normed = F.normalize(resample_inputs, dim=1)

    sae_model.encoder.weight.data[dead_neurons] = normed * resample_weight
    sae_model.encoder.bias.data[dead_neurons] = resample_bias
    sae_model.decoder.weight.data[:, dead_neurons] = (
        sae_model.encoder.weight.data[dead_neurons].T
    )

    # Zero Adam moments so the fresh weights aren't dragged back to 0
    if isinstance(optimizer, (torch.optim.AdamW, torch.optim.Adam)):
        for group in optimizer.param_groups:
            for p in group["params"]:
                state = optimizer.state[p]
                if not state:  # param not in state dict yet
                    continue
                if p is sae_model.encoder.weight or p is sae_model.encoder.bias:
                    state["exp_avg"][dead_neurons] = 0
                    state["exp_avg_sq"][dead_neurons] = 0
                elif p is sae_model.decoder.weight:
                    state["exp_avg"][:, dead_neurons] = 0
                    state["exp_avg_sq"][:, dead_neurons] = 0

    # Reset running usage estimate for those neurons
    live = sae_model.usage_ema > sae_config.dead_neuron_threshold
    sae_model.usage_ema[dead_neurons] = (
        sae_model.usage_ema[live].mean()
        if live.any()
        else sae_config.target_usage
    )

    logger.info("Finished resampling %d neurons", n_dead)
    return n_dead
