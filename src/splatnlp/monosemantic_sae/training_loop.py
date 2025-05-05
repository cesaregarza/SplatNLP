from dataclasses import dataclass

import torch
import tqdm
from torch import nn

from splatnlp.monosemantic_sae.data_objects import ActivationHook, SAEConfig
from splatnlp.monosemantic_sae.models import SparseAutoencoder


@dataclass
class TrainingStats:
    total_loss: float = 0.0
    reconstruction_loss: float = 0.0
    l1_loss: float = 0.0
    l2_loss: float = 0.0
    scale_loss: float = 0.0
    entropy_loss: float = 0.0
    num_batches: int = 0
    sparsity: float = 0.0
    max_activation: float = 0.0
    mean_scale: float = 0.0

    @property
    def average_loss(self) -> float:
        return self.total_loss / max(self.num_batches, 1)

    @property
    def average_reconstruction_loss(self) -> float:
        return self.reconstruction_loss / max(self.num_batches, 1)

    @property
    def average_l1_loss(self) -> float:
        return self.l1_loss / max(self.num_batches, 1)

    @property
    def average_l2_loss(self) -> float:
        return self.l2_loss / max(self.num_batches, 1)

    @property
    def average_scale_loss(self) -> float:
        return self.scale_loss / max(self.num_batches, 1)

    @property
    def average_entropy_loss(self) -> float:
        return self.entropy_loss / max(self.num_batches, 1)

    @property
    def average_sparsity(self) -> float:
        return self.sparsity / max(self.num_batches, 1)

    @property
    def average_max_activation(self) -> float:
        return self.max_activation / max(self.num_batches, 1)

    @property
    def average_mean_scale(self) -> float:
        return self.mean_scale / max(self.num_batches, 1)


def train_epoch_sae(
    primary_model: nn.Module,
    sae_model: SparseAutoencoder,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    hook: ActivationHook,
    vocab: dict[str, int],
    subbatch_size: int = 32,
    gradient_clip_val: float | None = None,
) -> TrainingStats:
    """Enhanced training function with proper sequence padding handling."""
    stats = TrainingStats()
    subbatch: list[torch.Tensor] = []
    current_sequence_length: int | None = None

    primary_model.eval()
    sae_model.train()

    progress_bar = tqdm.tqdm(total=len(dataloader), desc="Training epoch")

    def process_subbatch(
        subbatch_tensors: list[torch.Tensor],
    ) -> dict[str, float]:
        # All tensors in subbatch should have same sequence length due to our collection logic
        subbatch_tensor = torch.cat(subbatch_tensors, dim=0)
        metrics = sae_model.training_step(subbatch_tensor, optimizer)

        if gradient_clip_val is not None:
            torch.nn.utils.clip_grad_norm_(
                sae_model.parameters(), gradient_clip_val
            )

        scheduler.step()

        return metrics

    try:
        for batch_idx, (
            batch_inputs,
            batch_weapons,
            batch_targets,
            _,
        ) in enumerate(dataloader):
            batch_inputs = batch_inputs.to(device)
            batch_weapons = batch_weapons.to(device)
            key_padding_mask = (batch_inputs == vocab["<PAD>"]).to(device)

            with torch.no_grad():
                _ = primary_model(
                    batch_inputs,
                    batch_weapons,
                    key_padding_mask=key_padding_mask,
                )

            activation = hook.get_and_clear()
            if activation is None:
                raise RuntimeError("Hook failed to capture activation inputs")

            # Check sequence length of current activation
            current_batch_seq_len = activation.size(1)

            # If we have a different sequence length, process current subbatch and start new one
            if (
                current_sequence_length is not None
                and current_batch_seq_len != current_sequence_length
            ):
                if subbatch:
                    metrics = process_subbatch(subbatch)
                    stats.total_loss += metrics["total"]
                    stats.reconstruction_loss += metrics["reconstruction"]
                    stats.l1_loss += metrics["l1"]
                    stats.l2_loss += metrics["l2"]
                    stats.scale_loss += metrics["scale"]
                    stats.entropy_loss += metrics["entropy"]
                    stats.sparsity += metrics["sparsity"]
                    stats.max_activation += metrics["max_activation"]
                    stats.mean_scale += metrics["mean_scale"]
                    stats.num_batches += 1

                    progress_bar.set_postfix(
                        {
                            "Loss": f"{metrics['total']:.4f}",
                            "MSE": f"{metrics['reconstruction']:.4f}",
                            "Sparsity": f"{metrics['sparsity']:.3f}",
                            "MaxAct": f"{metrics['max_activation']:.3f}",
                            "Scales": f"{metrics['mean_scale']:.3f}",
                        }
                    )
                subbatch = []

            current_sequence_length = current_batch_seq_len
            subbatch.append(activation)

            # Process when subbatch is full
            if len(subbatch) >= subbatch_size:
                metrics = process_subbatch(subbatch)
                stats.total_loss += metrics["total"]
                stats.reconstruction_loss += metrics["reconstruction"]
                stats.l1_loss += metrics["l1"]
                stats.l2_loss += metrics["l2"]
                stats.scale_loss += metrics["scale"]
                stats.entropy_loss += metrics["entropy"]
                stats.sparsity += metrics["sparsity"]
                stats.max_activation += metrics["max_activation"]
                stats.mean_scale += metrics["mean_scale"]
                stats.num_batches += 1

                progress_bar.set_postfix(
                    {
                        "Loss": f"{metrics['total']:.4f}",
                        "MSE": f"{metrics['reconstruction']:.4f}",
                        "Sparsity": f"{metrics['sparsity']:.3f}",
                        "MaxAct": f"{metrics['max_activation']:.3f}",
                        "Scales": f"{metrics['mean_scale']:.3f}",
                    }
                )
                subbatch = []
                current_sequence_length = None

            progress_bar.update(1)

        # Process any remaining data
        if subbatch:
            metrics = process_subbatch(subbatch)
            stats.total_loss += metrics["total"]
            stats.reconstruction_loss += metrics["reconstruction"]
            stats.l1_loss += metrics["l1"]
            stats.l2_loss += metrics["l2"]
            stats.scale_loss += metrics["scale"]
            stats.entropy_loss += metrics["entropy"]
            stats.sparsity += metrics["sparsity"]
            stats.max_activation += metrics["max_activation"]
            stats.mean_scale += metrics["mean_scale"]
            stats.num_batches += 1

    except Exception as e:
        progress_bar.close()
        raise e

    progress_bar.close()
    return stats
