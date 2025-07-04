# splatnlp/monosemanticity_train/data_objects.py

import logging
from dataclasses import dataclass
from typing import NamedTuple, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


@dataclass
class SAEConfig:
    """Configuration for Sparse Autoencoder following
    'Towards Monospecificity' paper.
    """

    input_dim: int
    expansion_factor: float = 8.0
    l1_coefficient: float = 1e-5
    dead_neuron_threshold: float = 1e-6
    resample_interval: int = 12500
    learning_rate: float = 1e-4
    dead_neuron_steps: int = 12500
    target_usage: float = 0.05
    usage_coeff: float = 1e-3
    miracle_mse_target: float = 0.0003
    miracle_sparsity_target: float = 0.09
    miracle_dead_neuron_target: float = 3.5
    miracle_mse_weight: float = 1.0
    miracle_sparsity_weight: float = 1.0
    miracle_dead_neuron_weight: float = 1.0


class SAEOutput(NamedTuple):
    """Structured output from SAE forward pass."""

    reconstructed: torch.Tensor
    hidden: torch.Tensor


class ActivationHook:
    """Captures activations from a module."""

    def __init__(self, *, capture: str = "output"):
        if capture not in {"output", "input"}:
            raise ValueError("capture must be 'output' or 'input'")
        self.capture = capture
        self.activations: Optional[torch.Tensor] = None

    def __call__(
        self,
        module: nn.Module,
        inputs: Tuple[torch.Tensor, ...],
        output: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if self.capture == "input":
            if inputs:
                self.activations = (
                    inputs[0].detach().clone()
                    if isinstance(inputs[0], torch.Tensor)
                    else None
                )
            else:
                self.activations = None
            return None
        elif self.capture == "output":
            if output is not None:
                self.activations = (
                    output.detach().clone()
                    if isinstance(output, torch.Tensor)
                    else None
                )
            else:
                self.activations = None
            return output
        return None

    def get_activations(self) -> Optional[torch.Tensor]:
        return self.activations

    def clear_activations(self) -> None:
        self.activations = None

    def get_and_clear(self) -> Optional[torch.Tensor]:
        acts = self.activations
        self.activations = None
        return acts


def register_activation_hook_generic(
    module: nn.Module, hook: ActivationHook
) -> torch.utils.hooks.RemovableHandle:
    """Registers an activation hook on a module."""
    if hook.capture == "input":
        return module.register_forward_hook(hook)
    else:
        return module.register_forward_hook(hook)


class ActivationBuffer:
    """A circular buffer for storing activation tensors in a fixed-size buffer.

    The buffer automatically overwrites older activations when full, allowing
    continuous collection of activations from neural networks.

    Attributes:
        max_size (int): Maximum number of activation vectors the buffer can
            store.
        d (Optional[int]): Dimension of activation vectors. Inferred from data
            if not provided.
        device (torch.device): Device on which the buffer tensor is stored.
        buf (Optional[torch.Tensor]): Tensor buffer storing activations.
        ptr (int): Current write position in the buffer.
        full (bool): Indicates whether the buffer has reached maximum capacity.
    """

    def __init__(self, max_size=819200, d=None, device="cuda"):
        """Initializes an ActivationBuffer.

        Args:
            max_size (int, optional): Maximum buffer size. Defaults to 819200.
            d (Optional[int], optional): Dimension of activation vectors.
                Defaults to None.
            device (str or torch.device, optional): Device for buffer storage.
                Defaults to 'cuda'.
        """
        self.max_size = max_size
        self.device = (
            torch.device(device) if isinstance(device, str) else device
        )
        self.d = d
        self.buf: Optional[torch.Tensor] = None
        self.ptr = 0
        self.full = False
        if self.d is not None:
            self._initialize_buffer()

    def _initialize_buffer(self):
        """Initializes the buffer tensor."""
        if self.buf is None and self.d is not None:
            logger.info(
                f"Initializing activation buffer: size={self.max_size}, "
                f"dim={self.d}, device={self.device}"
            )
            try:
                self.buf = torch.empty(
                    self.max_size,
                    self.d,
                    dtype=torch.float32,
                    device=self.device,
                )
            except Exception as e:
                logger.exception(
                    f"Failed to allocate buffer tensor on {self.device}: {e}"
                )
                self.buf = None
        elif self.buf is not None:
            logger.warning(
                "Attempted to initialize buffer that was already initialized."
            )
        elif self.d is None:
            logger.error("Cannot initialize buffer without dimension 'd'.")

    @torch.no_grad()
    def add(self, acts: torch.Tensor):
        """Adds new activations to the buffer.

        Args:
            acts (torch.Tensor): Tensor of activations to add, must be 2D
                (batch, features) or 3D (batch, sequence, features).
        """
        if not isinstance(acts, torch.Tensor):
            logger.warning(
                "Input to ActivationBuffer.add is not a tensor (type: "
                f"{type(acts)}). Skipping."
            )
            return

        if acts.dim() == 3:
            acts = acts.reshape(-1, acts.size(-1))
        elif acts.dim() != 2:
            logger.error(
                f"Activations must be 2D or 3D, got {acts.dim()}D. Skipping "
                "add."
            )
            return

        N = acts.size(0)
        if N == 0:
            logger.debug("Attempted to add zero activations.")
            return

        if self.buf is None:
            if self.d is None:
                self.d = acts.size(1)
                logger.info(
                    f"Inferred buffer dimension d={self.d} from first "
                    "activation batch."
                )
            self._initialize_buffer()

        if self.buf is None:
            logger.error(
                "Buffer initialization failed. Cannot add activations."
            )
            return

        if acts.size(1) != self.d:
            logger.error(
                f"Activation dimension mismatch! Buffer expects {self.d}, "
                f"got {acts.size(1)}. Skipping add."
            )
            return

        acts = acts.to(self.device)

        if N >= self.max_size:
            self.buf.copy_(acts[-self.max_size :].to(torch.float32))
            self.ptr, self.full = 0, True
            logger.info(
                "Buffer filled entirely by large input batch "
                f"({N}/{self.max_size} activations)."
            )
            return

        end = self.ptr + N
        if end <= self.max_size:
            self.buf[self.ptr : end] = acts.to(torch.float32)
        else:
            first = self.max_size - self.ptr
            self.buf[self.ptr :] = acts[:first].to(torch.float32)
            self.buf[: N - first] = acts[first:].to(torch.float32)
            self.full = True
        self.ptr = (self.ptr + N) % self.max_size

    def get_loader(self, batch_size=1024, shuffle=True):
        """Creates a DataLoader for the activations stored in the buffer.

        Args:
            batch_size (int, optional): Number of samples per batch. Defaults
                to 1024.
            shuffle (bool, optional): Whether to shuffle data each epoch.
                Defaults to True.

        Returns:
            Optional[DataLoader]: DataLoader object or None if buffer is empty
                or uninitialized.
        """
        size = self.max_size if self.full else self.ptr
        if size == 0 or self.buf is None:
            logger.warning(
                "Attempting to get loader from empty or uninitialized buffer."
            )
            return None

        current_buffer_data = self.buf[:size]
        ds = TensorDataset(current_buffer_data)

        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=False,
            num_workers=0,
        )

    def __len__(self):
        """Returns the number of activations currently stored in the buffer.

        Returns:
            int: Number of activations in the buffer.
        """
        return self.max_size if self.full else self.ptr
