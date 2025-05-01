# splatnlp/monosemanticity_train/data_objects.py

import logging
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Assuming SetCompletionModel is importable if needed for type hints elsewhere
# from splatnlp.model.models import SetCompletionModel

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
    resample_interval: int = (
        12500  # Not directly used by model, but by training loop
    )
    learning_rate: float = 1e-4
    dead_neuron_steps: int = 12500  # Used by SAE model if implemented there
    target_usage: float = 0.05
    usage_coeff: float = 1e-3


class SAEOutput(NamedTuple):
    """Structured output from SAE forward pass."""

    reconstructed: torch.Tensor
    hidden: torch.Tensor
    # Removed pre_activation as it's not used in current SAE model


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
        output: Optional[
            torch.Tensor
        ] = None,  # Make output optional for pre-hooks
    ) -> Optional[torch.Tensor]:  # Return type optional for pre-hooks
        if self.capture == "input":
            if inputs:
                # Detach to prevent gradient flow through the hook capture
                self.activations = (
                    inputs[0].detach().clone()
                    if isinstance(inputs[0], torch.Tensor)
                    else None
                )
            else:
                self.activations = None
            # For pre-hooks, return None to not modify input
            # For forward hooks used as pre-hooks, return None (or original input if needed)
            return None  # Or potentially inputs depending on hook type usage
        elif self.capture == "output":
            if output is not None:
                # Detach to prevent gradient flow back through the hook capture
                self.activations = (
                    output.detach().clone()
                    if isinstance(output, torch.Tensor)
                    else None
                )
            else:
                self.activations = None
            # For standard forward hooks, return the original output
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
        # Use register_forward_pre_hook if specifically targeting input *before* module runs
        # return module.register_forward_pre_hook(hook)
        # Or use register_forward_hook if capturing input *within* the forward call
        return module.register_forward_hook(
            hook
        )  # Assuming forward hook used to capture input
    else:  # capture == "output"
        return module.register_forward_hook(hook)


class ActivationBuffer:
    def __init__(self, max_size=819200, d=None, device="cuda"):
        self.max_size = max_size
        self.device = (
            torch.device(device) if isinstance(device, str) else device
        )
        self.d = d  # Dimension can be pre-set or inferred
        self.buf: Optional[torch.Tensor] = (
            None  # Explicitly type hint as Optional
        )
        self.ptr = 0
        self.full = False
        # Initialize buffer immediately if dimension 'd' is provided
        if self.d is not None:
            self._initialize_buffer()

    def _initialize_buffer(self):
        """Initializes the buffer tensor if not already done."""
        # Check if buffer needs initialization and dimension is known
        if self.buf is None and self.d is not None:
            logger.info(
                f"Initializing activation buffer: size={self.max_size}, dim={self.d}, device={self.device}"
            )
            try:
                # Use float32 for the buffer for numerical stability during SAE training
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
                self.buf = None  # Ensure buf remains None if allocation fails
        elif self.buf is not None:
            logger.warning(
                "Attempted to initialize buffer that was already initialized."
            )
        elif self.d is None:
            logger.error("Cannot initialize buffer without dimension 'd'.")

    @torch.no_grad()
    def add(self, acts: torch.Tensor):
        # Ensure acts is a tensor before proceeding
        if not isinstance(acts, torch.Tensor):
            logger.warning(
                f"Input to ActivationBuffer.add is not a tensor (type: {type(acts)}). Skipping."
            )
            return

        if acts.dim() == 3:
            # Assuming last dim is feature dim
            acts = acts.reshape(-1, acts.size(-1))
        elif acts.dim() != 2:
            logger.error(
                f"Activations must be 2D or 3D, got {acts.dim()}D. Skipping add."
            )
            return

        N = acts.size(0)
        if N == 0:
            logger.debug("Attempted to add zero activations.")
            return

        # --- Buffer Initialization Check ---
        # Ensure buffer is initialized before any assignment attempts
        if self.buf is None:
            if self.d is None:
                # Infer dimension from the first non-empty tensor added
                self.d = acts.size(1)
                logger.info(
                    f"Inferred buffer dimension d={self.d} from first activation batch."
                )
            self._initialize_buffer()  # Attempt to create the buffer tensor

            # Crucially, check if initialization succeeded before proceeding
            if self.buf is None:
                logger.error(
                    "Buffer initialization failed. Cannot add activations."
                )
                return
        # --- End Initialization Check ---

        # Check dimension consistency if buffer already exists
        if acts.size(1) != self.d:
            logger.error(
                f"Activation dimension mismatch! Buffer expects {self.d}, got {acts.size(1)}. Skipping add."
            )
            return

        # Ensure input activations are on the correct device (buffer's device)
        acts = acts.to(self.device)

        # Now self.buf is guaranteed to be a Tensor
        # Proceed with assignment logic
        if N >= self.max_size:
            self.buf.copy_(
                acts[-self.max_size :].to(torch.float32)
            )  # Ensure dtype for copy
            self.ptr, self.full = 0, True
            logger.info(
                f"Buffer filled entirely by large input batch ({N}/{self.max_size} activations)."
            )
            return

        end = self.ptr + N
        if end <= self.max_size:
            # Assign slice; ensure input tensor has correct dtype
            self.buf[self.ptr : end] = acts.to(torch.float32)
        else:
            # Handle wrap-around assignment
            first = self.max_size - self.ptr
            self.buf[self.ptr :] = acts[:first].to(torch.float32)
            self.buf[: N - first] = acts[first:].to(torch.float32)
            self.full = True
        self.ptr = (self.ptr + N) % self.max_size

    def get_loader(self, batch_size=1024, shuffle=True):
        """Creates a DataLoader for the buffered activations."""
        size = self.max_size if self.full else self.ptr
        if (
            size == 0 or self.buf is None
        ):  # Check if buffer is empty or not initialized
            logger.warning(
                "Attempting to get loader from empty or uninitialized buffer."
            )
            return None

        # self.buf[:size] is already on self.device
        current_buffer_data = self.buf[:size]
        ds = TensorDataset(current_buffer_data)

        # IMPORTANT: Set pin_memory=False because data might already be on GPU
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=False,  # Explicitly disable pinning if data is on GPU
            num_workers=0,  # Often safer for TensorDatasets, especially on GPU
        )

    def __len__(self):
        return self.max_size if self.full else self.ptr
