import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from splatnlp.model.models import SetCompletionModel
from splatnlp.monosemanticity_train.models import SparseAutoencoder

logger = logging.getLogger(__name__)


class SetCompletionHook:
    """
    forward-PRE hook that intercepts the 512-d masked-mean vector, runs it
    through the SAE logic *matching its forward pass* (center -> encoder ->
    ReLU -> normalize decoder -> decode -> add bias), optionally edits ONE
    hidden unit (POST-ReLU), and feeds the reconstructed 512-d vector
    onward to model.output_layer.
    """

    def __init__(
        self,
        sae_model: SparseAutoencoder,
        neuron_number: int | None = None,
        neuron_value: float | None = None,
        bypass: bool = False,
        no_change: bool = False,
    ) -> None:
        """Initializes the SetCompletionHook object.

        Args:
            sae_model (SparseAutoencoder): The sparse autoencoder model used for
                reconstruction.
            neuron_number (int | None, optional): Index of the neuron to edit
                post-ReLU. Defaults to None.
            neuron_value (float | None, optional): Value to assign to the
                specified neuron. Defaults to None.
            bypass (bool, optional): If True, bypasses reconstruction and passes
                input unchanged regardless of `no_change`. Defaults to False.
            no_change (bool, optional): If True, performs reconstruction without
                modifying neuron values. Ignored if `bypass` is True. Defaults
                to False.
        """
        self.sae = sae_model
        self.idx = neuron_number
        self.value = neuron_value
        self.bypass = bypass
        self.no_change = no_change
        self.last_in = None
        self.last_h_pre = None
        self.last_h_post = None
        self.last_x_recon = None

        if not hasattr(self.sae, "decoder_bias"):
            logger.warning(
                "SAE model provided to hook is missing 'decoder_bias'. "
                "Reconstruction might be incorrect."
            )
        if not hasattr(self.sae, "encoder") or not isinstance(
            self.sae.encoder, nn.Linear
        ):
            logger.warning(
                "SAE model provided to hook is missing 'encoder' or it's not "
                "nn.Linear."
            )
        if not hasattr(self.sae, "decoder") or not isinstance(
            self.sae.decoder, nn.Linear
        ):
            logger.warning(
                "SAE model provided to hook is missing 'decoder' or it's not "
                "nn.Linear."
            )

    def __call__(
        self, module: nn.Module, inputs: tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor, ...]:
        """Processes the input tensor through the Sparse Autoencoder
        reconstruction pipeline.

        Args:
            module (nn.Module): The module the hook is attached to (unused).
            inputs (tuple[torch.Tensor, ...]): Tuple containing a single input
                tensor (masked-mean vector).

        Returns:
            tuple[torch.Tensor]: Tuple containing the reconstructed (and
                optionally neuron-edited) tensor.
        """
        logger.debug("[Hook %s]", self.mode_str)
        self.original_value = None
        if self.bypass:
            logger.debug("[Hook BYPASS]")
            return

        x = inputs[0]
        self.last_in = x.detach().clone()

        h_pre, h_post = self.sae.encode(x)
        self.last_h_pre = h_pre.detach().clone()
        self.last_h_post = h_post.detach().clone()

        h_final = h_post
        if (
            (not self.no_change)
            and (self.idx is not None)
            and (self.value is not None)
        ):
            if self.idx >= h_final.shape[-1]:
                raise IndexError(
                    f"neuron {self.idx} >= hidden dim {h_final.shape[-1]}"
                )
            self.original_value = h_final[:, self.idx].mean().item()
            h_final = h_final.clone()
            h_final[:, self.idx] = self.value

        x_recon = self.sae.decode(h_final)
        self.last_x_recon = x_recon.detach().clone()

        l1_diff = (x_recon - x).abs().sum().item()
        mse_diff = F.mse_loss(x_recon, x).item()
        logger.debug(
            "[Hook %s] L1 Recon Diff: %.4f, MSE Recon Diff: %.6f",
            self.mode_str,
            l1_diff,
            mse_diff,
        )
        if self.original_value is not None:
            logger.debug(
                "[Hook %s] Original Neuron Value (mean): %.4f",
                self.mode_str,
                self.original_value,
            )

        return (x_recon,)

    @property
    def mode_str(self) -> str:
        """Provides a human-readable description of the current hook mode.

        Returns:
            str: The current mode description ("BYPASS", "NO-CHANGE", or
                "EDIT(n=<idx>, v=<value>)").
        """
        return (
            "BYPASS"
            if self.bypass
            else (
                "NO-CHANGE"
                if self.no_change
                else f"EDIT(n={self.idx}, v={self.value})"
            )
        )

    def set_mode(
        self, *, bypass: bool | None = None, no_change: bool | None = None
    ) -> None:
        """Sets the operation mode of the hook.

        Args:
            bypass (bool | None, optional): If True, bypasses reconstruction
                entirely. Overrides `no_change`. Defaults to None.
            no_change (bool | None, optional): If True, reconstruction occurs
                without neuron modification. Ignored if `bypass` is True.
                Defaults to None.
        """
        if bypass is not None:
            self.bypass = bypass
        if no_change is not None:
            self.no_change = no_change
        logger.debug("[Hook %s]", self.mode_str)

    def update_neuron(self, neuron_number: int, neuron_value: float) -> None:
        """Updates the neuron editing parameters and ensures the hook enters
        edit mode.

        Args:
            neuron_number (int): Index of the neuron to modify post-ReLU.
            neuron_value (float): The value to set for the specified neuron.
        """
        self.idx = neuron_number
        self.value = neuron_value
        self.bypass = False
        self.no_change = False
        logger.debug("[Hook %s]", self.mode_str)


def register_hooks(
    primary_model: SetCompletionModel,
    sae_model: SparseAutoencoder,
    neuron_number: int | None = None,
    neuron_value: float | None = None,
    bypass: bool = False,
    no_change: bool = False,
) -> None:
    hook = SetCompletionHook(
        sae_model,
        neuron_number=neuron_number,
        neuron_value=neuron_value,
        bypass=bypass,
        no_change=no_change,
    )
    primary_model.output_layer.register_forward_pre_hook(hook)
    logger.info("Registered SetCompletionHook with %s", hook.mode_str)
    return hook
