from typing import Callable

import numpy as np
import pandas as pd
import torch

from splatnlp.model.models import SetCompletionModel
from splatnlp.monosemantic_sae.hooks import SetCompletionHook


def build_predict_abilities(
    vocab: dict[str, int],
    weapon_vocab: dict[str, int],
    pad_token: str = "<PAD>",
    hook: SetCompletionHook | None = None,
    device: torch.device | None = None,
) -> Callable[
    [
        SetCompletionModel,
        list[str],
        str,
        bool,
        bool,
    ],
    pd.DataFrame,
]:
    """Builds a function that predicts abilities for a given target and weapon
    using a SetCompletionModel. If a hook is provided, this will also set the
    hook's bypass and no_change attributes to the provided values.

    Args:
        hook (SetCompletionHook | None, optional): The hook to use for the
            prediction. Defaults to None.
        device (torch.device | None, optional): The device to use for the
            prediction. Defaults to None.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    inv_vocab = {v: k for k, v in vocab.items()}

    def predict_abilities(
        model: SetCompletionModel,
        target: list[str],
        weapon_id: str,
        bypass: bool = False,
        no_change: bool = False,
    ) -> pd.DataFrame:
        model.eval()
        if hook is not None:
            hook.bypass = bypass
            hook.no_change = no_change

        input_tokens = [vocab[token] for token in target]
        input_tokens = torch.tensor(input_tokens, device=device).unsqueeze(0)
        input_weapons = torch.tensor(
            [weapon_vocab[weapon_id]], device=device
        ).unsqueeze(0)
        key_padding_mask = (input_tokens == vocab[pad_token]).to(device)

        with torch.no_grad():
            outputs = model(
                input_tokens, input_weapons, key_padding_mask=key_padding_mask
            )
            probs = torch.sigmoid(outputs)

        probs_np = probs.squeeze().cpu().detach().numpy()
        if probs_np.ndim == 0:
            probs_np = np.array([probs_np])

        df = pd.DataFrame(
            {
                "label": [inv_vocab[i] for i in range(len(probs_np))],
                "probability": probs_np,
            }
        )

        return df.sort_values("probability", ascending=False).reset_index(
            drop=True
        )

    return predict_abilities
