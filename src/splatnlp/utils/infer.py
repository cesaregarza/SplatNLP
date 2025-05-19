from typing import Callable, Literal

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
    output_type: Literal["df", "dict", "list"] = "df",
) -> Callable[
    [
        SetCompletionModel,
        list[str],
        str,
        bool,
        bool,
    ],
    pd.DataFrame | dict[str, float] | list[str],
]:
    """Builds a function that predicts abilities for a given target and weapon
    using a SetCompletionModel. If a hook is provided, this will also set the
    hook's bypass and no_change attributes to the provided values.

    Args:
        hook (SetCompletionHook | None, optional): The hook to use for the
            prediction. Defaults to None.
        device (torch.device | None, optional): The device to use for the
            prediction. Defaults to None.
        output_type (Literal["df", "dict", "list"], optional): The type of
            output to return. Defaults to "df".
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
    ) -> pd.DataFrame | dict[str, float] | list[str]:
        """Predicts abilities for a given target and weapon using a
        SetCompletionModel.

        Args:
            model (SetCompletionModel): The model to use for prediction.
            target (list[str]): List of target tokens to predict abilities for.
            weapon_id (str): The weapon ID to use for prediction.
            bypass (bool, optional): Whether to bypass the hook. Defaults to
                False.
            no_change (bool, optional): Whether to prevent changes in the hook.
                Defaults to False.

        Returns:
            pd.DataFrame | dict[str, float] | list[str]: The predicted abilities
                in the format specified by output_type.
                - If output_type is "df": Returns a DataFrame with labels as
                    index and probabilities as values.
                - If output_type is "dict": Returns a dictionary mapping labels
                    to probabilities.
                - If output_type is "list": Returns a list of labels.
        """
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

        if output_type == "df":
            return df.sort_values("probability", ascending=False).set_index(
                "label"
            )
        elif output_type == "dict":
            return {
                row["label"]: row["probability"] for _, row in df.iterrows()
            }
        elif output_type == "list":
            return df["label"].tolist()

    return predict_abilities
