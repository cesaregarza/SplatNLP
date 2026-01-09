from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from splatnlp.model.models import SetCompletionModel
from splatnlp.monosemantic_sae.data_objects import ActivationHook


@dataclass
class PooledPredictor:
    model: SetCompletionModel
    vocab: dict[str, int]
    weapon_vocab: dict[str, int]
    device: torch.device
    normalize: bool

    def __post_init__(self) -> None:
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.labels = [self.inv_vocab[i] for i in range(len(self.inv_vocab))]
        self.pad_id = self.vocab["<PAD>"]
        self.hook = ActivationHook(capture="input")
        self.handle = self.model.output_layer.register_forward_hook(self.hook)

    def close(self) -> None:
        self.handle.remove()

    def _extract_pooled(self) -> torch.Tensor:
        pooled = self.hook.get_and_clear()
        if pooled is None:
            raise RuntimeError("Failed to capture pooled embeddings")
        if self.normalize:
            pooled = F.normalize(pooled, p=2, dim=-1)
        return pooled

    def predict(
        self, tokens: list[str], weapon_id: str
    ) -> tuple[dict[str, float], np.ndarray]:
        if not tokens:
            tokens = ["<NULL>"]
        input_tokens = [self.vocab[tok] for tok in tokens]
        input_tensor = torch.tensor(
            input_tokens, device=self.device
        ).unsqueeze(0)
        weapon_tensor = torch.tensor(
            [self.weapon_vocab[weapon_id]], device=self.device
        ).unsqueeze(0)
        key_padding_mask = (input_tensor == self.pad_id).to(self.device)

        self.hook.clear_activations()
        with torch.inference_mode():
            outputs = self.model(
                input_tensor, weapon_tensor, key_padding_mask=key_padding_mask
            )
            probs = torch.sigmoid(outputs).squeeze(0)
            pooled = self._extract_pooled()

        probs_np = probs.detach().cpu().numpy()
        probs_dict = {
            label: float(probs_np[i]) for i, label in enumerate(self.labels)
        }
        return probs_dict, pooled.squeeze(0).detach().cpu().numpy()

    def predict_batch(
        self, batch_tokens: list[list[str]], weapon_id: str
    ) -> tuple[list[dict[str, float]], list[np.ndarray]]:
        if not batch_tokens:
            return [], []

        max_len = max(len(tokens) for tokens in batch_tokens)
        batch_size = len(batch_tokens)

        input_tokens = torch.full(
            (batch_size, max_len),
            self.pad_id,
            device=self.device,
            dtype=torch.long,
        )
        for row_idx, tokens in enumerate(batch_tokens):
            if not tokens:
                tokens = ["<NULL>"]
            token_ids = [self.vocab[token] for token in tokens]
            input_tokens[row_idx, : len(token_ids)] = torch.tensor(
                token_ids, device=self.device, dtype=torch.long
            )

        weapon_tensor = torch.tensor(
            [self.weapon_vocab[weapon_id]] * batch_size,
            device=self.device,
            dtype=torch.long,
        ).unsqueeze(1)
        key_padding_mask = input_tokens == self.pad_id

        self.hook.clear_activations()
        with torch.inference_mode():
            outputs = self.model(
                input_tokens, weapon_tensor, key_padding_mask=key_padding_mask
            )
            probs = torch.sigmoid(outputs)
            pooled = self._extract_pooled()

        probs_np = probs.detach().cpu().numpy()
        probs_list = [
            {label: float(row[i]) for i, label in enumerate(self.labels)}
            for row in probs_np
        ]

        activations = pooled.detach().cpu().numpy()
        return probs_list, list(activations)


def build_predictors(
    model: SetCompletionModel,
    vocab: dict[str, int],
    weapon_vocab: dict[str, int],
    *,
    device: torch.device,
    normalize: bool = True,
) -> PooledPredictor:
    return PooledPredictor(
        model=model,
        vocab=vocab,
        weapon_vocab=weapon_vocab,
        device=device,
        normalize=normalize,
    )
