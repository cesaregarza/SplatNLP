import dataclasses

import torch


@dataclasses.dataclass
class TrainingConfig:
    num_epochs: int
    patience: int
    learning_rate: float
    weight_decay: float
    clip_grad_norm: float
    scheduler_factor: float
    scheduler_patience: int
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    distributed: bool = False
