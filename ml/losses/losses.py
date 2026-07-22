# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn

from ml.configs import config

class MultiTaskLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.eta_loss = nn.HuberLoss()

        self.delay_loss = nn.BCEWithLogitsLoss()

    def forward(self,predictions: dict[str, torch.Tensor],targets: dict[str, torch.Tensor],) -> torch.Tensor:

        eta_loss = self.eta_loss(
            predictions["eta"],
            targets["eta_target"],
        )

        delay_loss = self.delay_loss(
            predictions["delay"],
            targets["delay_target"],
        )

        total_loss = (
            config.ETA_LOSS_WEIGHT * eta_loss
            + config.DELAY_LOSS_WEIGHT * delay_loss
        )

        return total_loss