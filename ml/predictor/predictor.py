# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn
# pyrefly: ignore [missing-import]
from torch.utils.data import DataLoader
import numpy as np

class Predictor:
    def __init__(self,model: nn.Module,device: torch.device | str,) -> None:

        self.model = model

        self.device = device

        self.model.to(self.device)

    def predict(self,dataloader: DataLoader,) -> dict[str, torch.Tensor]:
        self.model.eval()

        eta_predictions = []

        delay_probabilities = []

        with torch.no_grad():

            for batch in dataloader:

                numerical_features = batch["numerical_features"].to(self.device)

                categorical_features = batch["categorical_features"].to(self.device)

                outputs = self.model(
                numerical_features,
                categorical_features,
                )

                eta_predictions.append(outputs["eta"].cpu().numpy().ravel())
                delay_probabilities.append(
                torch.sigmoid(outputs["delay"]).cpu().numpy().ravel())

        eta_predictions = torch.tensor(np.concatenate(eta_predictions,))
        delay_probabilities = torch.tensor(np.concatenate(delay_probabilities,))

        return {
            "eta": eta_predictions,
            "delay_probability": delay_probabilities,
        }