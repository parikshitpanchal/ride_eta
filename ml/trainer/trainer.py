import numpy as np
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn
# pyrefly: ignore [missing-import]
from torch.utils.data import DataLoader

from ml.callbacks.checkpoint import ModelCheckpoint
from ml.callbacks.early_stopping import EarlyStopping
from ml.configs import config
from ml.metrics.classification_metrics import calculate_classification_metrics
from ml.metrics.regression_metrics import calculate_regression_metrics

import logging

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self,model: nn.Module,optimizer: torch.optim.Optimizer | None,criterion: nn.Module,train_loader: DataLoader,validation_loader: DataLoader,) -> None:
        
        self.model = model

        self.optimizer = optimizer
        # self.optimizer: torch.optim.Optimizer | None = None

        self.criterion = criterion

        self.train_loader = train_loader

        self.validation_loader = validation_loader

        self.device = config.DEVICE

        self.model.to(self.device)

        self.early_stopping = EarlyStopping(
            patience=config.EARLY_STOPPING_PATIENCE,
        )

        self.checkpoint = ModelCheckpoint(
            save_directory=config.SAVED_MODEL_DIR,
            checkpoint_name=config.CHECKPOINT_NAME,
        )

        self.history = {
            "train_loss": [],
            "validation_loss": [],
            "train_eta_metrics": [],
            "validation_eta_metrics": [],
            "train_delay_metrics": [],
            "validation_delay_metrics": [],
        }

    def train(self,) -> dict[str, list]:

        for epoch in range(config.EPOCHS):

            train_loss, train_eta_metrics, train_delay_metrics = (
                self._train_epoch()
            )

            validation_loss, validation_eta_metrics, validation_delay_metrics = (
                self._validate_epoch()
            )

            self.history["train_loss"].append(
                train_loss,
            )

            self.history["validation_loss"].append(
                validation_loss,
            )

            self.history["train_eta_metrics"].append(
                train_eta_metrics,
            )

            self.history["validation_eta_metrics"].append(
                validation_eta_metrics,
            )

            self.history["train_delay_metrics"].append(
                train_delay_metrics,
            )

            self.history["validation_delay_metrics"].append(
                validation_delay_metrics,
            )

            if config.SAVE_BEST_MODEL:
                self.checkpoint(
                    model=self.model,
                    validation_loss=validation_loss,
                )
                  
            logger.info(
                "Epoch %d/%d | "
                "Train Loss: %.4f | "
                "Validation Loss: %.4f | "
                "Train MAE: %.4f | "
                "Validation MAE: %.4f | "
                "Train F1: %.4f | "
                "Validation F1: %.4f",
                epoch + 1,
                config.EPOCHS,
                train_loss,
                validation_loss,
                train_eta_metrics["mae"],
                validation_eta_metrics["mae"],
                train_delay_metrics["f1"],
                validation_delay_metrics["f1"],
                )
            if self.early_stopping(validation_loss=validation_loss,):
                logger.info(f"Early stopping at epoch {epoch + 1}",)
                break

        return self.history

    def evaluate(self,dataloader: DataLoader):
        return self._run_epoch(
            dataloader=dataloader,
            training=False,
        )

    def _run_epoch(self,dataloader: DataLoader,training: bool,) -> tuple[float, dict[str, float], dict[str, float]]:

        if training:
            self.model.train()
        else:
            self.model.eval()

        running_loss = 0.0

        eta_predictions = []
        eta_targets = []

        delay_probabilities = []
        delay_targets = []

        for batch in dataloader:

            numerical_features = batch["numerical_features"].to(self.device)

            categorical_features = batch["categorical_features"].to(self.device)

            eta_target = batch["eta_target"].to(self.device)

            delay_target = batch["delay_target"].to(self.device)

            targets = {"eta_target": eta_target,"delay_target": delay_target,}

            if training:
                if self.optimizer is None:
                    raise RuntimeError("Optimizer is required during training.")
                self.optimizer.zero_grad()

            with torch.set_grad_enabled(training):

                outputs = self.model(
                    numerical_features,
                    categorical_features,
                )

                loss = self.criterion(
                predictions=outputs,
                targets=targets,
                )   

                if training:
                    loss.backward()
                    self.optimizer.step()

            running_loss += loss.item()

            eta_predictions.append(outputs["eta"].detach().cpu().numpy().ravel())

            eta_targets.append(eta_target.detach().cpu().numpy().ravel())

            delay_probabilities.append(torch.sigmoid(outputs["delay"]).detach().cpu().numpy().ravel())

            delay_targets.append(delay_target.detach().cpu().numpy().ravel())
            
        eta_predictions = np.concatenate(
            eta_predictions,
        )

        eta_targets = np.concatenate(
            eta_targets,
        )

        delay_probabilities = np.concatenate(
            delay_probabilities,
        )

        delay_targets = np.concatenate(
            delay_targets,
        )

        eta_metrics = calculate_regression_metrics(
            predictions=eta_predictions,
            targets=eta_targets,
        )

        delay_metrics = calculate_classification_metrics(
            probabilities=delay_probabilities,
            targets=delay_targets,
        )

        epoch_loss = running_loss / len(dataloader)

        return (
            epoch_loss,
            eta_metrics,
            delay_metrics,
        )
    def _train_epoch(self,) -> tuple[float, dict[str, float], dict[str, float]]:
        return self._run_epoch(
            dataloader=self.train_loader,
            training=True,
            )


    def _validate_epoch(self,) -> tuple[float, dict[str, float], dict[str, float]]:
        return self._run_epoch(
            dataloader=self.validation_loader,
            training=False,
        )

    