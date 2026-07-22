import logging

import pandas as pd
# pyrefly: ignore [missing-import]
import torch

from ml.configs import config
# pyrefly: ignore [missing-import]
from ml.datasets.dataloader import create_dataloader
# pyrefly: ignore [missing-import]
from ml.datasets.ride_dataset import RideDataset
# pyrefly: ignore [missing-import]
from ml.losses.losses import MultiTaskLoss
from ml.models.ride_eta_network import RideETANetwork
from ml.preprocessing.encoder import CategoricalEncoder
from ml.preprocessing.scaler import NumericalScaler
from ml.preprocessing.validator import DataValidator
from ml.trainer.trainer import Trainer
from ml.utils.model_metadata import load_model_metadata
from ml.preprocessing.pipeline import DataPipeline

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=config.LOG_LEVEL,
    format=config.LOG_FORMAT
)


def evaluate_model(
    model,
    dataframe: pd.DataFrame,
    validator: DataValidator,
    encoder: CategoricalEncoder,
    scaler: NumericalScaler,
):
    """
    Evaluate a trained model on the provided dataframe.

    Returns:
        validation_loss,
        eta_metrics,
        delay_metrics
    """

    # pipeline = DataPipeline(
    #     validator=validator,
    #     encoder=encoder,
    #     scaler=scaler,
    # )

    # dataframe = pipeline.transform(dataframe)

    # _, validation_dataframe = pipeline.split_dataset(dataframe)

    # validation_dataset = RideDataset(
    #     dataframe=validation_dataframe,
    # )
    validation_dataset = RideDataset(
        dataframe=dataframe,
    )

    validation_loader = create_dataloader(
        dataset=validation_dataset,
        shuffle=False,
    )

    criterion = MultiTaskLoss()

    trainer = Trainer(
        model=model,
        optimizer=None,
        criterion=criterion,
        train_loader=None,
        validation_loader=validation_loader,
    )

    return trainer.evaluate(validation_loader)
    
def main() -> None:
    logger.info("Loading model metadata...")

    metadata = load_model_metadata(
        config.SAVED_MODEL_DIR / config.MODEL_METADATA_FILE,
        )
        
    logger.info("Building model...")

    model = RideETANetwork(
        numerical_feature_count=metadata["numerical_feature_count"],
        categorical_cardinalities=metadata["categorical_cardinalities"],
        )

    logger.info("Loading model weights...")

    model.load_state_dict(
        torch.load(
            config.SAVED_MODEL_DIR / config.CHECKPOINT_NAME,
            map_location=config.DEVICE,
        )
    )

    model.to(config.DEVICE)
    logger.info("Loading preprocessing artifacts...")

    validator = DataValidator()
    encoder = CategoricalEncoder()
    scaler = NumericalScaler()

    encoder.load()
    scaler.load()

    logger.info("Loading dataset...")
    dataframe = pd.read_csv(config.TRAIN_DATA_FILE)

    
    
    pipeline = DataPipeline(
    validator=validator,
    encoder=encoder,
    scaler=scaler,
    )

    dataframe = pipeline.transform(dataframe)

    _, validation_dataframe = pipeline.split_dataset(dataframe)
    validation_loss, eta_metrics, delay_metrics = evaluate_model(
        model=model,
        dataframe=validation_dataframe,
        validator=validator,
        encoder=encoder,
        scaler=scaler,
        )

    logger.info("Validation Loss: %.4f", validation_loss)
    logger.info("ETA Metrics: %s", eta_metrics)
    logger.info("Delay Metrics: %s", delay_metrics)


if __name__ == "__main__":
    main()