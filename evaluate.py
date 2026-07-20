import logging

import pandas as pd
# pyrefly: ignore [missing-import]
import torch

from configs import config
# pyrefly: ignore [missing-import]
from datasets.dataloader import create_dataloader
# pyrefly: ignore [missing-import]
from datasets.ride_dataset import RideDataset
# pyrefly: ignore [missing-import]
from losses.losses import MultiTaskLoss
from models.ride_eta_network import RideETANetwork
from preprocessing.encoder import CategoricalEncoder
from preprocessing.scaler import NumericalScaler
from preprocessing.validator import DataValidator
from trainer.trainer import Trainer
from utils.model_metadata import load_model_metadata
from preprocessing.pipeline import DataPipeline

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=config.LOG_LEVEL,
    format=config.LOG_FORMAT
)

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

    train_dataframe, validation_dataframe = (pipeline.split_dataset(dataframe,))
    logger.info("Creating train and validation dataset...")

    train_dataset = RideDataset(
        dataframe=train_dataframe,
    )

    validation_dataset = RideDataset(
        dataframe=validation_dataframe,
    )

    logger.info("Creating train and validation dataloader...")

    train_loader= create_dataloader(
        dataset=train_dataset,
        shuffle=False,
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
        train_loader=train_loader,
        validation_loader=validation_loader,
    )
    
    
    validation_loss, eta_metrics, delay_metrics = trainer.evaluate(validation_loader)
    logger.info("Validation Loss: %.4f", validation_loss)
    logger.info("ETA Metrics: %s", eta_metrics)
    logger.info("Delay Metrics: %s", delay_metrics)

    # logger.info("Creating test dataset...")
    # train_end = int(len(dataframe) * config.TRAIN_SIZE)
    # validation_end = train_end + int(len(dataframe) * config.VALIDATION_SIZE)
    # test_dataframe = dataframe.iloc[validation_end:].copy()


if __name__ == "__main__":
    main()