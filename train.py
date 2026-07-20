# pyrefly: ignore [missing-import]
from torch.optim import Adam
import pandas as pd
import logging
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
from utils.model_metadata import save_model_metadata
from trainer.trainer import Trainer
from preprocessing.pipeline import DataPipeline

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=config.LOG_LEVEL,
    format=config.LOG_FORMAT
)

def set_random_seed() -> None:
    torch.manual_seed(config.RANDOM_SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.RANDOM_SEED)
        torch.cuda.manual_seed_all(config.RANDOM_SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info("Using device: %s", config.DEVICE)

def main() -> None:

    set_random_seed()
    logger.info("Loading dataset...")

    dataframe = pd.read_csv(config.TRAIN_DATA_FILE)

    validator = DataValidator()
    encoder = CategoricalEncoder()
    scaler = NumericalScaler()

    pipeline = DataPipeline(
        validator=validator,
        encoder=encoder,
        scaler=scaler)
    dataframe = pipeline.fit_transform(dataframe)

    train_dataframe, validation_dataframe = (
        pipeline.split_dataset(dataframe,)
        )
    
    logger.info("Splitting dataset...")

    # train_end = int(
    #     len(dataframe) * config.TRAIN_SIZE
    # )

    # validation_end = train_end + int(
    #     len(dataframe) * config.VALIDATION_SIZE
    # )

    # train_dataframe = dataframe.iloc[:train_end].copy()

    # validation_dataframe = dataframe.iloc[train_end:validation_end].copy()

    # test_dataframe = dataframe.iloc[validation_end:].copy()

    logger.info("Creating datasets...")

    train_dataset = RideDataset(dataframe=train_dataframe)

    validation_dataset = RideDataset(dataframe=validation_dataframe)

    # test_dataset = RideDataset(dataframe=test_dataframe)

    logger.info("Creating dataloaders...")

    train_loader = create_dataloader(
        dataset=train_dataset,
        shuffle=config.SHUFFLE,
    )

    validation_loader = create_dataloader(
        dataset=validation_dataset,
        shuffle=False,
    )

    # test_loader = create_dataloader(
    #     dataset=test_dataset,
    #     shuffle=False,
    # )

    logger.info("Building model...")

    model = RideETANetwork(
        numerical_feature_count=len(
            config.NUMERICAL_COLUMNS,
        ),
        categorical_cardinalities = encoder.get_cardinalities(),
    )

    logger.info("Building optimizer...")

    optimizer = Adam(
        params=model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    logger.info("Building loss function...")

    criterion = MultiTaskLoss()

    logger.info("Building trainer...")

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        validation_loader=validation_loader,
    )

    logger.info("Starting training...")

    history = trainer.train()

    logger.info("Saving preprocessing artifacts...")
    save_model_metadata(
        file_path=config.SAVED_MODEL_DIR / config.MODEL_METADATA_FILE,
        numerical_feature_count=len(config.NUMERICAL_COLUMNS),
        categorical_cardinalities=encoder.get_cardinalities(),
        )

    encoder.save()

    scaler.save()

    logger.info("Training completed successfully.")

if __name__ == "__main__":
    main()