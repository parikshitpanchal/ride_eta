import logging
import pandas as pd
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
from torch.optim import Adam

from configs import config
from datasets.dataloader import create_dataloader
from datasets.ride_dataset import RideDataset
from losses.losses import MultiTaskLoss
from models.ride_eta_network import RideETANetwork
from preprocessing.encoder import CategoricalEncoder
from preprocessing.scaler import NumericalScaler
from preprocessing.validator import DataValidator
from preprocessing.pipeline import DataPipeline
from trainer.trainer import Trainer
from predictor.predictor import Predictor
from utils.model_metadata import save_model_metadata

logger = logging.getLogger("run_pipeline")

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

    # =========================================================================
    # STEP 1: PREPROCESS & LOAD TRAINING DATA
    # =========================================================================
    logger.info("--- STEP 1: Loading and preprocessing training dataset ---")
    train_df = pd.read_csv(config.TRAIN_DATA_FILE)

    validator = DataValidator()
    encoder = CategoricalEncoder()
    scaler = NumericalScaler()

    pipeline = DataPipeline(
        validator=validator,
        encoder=encoder,
        scaler=scaler
    )

    # Fit and transform the training data
    train_df = pipeline.fit_transform(train_df)

    # Use 100% of the training dataset for training.
    # Since prediction data is in a separate file, we do not split the training file.
    train_dataframe = train_df.copy()

    logger.info("Training samples: %d", len(train_dataframe))

    train_dataset = RideDataset(dataframe=train_dataframe)

    train_loader = create_dataloader(dataset=train_dataset, shuffle=config.SHUFFLE)
    # Pass the training dataset as validation_loader as well since the Trainer class requires it.
    validation_loader = create_dataloader(dataset=train_dataset, shuffle=False)

    # =========================================================================
    # STEP 2: BUILD AND TRAIN THE MODEL
    # =========================================================================
    logger.info("--- STEP 2: Building and training the model ---")
    model = RideETANetwork(
        numerical_feature_count=len(config.NUMERICAL_COLUMNS),
        categorical_cardinalities=encoder.get_cardinalities(),
    )

    optimizer = Adam(
        params=model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    criterion = MultiTaskLoss()

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        validation_loader=validation_loader,
    )

    logger.info("Starting training loop...")
    history = trainer.train()

    # =========================================================================
    # STEP 3: SAVE MODEL ARTIFACTS
    # =========================================================================
    logger.info("--- STEP 3: Saving preprocessing and model artifacts ---")
    save_model_metadata(
        file_path=config.SAVED_MODEL_DIR / config.MODEL_METADATA_FILE,
        numerical_feature_count=len(config.NUMERICAL_COLUMNS),
        categorical_cardinalities=encoder.get_cardinalities(),
    )
    encoder.save()
    scaler.save()
    logger.info("All preprocessing and model metadata saved successfully.")

    # =========================================================================
    # STEP 4: PREPROCESS PREDICTION/TEST DATA
    # =========================================================================
    logger.info("--- STEP 4: Loading and preprocessing prediction dataset ---")
    test_df = pd.read_csv(config.TEST_DATA_FILE)

    # We use transform (NOT fit_transform) to apply the fitted scaler/encoder
    test_df = pipeline.transform(test_df)
    
    test_dataset = RideDataset(dataframe=test_df)
    test_loader = create_dataloader(dataset=test_dataset, shuffle=False)

    # =========================================================================
    # STEP 5: EVALUATE MODEL ON PREDICTION DATA
    # =========================================================================
    logger.info("--- STEP 5: Evaluating model on the prediction dataset ---")
    
    # Reload the best weights saved during training
    model.load_state_dict(
        torch.load(
            config.SAVED_MODEL_DIR / config.CHECKPOINT_NAME,
            map_location=config.DEVICE
        )
    )
    model.to(config.DEVICE)

    test_loss, eta_metrics, delay_metrics = trainer.evaluate(test_loader)
    logger.info("Prediction Dataset Loss: %.4f", test_loss)
    logger.info("Prediction Dataset ETA Metrics: %s", eta_metrics)
    logger.info("Prediction Dataset Delay Metrics: %s", delay_metrics)

    # =========================================================================
    # STEP 6: RUN INFERENCE AND PRINT PREDICTIONS
    # =========================================================================
    logger.info("--- STEP 6: Running inference on prediction dataset ---")
    predictor = Predictor(model=model, device=config.DEVICE)
    predictions = predictor.predict(test_loader)
    
    predictions["delay"] = (predictions["delay_probability"] >= config.DELAY_THRESHOLD).int()

    logger.info("Generated %d predictions.", len(predictions["eta"]))

    logger.info("--- First 5 Predictions ---")
    for index in range(5):
        logger.info(
            "Ride %d | Predicted ETA: %.2f minutes | Predicted Delayed: %d (Probability: %.2f%%)",
            index + 1,
            predictions["eta"][index].item(),
            predictions["delay"][index].item(),
            predictions["delay_probability"][index].item() * 100
        )

    # Also check if any are predicted as delayed and print them
    delayed_indices = (predictions["delay"] == 1).nonzero(as_tuple=True)[0].tolist()
    if delayed_indices:
        logger.info("--- Examples of Predicted Delayed Rides (Delayed: 1) ---")
        for index in delayed_indices[:5]:
            logger.info(
                "Ride %d | Predicted ETA: %.2f minutes | Predicted Delayed: %d (Probability: %.2f%%)",
                index + 1,
                predictions["eta"][index].item(),
                predictions["delay"][index].item(),
                predictions["delay_probability"][index].item() * 100
            )

    logger.info("Pipeline execution completed successfully.")

if __name__ == "__main__":
    main()
