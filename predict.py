import logging

import pandas as pd
# pyrefly: ignore [missing-import]
import torch

from configs import config
# pyrefly: ignore [missing-import]
from datasets.dataloader import create_dataloader
# pyrefly: ignore [missing-import]
from datasets.ride_dataset import RideDataset
from models.ride_eta_network import RideETANetwork
from predictor.predictor import Predictor
from preprocessing.encoder import CategoricalEncoder
from preprocessing.pipeline import DataPipeline
from preprocessing.scaler import NumericalScaler
from preprocessing.validator import DataValidator
from utils.model_metadata import load_model_metadata

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

    encoder = CategoricalEncoder()
    encoder.load()

    scaler = NumericalScaler()
    scaler.load()

    validator = DataValidator()

    pipeline = DataPipeline(
        validator=validator,
        encoder=encoder,
        scaler=scaler,
    )

    logger.info("Loading prediction dataset...")

    dataframe = pd.read_csv(config.TEST_DATA_FILE)
    dataframe = pipeline.transform(dataframe)

    prediction_dataset = RideDataset(dataframe=dataframe,)

    prediction_loader = create_dataloader(
        dataset=prediction_dataset,
        shuffle=False,
    )

    logger.info("Running predictions...")

    predictor = Predictor(
        model=model,
        device=config.DEVICE,
    )

    predictions = predictor.predict(prediction_loader)
    predictions["delay"] = (predictions["delay_probability"] >= config.DELAY_THRESHOLD).int()
    
    logger.info("Generated %d predictions.",len(predictions["eta"]),)

    for index in range(50):
        logger.info(
            "Ride %d | ETA: %.2f minutes | Delayed: %d",
            index + 1,
            predictions["eta"][index].item(),
            predictions["delay"][index].item(),
        )



    output_df = pd.read_csv(config.TEST_DATA_FILE)
    output_df["google_total_eta_minutes"] = output_df["google_pickup_eta_minutes"] + output_df["google_trip_eta_minutes"]
    output_df=output_df[[config.ORDER_ID_COLUMN,config.TIMESTAMP_COLUMN,"total_delay_minutes","order_delayed","actual_total_time_minutes","google_total_eta_minutes"]]
    
    output_df["predicted_eta_minutes"] = predictions["eta"].numpy()
    output_df["predicted_delay_probability"] = predictions["delay_probability"].numpy()
    output_df["predicted_delay"] = predictions["delay"].numpy()


    # Save to a new CSV file
    output_path = config.DATA_DIR / "predicted_ride_orders_test.csv"
    output_df.to_csv(output_path, index=False)
    logger.info("Predictions saved successfully to: %s", output_path)


if __name__ == "__main__":
    main()