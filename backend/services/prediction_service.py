"""
Prediction service — wraps the existing ML predictor.
Runs model on an uploaded CSV and saves results to the predictions table.
"""
import sys
import logging

import pandas as pd
import torch
from sqlalchemy.orm import Session

from backend.config import ML_DIR
from backend.models.prediction import Prediction
from backend.models.training_run import TrainingRun

if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

logger = logging.getLogger(__name__)


def run_prediction_on_dataframe(df: pd.DataFrame, db: Session) -> dict:
    """Run model predictions on a DataFrame and save results to DB."""
    from ml.configs import config as ml_config
    from ml.preprocessing.encoder import CategoricalEncoder
    from ml.preprocessing.scaler import NumericalScaler
    from ml.preprocessing.validator import DataValidator
    from ml.preprocessing.pipeline import DataPipeline
    from ml.datasets.ride_dataset import RideDataset
    from ml.datasets.dataloader import create_dataloader
    from ml.models.ride_eta_network import RideETANetwork
    from ml.predictor.predictor import Predictor
    from ml.utils.model_metadata import load_model_metadata

    # Load model metadata and weights
    metadata = load_model_metadata(ml_config.SAVED_MODEL_DIR / ml_config.MODEL_METADATA_FILE)
    model = RideETANetwork(
        numerical_feature_count=metadata["numerical_feature_count"],
        categorical_cardinalities=metadata["categorical_cardinalities"],
    )
    model.load_state_dict(
        torch.load(ml_config.SAVED_MODEL_DIR / ml_config.CHECKPOINT_NAME, map_location=ml_config.DEVICE)
    )
    model.to(ml_config.DEVICE)

    # Load preprocessing artifacts
    encoder = CategoricalEncoder()
    encoder.load()
    scaler = NumericalScaler()
    scaler.load()
    validator = DataValidator()
    pipeline = DataPipeline(validator=validator, encoder=encoder, scaler=scaler)

    # Keep original data for output
    original_df = df.copy()

    # Preprocess
    df = pipeline.transform(df)

    # Create dataset and loader
    dataset = RideDataset(dataframe=df)
    loader = create_dataloader(dataset=dataset, shuffle=False)

    # Run predictions
    predictor = Predictor(model=model, device=ml_config.DEVICE)
    predictions = predictor.predict(loader)
    predictions["delay"] = (predictions["delay_probability"] >= ml_config.DELAY_THRESHOLD).int()

    # Get latest completed training run
    latest_run = (
        db.query(TrainingRun)
        .filter(TrainingRun.status == "completed")
        .order_by(TrainingRun.completed_at.desc())
        .first()
    )
    training_run_id = latest_run.id if latest_run else None

    # Clear all existing predictions so only the latest run is stored
    deleted = db.query(Prediction).delete()
    db.flush()
    logger.info(f"Cleared {deleted} old predictions from database")

    # Build output and save to DB
    google_total = original_df["google_pickup_eta_minutes"] + original_df["google_trip_eta_minutes"]
    rows_inserted = 0

    for i in range(len(original_df)):
        pred = Prediction(
            order_id=original_df.iloc[i].get("order_id", f"PRED_{i:06d}"),
            booking_timestamp=str(original_df.iloc[i].get("booking_timestamp", "")),
            actual_total_time_minutes=float(original_df.iloc[i].get("actual_total_time_minutes", 0)),
            google_total_eta_minutes=float(google_total.iloc[i]),
            total_delay_minutes=float(original_df.iloc[i].get("total_delay_minutes", 0)),
            order_delayed=int(original_df.iloc[i].get("order_delayed", 0)),
            predicted_eta_minutes=float(predictions["eta"][i].item()),
            predicted_delay_probability=float(predictions["delay_probability"][i].item()),
            predicted_delay=int(predictions["delay"][i].item()),
            training_run_id=training_run_id,
        )
        db.add(pred)
        rows_inserted += 1

    db.commit()
    logger.info(f"Saved {rows_inserted} predictions to database")

    return {
        "message": f"Generated and saved {rows_inserted} predictions",
        "rows_inserted": rows_inserted,
        "training_run_id": training_run_id,
    }
