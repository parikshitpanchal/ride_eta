"""
Admin API router.
Handles CSV upload, feature engineering trigger, prediction trigger, and config management.
"""
import io
import logging

import pandas as pd
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func

from backend.database import get_db
from backend.models.raw_ride_order import RawRideOrder
from backend.models.engineered_ride_order import EngineeredRideOrder
from backend.models.prediction import Prediction
from backend.models.training_run import TrainingRun
from backend.models.driver import Driver
from backend.schemas.admin import (
    ConfigResponse,
    ConfigUpdateRequest,
    PipelineStatusResponse,
    UploadResponse,
)
from backend.services.auth_service import require_role

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin", tags=["Admin"])


# In-memory config overrides (per session; persists until server restart)
_config_overrides: dict = {}


def _get_ml_config():
    """Import the ML config module."""
    import sys
    from backend.config import ML_DIR
    if str(ML_DIR) not in sys.path:
        sys.path.insert(0, str(ML_DIR))
    from ml.configs import config as ml_config
    return ml_config


@router.get("/config", response_model=ConfigResponse)
def get_config():
    """Get current ML configuration values."""
    ml_config = _get_ml_config()
    return ConfigResponse(
        delay_threshold=_config_overrides.get("delay_threshold", ml_config.DELAY_THRESHOLD),
        learning_rate=_config_overrides.get("learning_rate", ml_config.LEARNING_RATE),
        epochs=_config_overrides.get("epochs", ml_config.EPOCHS),
        batch_size=_config_overrides.get("batch_size", ml_config.BATCH_SIZE),
        weight_decay=_config_overrides.get("weight_decay", ml_config.WEIGHT_DECAY),
        train_size=ml_config.TRAIN_SIZE,
        validation_size=ml_config.VALIDATION_SIZE,
        dropout=_config_overrides.get("dropout", ml_config.DROPOUT),
        hidden_layers=list(ml_config.HIDDEN_LAYERS),
    )


@router.put("/config", response_model=ConfigResponse)
def update_config(request: ConfigUpdateRequest):
    """Update ML configuration values (applied at next training run)."""
    ml_config = _get_ml_config()

    if request.delay_threshold is not None:
        _config_overrides["delay_threshold"] = request.delay_threshold
        ml_config.DELAY_THRESHOLD = request.delay_threshold

    if request.learning_rate is not None:
        _config_overrides["learning_rate"] = request.learning_rate
        ml_config.LEARNING_RATE = request.learning_rate

    if request.epochs is not None:
        _config_overrides["epochs"] = request.epochs
        ml_config.EPOCHS = request.epochs

    if request.batch_size is not None:
        _config_overrides["batch_size"] = request.batch_size
        ml_config.BATCH_SIZE = request.batch_size

    if request.weight_decay is not None:
        _config_overrides["weight_decay"] = request.weight_decay
        ml_config.WEIGHT_DECAY = request.weight_decay

    if request.dropout is not None:
        _config_overrides["dropout"] = request.dropout
        ml_config.DROPOUT = request.dropout

    return get_config()


@router.post("/upload-csv", response_model=UploadResponse)
async def upload_csv(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Upload a CSV file and insert rows into raw_ride_orders."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted")

    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))
    logger.info(f"Uploading CSV: {file.filename} with {len(df)} rows")

    rows_inserted = 0

    try:
        # 1. Delete engineered data first
        db.query(EngineeredRideOrder).delete()

        # 2. Delete raw data
        db.query(RawRideOrder).delete()

        db.commit()
        logger.info(f"EngineeredRideOrder and RawRideOrder is cleared successfully")
    
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error clearing database: {str(e)}")

    rows = []
    
    for _, row in df.iterrows():


        # Skip if order_id already exists
        # existing = db.query(RawRideOrder).filter(RawRideOrder.order_id == row.get("order_id")).first()
        # if existing:
        #     continue

        rows.append({
                "order_id":row.get("order_id"),
            "booking_timestamp":str(row.get("booking_timestamp", "")),
            "hour":int(row["hour"]) if pd.notna(row.get("hour")) else None,
            "day_of_week":int(row["day_of_week"]) if pd.notna(row.get("day_of_week")) else None,
            "week_of_year":int(row["week_of_year"]) if pd.notna(row.get("week_of_year")) else None,
            "month":int(row["month"]) if pd.notna(row.get("month")) else None,
            "is_weekend":int(row["is_weekend"]) if pd.notna(row.get("is_weekend")) else None,
            "festival_flag":int(row["festival_flag"]) if pd.notna(row.get("festival_flag")) else None,
            "holiday_flag":int(row["holiday_flag"]) if pd.notna(row.get("holiday_flag")) else None,
            "is_peak_hour":int(row["is_peak_hour"]) if pd.notna(row.get("is_peak_hour")) else None,
            "pickup_zone":row.get("pickup_zone"),
            "drop_zone":row.get("drop_zone"),
            "pickup_latitude":row.get("pickup_latitude"),
            "pickup_longitude":row.get("pickup_longitude"),
            "drop_latitude":row.get("drop_latitude"),
            "drop_longitude":row.get("drop_longitude"),
            "driver_to_pickup_distance_km":row.get("driver_to_pickup_distance_km"),
            "trip_distance_km":row.get("trip_distance_km"),
            "driver_id":row.get("driver_id"),
            "driver_base_rating":row.get("driver_base_rating"),
            "driver_experience_days":int(row["driver_experience_days"]) if pd.notna(row.get("driver_experience_days")) else None,
            "driver_acceptance_rate":row.get("driver_acceptance_rate"),
            "driver_cancellation_rate":row.get("driver_cancellation_rate"),
            "driver_total_rides":int(row["driver_total_rides"]) if pd.notna(row.get("driver_total_rides")) else None,
            "google_pickup_eta_minutes":row.get("google_pickup_eta_minutes"),
            "google_trip_eta_minutes":row.get("google_trip_eta_minutes"),
            "time_of_day":row.get("time_of_day"),
            "pickup_h3_cell":row.get("pickup_h3_cell"),
            "drop_h3_cell":row.get("drop_h3_cell"),
            "actual_pickup_time_minutes":row.get("actual_pickup_time_minutes"),
            "actual_trip_time_minutes":row.get("actual_trip_time_minutes"),
            "actual_total_time_minutes":row.get("actual_total_time_minutes"),
            "pickup_delay_minutes":row.get("pickup_delay_minutes"),
            "trip_delay_minutes":row.get("trip_delay_minutes"),
            "total_delay_minutes":row.get("total_delay_minutes"),
            "order_delayed":int(row["order_delayed"]) if pd.notna(row.get("order_delayed")) else None,
            "delay_severity":str(row.get("delay_severity", "")),
        })
        
        # db.add(order)
        rows_inserted += 1
    db.bulk_insert_mappings(RawRideOrder,rows)
    db.commit()
    logger.info(f"Inserted {rows_inserted} rows from {file.filename}")

    return UploadResponse(
        message=f"Successfully uploaded {file.filename}",
        rows_inserted=rows_inserted,
        filename=file.filename,
    )


@router.post("/run-feature-engineering")
def run_feature_engineering(db: Session = Depends(get_db)):
    """Trigger feature engineering on unprocessed raw orders."""
    from backend.services.feature_engineering_service import run_feature_engineering_on_db
    result = run_feature_engineering_on_db(db)
    return result


@router.post("/run-prediction")
async def run_prediction(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Run model predictions on an uploaded CSV file."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted")

    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))

    from backend.services.prediction_service import run_prediction_on_dataframe
    result = run_prediction_on_dataframe(df, db)
    return result


@router.get("/pipeline-status", response_model=PipelineStatusResponse)
def get_pipeline_status(db: Session = Depends(get_db)):
    """Get data pipeline status: counts of raw, engineered, predictions, etc."""
    total_raw = db.query(func.count(RawRideOrder.id)).scalar() or 0
    engineered = db.query(func.count(EngineeredRideOrder.id)).scalar() or 0
    unprocessed = db.query(func.count(RawRideOrder.id)).filter(RawRideOrder.is_engineered == False).scalar() or 0
    total_preds = db.query(func.count(Prediction.id)).scalar() or 0
    total_runs = db.query(func.count(TrainingRun.id)).scalar() or 0
    total_drivers = db.query(func.count(Driver.id)).scalar() or 0

    return PipelineStatusResponse(
        total_raw_orders=total_raw,
        engineered_orders=engineered,
        unprocessed_orders=unprocessed,
        total_predictions=total_preds,
        total_training_runs=total_runs,
        total_drivers=total_drivers,
    )
