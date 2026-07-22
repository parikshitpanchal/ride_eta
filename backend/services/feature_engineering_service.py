"""
Feature engineering service — wraps the existing feature_engineering.py.
Reads raw orders from DB, runs feature engineering, saves to engineered_ride_orders.
"""
import sys
import logging

import pandas as pd
from sqlalchemy.orm import Session

from backend.config import ML_DIR
from backend.models.raw_ride_order import RawRideOrder
from backend.models.engineered_ride_order import EngineeredRideOrder

if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

logger = logging.getLogger(__name__)

# All raw columns that exist on the RawRideOrder model (for DataFrame construction)
RAW_COLUMNS = [
    "order_id", "booking_timestamp", "hour", "day_of_week", "week_of_year", "month",
    "is_weekend", "festival_flag", "holiday_flag", "is_peak_hour",
    "pickup_zone", "drop_zone", "pickup_latitude", "pickup_longitude",
    "drop_latitude", "drop_longitude", "driver_to_pickup_distance_km", "trip_distance_km",
    "driver_id", "driver_base_rating", "driver_experience_days",
    "driver_acceptance_rate", "driver_cancellation_rate", "driver_total_rides",
    "google_pickup_eta_minutes", "google_trip_eta_minutes",
    "time_of_day", "pickup_h3_cell", "drop_h3_cell",
    "actual_pickup_time_minutes", "actual_trip_time_minutes", "actual_total_time_minutes",
    "pickup_delay_minutes", "trip_delay_minutes", "total_delay_minutes",
    "order_delayed", "delay_severity",
]

# Engineered feature column names (output of feature_engineering.py)
ENGINEERED_FEATURES = [
    "individual_trip_ratio", "average_delay", "driver_avg_pickup_delay",
    "driver_avg_trip_delay", "driver_last_7_day_delay_rate", "driver_last_30_day_delay_rate",
    "driver_route_avg_delay", "driver_zone_avg_pickup_delay", "driver_zone_avg_trip_delay",
    "historical_route_ratio_avg", "driver_reliability", "driver_risk_score",
    "driver_efficiency_score", "driver_recent_delay_rate", "route_frequency",
    "historical_route_delay_rate", "route_reliability_score",
]


def run_feature_engineering_on_db(db: Session) -> dict:
    """
    Fetch unprocessed raw orders from DB, run feature engineering,
    and save computed features to engineered_ride_orders table.
    """
    # Fetch unprocessed orders
    unprocessed = db.query(RawRideOrder).filter(RawRideOrder.is_engineered == False).all()

    if not unprocessed:
        return {"message": "No unprocessed orders found", "rows_processed": 0}

    logger.info(f"Processing {len(unprocessed)} raw orders for feature engineering")

    # Convert to DataFrame
    rows = []
    id_map = {}  # Map order_id -> raw_ride_orders.id for FK
    for order in unprocessed:
        row = {col: getattr(order, col) for col in RAW_COLUMNS}
        rows.append(row)
        id_map[order.order_id] = order.id

    df = pd.DataFrame(rows)

    # Run feature engineering
    from ml.feature_engineering import FeatureEngineer
    engineer = FeatureEngineer()
    df = engineer.engineer_features(df)

    # Save engineered features to DB
    rows_saved = 0
    for _, row in df.iterrows():
        raw_id = id_map.get(row["order_id"])
        if raw_id is None:
            continue

        # Check if already exists
        existing = db.query(EngineeredRideOrder).filter(
            EngineeredRideOrder.raw_order_id == raw_id
        ).first()
        if existing:
            continue

        eng_order = EngineeredRideOrder(
            raw_order_id=raw_id,
            **{feat: float(row[feat]) if pd.notna(row.get(feat)) else None for feat in ENGINEERED_FEATURES}
        )
        db.add(eng_order)

        # Mark raw order as engineered
        raw_order = db.query(RawRideOrder).filter(RawRideOrder.id == raw_id).first()
        if raw_order:
            raw_order.is_engineered = True

        rows_saved += 1

    db.commit()
    logger.info(f"Feature engineering complete. {rows_saved} rows saved.")

    return {"message": f"Feature engineering completed", "rows_processed": rows_saved}
