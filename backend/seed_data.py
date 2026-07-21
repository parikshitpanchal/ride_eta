"""
Database Seeding Script.
Populates PostgreSQL database from initial CSV files.
"""
import sys
import logging
import pandas as pd
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.database import SessionLocal, engine, Base
from backend.models import (
    RawRideOrder,
    EngineeredRideOrder,
    Prediction,
    TrainingRun,
    TrainingEpoch,
    Driver,
    User,
)
from backend.services.driver_service import refresh_driver_stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("seed_data")

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

ENGINEERED_FEATURES = [
    "individual_trip_ratio", "average_delay", "driver_avg_pickup_delay",
    "driver_avg_trip_delay", "driver_last_7_day_delay_rate", "driver_last_30_day_delay_rate",
    "driver_route_avg_delay", "driver_zone_avg_pickup_delay", "driver_zone_avg_trip_delay",
    "historical_route_ratio_avg", "driver_reliability", "driver_risk_score",
    "driver_efficiency_score", "driver_recent_delay_rate", "route_frequency",
    "historical_route_delay_rate", "route_reliability_score",
]


def seed_database():
    logger.info("Initializing database tables...")
    Base.metadata.create_all(bind=engine)

    db = SessionLocal()
    try:
        csv_path = PROJECT_ROOT / "data" / "ride_orders_engineered_D30K.csv"
        if not csv_path.exists():
            logger.error(f"Seed file not found: {csv_path}")
            return

        logger.info(f"Reading {csv_path}...")
        df = pd.read_csv(csv_path)

        existing_count = db.query(RawRideOrder).count()
        if existing_count > 0:
            logger.info(f"Database already has {existing_count} raw orders. Skipping order seed.")
        else:
            logger.info(f"Seeding {len(df)} orders into PostgreSQL...")
            raw_objects = []
            for idx, row in df.iterrows():
                raw_order = RawRideOrder(
                    order_id=str(row["order_id"]),
                    booking_timestamp=str(row["booking_timestamp"]),
                    hour=int(row["hour"]) if pd.notna(row.get("hour")) else None,
                    day_of_week=int(row["day_of_week"]) if pd.notna(row.get("day_of_week")) else None,
                    week_of_year=int(row["week_of_year"]) if pd.notna(row.get("week_of_year")) else None,
                    month=int(row["month"]) if pd.notna(row.get("month")) else None,
                    is_weekend=int(row["is_weekend"]) if pd.notna(row.get("is_weekend")) else None,
                    festival_flag=int(row["festival_flag"]) if pd.notna(row.get("festival_flag")) else None,
                    holiday_flag=int(row["holiday_flag"]) if pd.notna(row.get("holiday_flag")) else None,
                    is_peak_hour=int(row["is_peak_hour"]) if pd.notna(row.get("is_peak_hour")) else None,
                    pickup_zone=str(row["pickup_zone"]),
                    drop_zone=str(row["drop_zone"]),
                    pickup_latitude=float(row["pickup_latitude"]) if pd.notna(row.get("pickup_latitude")) else None,
                    pickup_longitude=float(row["pickup_longitude"]) if pd.notna(row.get("pickup_longitude")) else None,
                    drop_latitude=float(row["drop_latitude"]) if pd.notna(row.get("drop_latitude")) else None,
                    drop_longitude=float(row["drop_longitude"]) if pd.notna(row.get("drop_longitude")) else None,
                    driver_to_pickup_distance_km=float(row["driver_to_pickup_distance_km"]) if pd.notna(row.get("driver_to_pickup_distance_km")) else None,
                    trip_distance_km=float(row["trip_distance_km"]) if pd.notna(row.get("trip_distance_km")) else None,
                    driver_id=str(row["driver_id"]),
                    driver_base_rating=float(row["driver_base_rating"]) if pd.notna(row.get("driver_base_rating")) else None,
                    driver_experience_days=int(row["driver_experience_days"]) if pd.notna(row.get("driver_experience_days")) else None,
                    driver_acceptance_rate=float(row["driver_acceptance_rate"]) if pd.notna(row.get("driver_acceptance_rate")) else None,
                    driver_cancellation_rate=float(row["driver_cancellation_rate"]) if pd.notna(row.get("driver_cancellation_rate")) else None,
                    driver_total_rides=int(row["driver_total_rides"]) if pd.notna(row.get("driver_total_rides")) else None,
                    google_pickup_eta_minutes=float(row["google_pickup_eta_minutes"]) if pd.notna(row.get("google_pickup_eta_minutes")) else None,
                    google_trip_eta_minutes=float(row["google_trip_eta_minutes"]) if pd.notna(row.get("google_trip_eta_minutes")) else None,
                    time_of_day=str(row["time_of_day"]),
                    pickup_h3_cell=str(row["pickup_h3_cell"]),
                    drop_h3_cell=str(row["drop_h3_cell"]),
                    actual_pickup_time_minutes=float(row["actual_pickup_time_minutes"]) if pd.notna(row.get("actual_pickup_time_minutes")) else None,
                    actual_trip_time_minutes=float(row["actual_trip_time_minutes"]) if pd.notna(row.get("actual_trip_time_minutes")) else None,
                    actual_total_time_minutes=float(row["actual_total_time_minutes"]) if pd.notna(row.get("actual_total_time_minutes")) else None,
                    pickup_delay_minutes=float(row["pickup_delay_minutes"]) if pd.notna(row.get("pickup_delay_minutes")) else None,
                    trip_delay_minutes=float(row["trip_delay_minutes"]) if pd.notna(row.get("trip_delay_minutes")) else None,
                    total_delay_minutes=float(row["total_delay_minutes"]) if pd.notna(row.get("total_delay_minutes")) else None,
                    order_delayed=int(row["order_delayed"]) if pd.notna(row.get("order_delayed")) else None,
                    delay_severity=str(row["delay_severity"]),
                    is_engineered=True,
                )
                raw_objects.append(raw_order)

            db.bulk_save_objects(raw_objects)
            db.commit()
            logger.info("Raw orders bulk inserted.")

            # Bulk insert engineered features
            logger.info("Seeding engineered features...")
            raw_orders_map = {r.order_id: r.id for r in db.query(RawRideOrder.order_id, RawRideOrder.id).all()}
            eng_objects = []
            for idx, row in df.iterrows():
                order_id = str(row["order_id"])
                raw_id = raw_orders_map.get(order_id)
                if not raw_id:
                    continue
                eng_order = EngineeredRideOrder(
                    raw_order_id=raw_id,
                    **{feat: float(row[feat]) if pd.notna(row.get(feat)) else None for feat in ENGINEERED_FEATURES}
                )
                eng_objects.append(eng_order)

            db.bulk_save_objects(eng_objects)
            db.commit()
            logger.info("Engineered features bulk inserted.")

        # Seed Predictions if predicted CSV exists
        pred_csv = PROJECT_ROOT / "data" / "predicted_ride_orders_test.csv"
        if pred_csv.exists() and db.query(Prediction).count() == 0:
            logger.info(f"Seeding predictions from {pred_csv}...")
            pred_df = pd.read_csv(pred_csv)
            pred_objects = []
            for _, row in pred_df.iterrows():
                pred = Prediction(
                    order_id=str(row.get("order_id", "")),
                    booking_timestamp=str(row.get("booking_timestamp", "")),
                    actual_total_time_minutes=float(row.get("actual_total_time_minutes", 0)),
                    google_total_eta_minutes=float(row.get("google_total_eta_minutes", 0)),
                    total_delay_minutes=float(row.get("total_delay_minutes", 0)),
                    order_delayed=int(row.get("order_delayed", 0)),
                    predicted_eta_minutes=float(row.get("predicted_eta_minutes", 0)),
                    predicted_delay_probability=float(row.get("predicted_delay_probability", 0)),
                    predicted_delay=int(row.get("predicted_delay", 0)),
                )
                pred_objects.append(pred)
            db.bulk_save_objects(pred_objects)
            db.commit()
            logger.info("Predictions bulk inserted.")

        # Refresh Driver Stats
        logger.info("Refreshing driver analytics stats...")
        refresh_driver_stats(db)

        # Seed Default Users for RBAC
        logger.info("Seeding default user accounts...")
        from backend.services.auth_service import get_password_hash
        default_users = [
            ("admin", "admin@rideeta.com", "admin123", "admin"),
            ("ds", "ds@rideeta.com", "ds123", "data_scientist"),
            ("viewer", "viewer@rideeta.com", "viewer123", "viewer"),
        ]
        for uname, email, pword, role in default_users:
            if not db.query(User).filter(User.username == uname).first():
                user = User(
                    username=uname,
                    email=email,
                    hashed_password=get_password_hash(pword),
                    role=role,
                )
                db.add(user)
        db.commit()
        logger.info("Default users seeded (admin, ds, viewer).")

        logger.info("Database seeding completed successfully!")

    except Exception as e:
        logger.error(f"Seeding failed: {e}", exc_info=True)
        db.rollback()
    finally:
        db.close()


if __name__ == "__main__":
    seed_database()
