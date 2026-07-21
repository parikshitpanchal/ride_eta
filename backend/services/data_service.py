"""
Data service — loads training data from PostgreSQL for the training pipeline.
JOINs raw_ride_orders with engineered_ride_orders to construct the full training DataFrame.
"""
import logging

import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy import text

logger = logging.getLogger(__name__)


def load_training_data_from_db(db: Session) -> pd.DataFrame | None:
    """
    Load full training-ready data by JOINing raw_ride_orders with engineered_ride_orders.
    Returns a DataFrame with all columns needed for the ML pipeline.
    """
    query = text("""
        SELECT
            r.order_id, r.booking_timestamp,
            r.hour, r.day_of_week, r.week_of_year, r.month,
            r.is_weekend, r.festival_flag, r.holiday_flag, r.is_peak_hour,
            r.pickup_zone, r.drop_zone,
            r.pickup_latitude, r.pickup_longitude, r.drop_latitude, r.drop_longitude,
            r.driver_to_pickup_distance_km, r.trip_distance_km,
            r.driver_id, r.driver_base_rating, r.driver_experience_days,
            r.driver_acceptance_rate, r.driver_cancellation_rate, r.driver_total_rides,
            r.google_pickup_eta_minutes, r.google_trip_eta_minutes,
            r.time_of_day, r.pickup_h3_cell, r.drop_h3_cell,
            r.actual_pickup_time_minutes, r.actual_trip_time_minutes,
            r.actual_total_time_minutes,
            r.pickup_delay_minutes, r.trip_delay_minutes, r.total_delay_minutes,
            r.order_delayed, r.delay_severity,
            e.individual_trip_ratio, e.average_delay,
            e.driver_avg_pickup_delay, e.driver_avg_trip_delay,
            e.driver_last_7_day_delay_rate, e.driver_last_30_day_delay_rate,
            e.driver_route_avg_delay,
            e.driver_zone_avg_pickup_delay, e.driver_zone_avg_trip_delay,
            e.historical_route_ratio_avg,
            e.driver_reliability, e.driver_risk_score, e.driver_efficiency_score,
            e.driver_recent_delay_rate, e.route_frequency,
            e.historical_route_delay_rate, e.route_reliability_score
        FROM raw_ride_orders r
        INNER JOIN engineered_ride_orders e ON e.raw_order_id = r.id
        ORDER BY r.booking_timestamp ASC
    """)

    result = db.execute(query)
    rows = result.fetchall()

    if not rows:
        logger.warning("No training data found in database")
        return None

    columns = result.keys()
    df = pd.DataFrame(rows, columns=columns)
    logger.info(f"Loaded {len(df)} training rows from database")
    return df
