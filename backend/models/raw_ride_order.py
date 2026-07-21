"""
ORM model for the raw_ride_orders table.
Stores raw ride data before feature engineering.
"""
from sqlalchemy import Column, Integer, String, Float, SmallInteger, Boolean, DateTime
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from backend.database import Base


class RawRideOrder(Base):
    __tablename__ = "raw_ride_orders"

    id = Column(Integer, primary_key=True, autoincrement=True)
    order_id = Column(String(20), unique=True, nullable=False, index=True)
    booking_timestamp = Column(String(50), nullable=False)

    # Time features
    hour = Column(SmallInteger)
    day_of_week = Column(SmallInteger)
    week_of_year = Column(SmallInteger)
    month = Column(SmallInteger)
    is_weekend = Column(SmallInteger)
    festival_flag = Column(SmallInteger)
    holiday_flag = Column(SmallInteger)
    is_peak_hour = Column(SmallInteger)

    # Location features
    pickup_zone = Column(String(50))
    drop_zone = Column(String(50))
    pickup_latitude = Column(Float)
    pickup_longitude = Column(Float)
    drop_latitude = Column(Float)
    drop_longitude = Column(Float)

    # Distance features
    driver_to_pickup_distance_km = Column(Float)
    trip_distance_km = Column(Float)

    # Driver features
    driver_id = Column(String(20), index=True)
    driver_base_rating = Column(Float)
    driver_experience_days = Column(Integer)
    driver_acceptance_rate = Column(Float)
    driver_cancellation_rate = Column(Float)
    driver_total_rides = Column(Integer)

    # Google ETA features
    google_pickup_eta_minutes = Column(Float)
    google_trip_eta_minutes = Column(Float)

    # Categorical features
    time_of_day = Column(String(20))
    pickup_h3_cell = Column(String(20))
    drop_h3_cell = Column(String(20))

    # Actual time & delay (ground truth)
    actual_pickup_time_minutes = Column(Float)
    actual_trip_time_minutes = Column(Float)
    actual_total_time_minutes = Column(Float)
    pickup_delay_minutes = Column(Float)
    trip_delay_minutes = Column(Float)
    total_delay_minutes = Column(Float)
    order_delayed = Column(SmallInteger)
    delay_severity = Column(String(5))

    # Processing flag
    is_engineered = Column(Boolean, default=False)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationship to engineered features
    engineered_features = relationship("EngineeredRideOrder", back_populates="raw_order", uselist=False)
