"""
ORM model for the engineered_ride_orders table.
Stores only computed/engineered features + FK to raw_ride_orders.
"""
from sqlalchemy import Column, Integer, Float, DateTime, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from backend.database import Base


class EngineeredRideOrder(Base):
    __tablename__ = "engineered_ride_orders"

    id = Column(Integer, primary_key=True, autoincrement=True)
    raw_order_id = Column(Integer, ForeignKey("raw_ride_orders.id"), unique=True, nullable=False, index=True)

    # Engineered features
    individual_trip_ratio = Column(Float)
    average_delay = Column(Float)
    driver_avg_pickup_delay = Column(Float)
    driver_avg_trip_delay = Column(Float)
    driver_last_7_day_delay_rate = Column(Float)
    driver_last_30_day_delay_rate = Column(Float)
    driver_route_avg_delay = Column(Float)
    driver_zone_avg_pickup_delay = Column(Float)
    driver_zone_avg_trip_delay = Column(Float)
    historical_route_ratio_avg = Column(Float)
    driver_reliability = Column(Float)
    driver_risk_score = Column(Float)
    driver_efficiency_score = Column(Float)
    driver_recent_delay_rate = Column(Float)
    route_frequency = Column(Float)
    historical_route_delay_rate = Column(Float)
    route_reliability_score = Column(Float)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationship back to raw order
    raw_order = relationship("RawRideOrder", back_populates="engineered_features")
