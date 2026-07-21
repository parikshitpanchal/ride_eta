"""
ORM model for the drivers table.
Aggregated driver statistics for the driver analytics dashboard.
"""
from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy.sql import func

from backend.database import Base


class Driver(Base):
    __tablename__ = "drivers"

    id = Column(Integer, primary_key=True, autoincrement=True)
    driver_id = Column(String(20), unique=True, nullable=False, index=True)

    total_rides = Column(Integer, default=0)
    delayed_rides = Column(Integer, default=0)
    delay_rate = Column(Float, default=0.0)
    avg_delay_minutes = Column(Float, default=0.0)
    avg_rating = Column(Float, default=0.0)
    reliability_score = Column(Float, default=0.0)
    risk_score = Column(Float, default=0.0)
    efficiency_score = Column(Float, default=0.0)
    last_ride_at = Column(DateTime(timezone=True))

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
