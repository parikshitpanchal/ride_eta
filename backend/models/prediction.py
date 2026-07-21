"""
ORM model for the predictions table.
Stores model predictions on test data.
"""
from sqlalchemy import Column, Integer, String, Float, SmallInteger, DateTime, ForeignKey
from sqlalchemy.sql import func

from backend.database import Base


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    order_id = Column(String(20), index=True)
    booking_timestamp = Column(String(50))

    # Ground truth
    actual_total_time_minutes = Column(Float)
    google_total_eta_minutes = Column(Float)
    total_delay_minutes = Column(Float)
    order_delayed = Column(SmallInteger)

    # Model predictions
    predicted_eta_minutes = Column(Float)
    predicted_delay_probability = Column(Float)
    predicted_delay = Column(SmallInteger)

    # Link to which training run produced this prediction
    training_run_id = Column(Integer, ForeignKey("training_runs.id"), nullable=True, index=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
