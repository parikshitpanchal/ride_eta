"""
Pydantic schemas for prediction API responses.
"""
from pydantic import BaseModel
from datetime import datetime


class PredictionResponse(BaseModel):
    id: int
    order_id: str
    booking_timestamp: str | None = None
    actual_total_time_minutes: float | None = None
    google_total_eta_minutes: float | None = None
    total_delay_minutes: float | None = None
    order_delayed: int | None = None
    predicted_eta_minutes: float | None = None
    predicted_delay_probability: float | None = None
    predicted_delay: int | None = None
    training_run_id: int | None = None
    created_at: datetime | None = None

    class Config:
        from_attributes = True


class PredictionStatsResponse(BaseModel):
    total_predictions: int
    total_delayed: int
    total_on_time: int
    avg_eta_error: float | None = None
    avg_delay_probability: float | None = None


class PaginatedPredictionsResponse(BaseModel):
    items: list[PredictionResponse]
    total: int
    page: int
    page_size: int
    total_pages: int
