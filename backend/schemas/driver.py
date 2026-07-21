"""
Pydantic schemas for driver analytics API responses.
"""
from pydantic import BaseModel
from datetime import datetime


class DriverResponse(BaseModel):
    id: int
    driver_id: str
    total_rides: int
    delayed_rides: int
    delay_rate: float
    avg_delay_minutes: float
    avg_rating: float
    reliability_score: float
    risk_score: float
    efficiency_score: float
    last_ride_at: datetime | None = None
    created_at: datetime | None = None

    class Config:
        from_attributes = True


class PaginatedDriversResponse(BaseModel):
    items: list[DriverResponse]
    total: int
    page: int
    page_size: int
    total_pages: int
