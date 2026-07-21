"""
Drivers API router.
Handles driver analytics: listing, filtering, top/worst performers.
"""
import math

from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func as sql_func

from backend.database import get_db
from backend.models.driver import Driver
from backend.models.raw_ride_order import RawRideOrder
from backend.schemas.driver import DriverResponse, PaginatedDriversResponse

router = APIRouter(prefix="/api/drivers", tags=["Drivers"])


@router.get("/top-performers", response_model=list[DriverResponse])
def get_top_performers(
    limit: int = Query(10, ge=1, le=50),
    db: Session = Depends(get_db),
):
    """Get top N most reliable drivers (lowest delay rate, min 5 rides)."""
    drivers = (
        db.query(Driver)
        .filter(Driver.total_rides >= 5)
        .order_by(Driver.delay_rate.asc(), Driver.avg_rating.desc())
        .limit(limit)
        .all()
    )
    return drivers


@router.get("/worst-performers", response_model=list[DriverResponse])
def get_worst_performers(
    limit: int = Query(10, ge=1, le=50),
    db: Session = Depends(get_db),
):
    """Get top N highest delay-rate drivers (min 5 rides)."""
    drivers = (
        db.query(Driver)
        .filter(Driver.total_rides >= 5)
        .order_by(Driver.delay_rate.desc())
        .limit(limit)
        .all()
    )
    return drivers


@router.get("/{driver_id}", response_model=DriverResponse)
def get_driver(driver_id: str, db: Session = Depends(get_db)):
    """Get single driver details."""
    driver = db.query(Driver).filter(Driver.driver_id == driver_id).first()
    if not driver:
        raise HTTPException(status_code=404, detail="Driver not found")
    return driver


@router.get("", response_model=PaginatedDriversResponse)
def list_drivers(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    search: str | None = Query(None, description="Search by driver_id"),
    sort_by: str = Query("delay_rate", description="Column to sort by"),
    sort_order: str = Query("desc", description="'asc' or 'desc'"),
    db: Session = Depends(get_db),
):
    """List all drivers with pagination and sorting."""
    query = db.query(Driver)

    if search:
        query = query.filter(Driver.driver_id.ilike(f"%{search}%"))

    total = query.count()

    sort_column = getattr(Driver, sort_by, Driver.delay_rate)
    if sort_order == "desc":
        query = query.order_by(sort_column.desc())
    else:
        query = query.order_by(sort_column.asc())

    offset = (page - 1) * page_size
    items = query.offset(offset).limit(page_size).all()

    return PaginatedDriversResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=math.ceil(total / page_size) if total > 0 else 0,
    )
