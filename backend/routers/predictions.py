"""
Predictions API router.
Handles listing, filtering, and retrieving prediction results.
"""
import math

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import func, case

from backend.database import get_db
from backend.models.prediction import Prediction
from backend.schemas.prediction import (
    PredictionResponse,
    PredictionStatsResponse,
    PaginatedPredictionsResponse,
)

router = APIRouter(prefix="/api/predictions", tags=["Predictions"])


@router.get("/stats", response_model=PredictionStatsResponse)
def get_prediction_stats(db: Session = Depends(get_db)):
    """Get summary statistics for all predictions."""
    stats = db.query(
        func.count(Prediction.id).label("total"),
        func.sum(case((Prediction.predicted_delay == 1, 1), else_=0)).label("delayed"),
        func.sum(case((Prediction.predicted_delay == 0, 1), else_=0)).label("on_time"),
        func.avg(
            func.abs(Prediction.predicted_eta_minutes - Prediction.actual_total_time_minutes)
        ).label("avg_eta_error"),
        func.avg(Prediction.predicted_delay_probability).label("avg_delay_prob"),
    ).first()

    return PredictionStatsResponse(
        total_predictions=stats.total or 0,
        total_delayed=stats.delayed or 0,
        total_on_time=stats.on_time or 0,
        avg_eta_error=round(stats.avg_eta_error, 2) if stats.avg_eta_error else None,
        avg_delay_probability=round(stats.avg_delay_prob, 4) if stats.avg_delay_prob else None,
    )


@router.get("/{prediction_id}", response_model=PredictionResponse)
def get_prediction(prediction_id: int, db: Session = Depends(get_db)):
    """Get a single prediction by ID."""
    prediction = db.query(Prediction).filter(Prediction.id == prediction_id).first()
    if not prediction:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Prediction not found")
    return prediction


@router.get("", response_model=PaginatedPredictionsResponse)
def list_predictions(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    delay_status: str | None = Query(None, description="Filter: 'delayed', 'on_time', or None for all"),
    search: str | None = Query(None, description="Search by order_id"),
    sort_by: str = Query("id", description="Column to sort by"),
    sort_order: str = Query("desc", description="'asc' or 'desc'"),
    training_run_id: int | None = Query(None, description="Filter by training run"),
    db: Session = Depends(get_db),
):
    """List predictions with pagination, filtering, and sorting."""
    query = db.query(Prediction)

    # Filters
    if delay_status == "delayed":
        query = query.filter(Prediction.predicted_delay == 1)
    elif delay_status == "on_time":
        query = query.filter(Prediction.predicted_delay == 0)

    if search:
        query = query.filter(Prediction.order_id.ilike(f"%{search}%"))

    if training_run_id is not None:
        query = query.filter(Prediction.training_run_id == training_run_id)

    # Count total
    total = query.count()

    # Sorting
    sort_column = getattr(Prediction, sort_by, Prediction.id)
    if sort_order == "desc":
        query = query.order_by(sort_column.desc())
    else:
        query = query.order_by(sort_column.asc())

    # Pagination
    offset = (page - 1) * page_size
    items = query.offset(offset).limit(page_size).all()

    return PaginatedPredictionsResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=math.ceil(total / page_size) if total > 0 else 0,
    )
