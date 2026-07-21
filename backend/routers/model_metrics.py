"""
Model metrics API router.
Handles fetching training run metrics and comparisons.
"""
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models.training_run import TrainingRun
from backend.schemas.training import TrainingRunResponse, TrainingRunDetailResponse

router = APIRouter(prefix="/api/model", tags=["Model Metrics"])


@router.get("/metrics", response_model=TrainingRunResponse | None)
def get_latest_metrics(db: Session = Depends(get_db)):
    """Get metrics from the latest completed training run."""
    run = (
        db.query(TrainingRun)
        .filter(TrainingRun.status == "completed")
        .order_by(TrainingRun.completed_at.desc())
        .first()
    )
    return run


@router.get("/history", response_model=list[TrainingRunResponse])
def get_training_history(
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """Get all training run history."""
    runs = (
        db.query(TrainingRun)
        .order_by(TrainingRun.created_at.desc())
        .limit(limit)
        .all()
    )
    return runs


@router.get("/compare", response_model=list[TrainingRunResponse])
def compare_runs(
    run_ids: str = Query(..., description="Comma-separated run IDs to compare"),
    db: Session = Depends(get_db),
):
    """Compare metrics across multiple training runs."""
    ids = [int(x.strip()) for x in run_ids.split(",") if x.strip().isdigit()]
    runs = db.query(TrainingRun).filter(TrainingRun.id.in_(ids)).all()
    return runs
