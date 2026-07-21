"""
Training API router.
Handles starting/stopping training, status, and epoch data.
"""
from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models.training_run import TrainingRun
from backend.models.training_epoch import TrainingEpoch
from backend.schemas.training import (
    TrainingRunResponse,
    TrainingRunDetailResponse,
    TrainingEpochResponse,
    TrainingStatusResponse,
)
from backend.services.auth_service import require_role

router = APIRouter(prefix="/api/training", tags=["Training"])

# Global reference to track if training is running
_training_task = None


@router.post("/start", response_model=TrainingRunResponse, dependencies=[Depends(require_role("admin"))])
def start_training(db: Session = Depends(get_db)):
    """Start a new training run in the background."""
    # Check if already training
    active = db.query(TrainingRun).filter(TrainingRun.status == "running").first()
    if active:
        raise HTTPException(status_code=409, detail="Training is already running")

    # Import here to avoid circular imports
    from backend.services.training_service import start_training_background
    run = start_training_background(db)
    return run


@router.get("/status", response_model=TrainingStatusResponse)
def get_training_status(db: Session = Depends(get_db)):
    """Get current training status."""
    active = db.query(TrainingRun).filter(TrainingRun.status == "running").first()
    return TrainingStatusResponse(
        is_training=active is not None,
        current_run=active,
    )


@router.post("/stop", dependencies=[Depends(require_role("admin"))])
def stop_training(db: Session = Depends(get_db)):
    """Stop the current training run."""
    active = db.query(TrainingRun).filter(TrainingRun.status == "running").first()
    if not active:
        raise HTTPException(status_code=404, detail="No training is currently running")

    from backend.services.training_service import stop_training_flag
    stop_training_flag.set()

    return {"message": "Stop signal sent to training process"}


@router.get("/runs", response_model=list[TrainingRunResponse])
def list_training_runs(
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """List all training runs."""
    runs = (
        db.query(TrainingRun)
        .order_by(TrainingRun.created_at.desc())
        .limit(limit)
        .all()
    )
    return runs


@router.get("/runs/{run_id}", response_model=TrainingRunDetailResponse)
def get_training_run(run_id: int, db: Session = Depends(get_db)):
    """Get a specific training run with its epoch data."""
    run = db.query(TrainingRun).filter(TrainingRun.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Training run not found")
    return run


@router.get("/runs/{run_id}/epochs", response_model=list[TrainingEpochResponse])
def get_training_epochs(run_id: int, db: Session = Depends(get_db)):
    """Get epoch-by-epoch data for a training run."""
    epochs = (
        db.query(TrainingEpoch)
        .filter(TrainingEpoch.training_run_id == run_id)
        .order_by(TrainingEpoch.epoch.asc())
        .all()
    )
    return epochs
