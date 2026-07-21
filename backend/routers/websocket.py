"""
WebSocket endpoint for live training updates.
Frontend connects to /ws/training to receive per-epoch metrics in real-time.
Only sends updates when state changes to avoid unnecessary re-renders.
"""
import asyncio
import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from sqlalchemy import desc

from backend.database import SessionLocal
from backend.models.training_run import TrainingRun
from backend.models.training_epoch import TrainingEpoch

logger = logging.getLogger(__name__)

router = APIRouter()


@router.websocket("/ws/training")
async def training_websocket(websocket: WebSocket):
    """
    WebSocket that polls the DB for training updates and pushes them to the frontend.
    Sends updates ONLY when training state or epoch count changes.
    """
    await websocket.accept()
    last_sent_key = None

    try:
        while True:
            db = SessionLocal()
            try:
                # Find active training run
                active_run = db.query(TrainingRun).filter(TrainingRun.status == "running").first()

                if active_run:
                    epochs = (
                        db.query(TrainingEpoch)
                        .filter(TrainingEpoch.training_run_id == active_run.id)
                        .order_by(TrainingEpoch.epoch.asc())
                        .all()
                    )

                    current_count = len(epochs)
                    current_key = f"running_{active_run.id}_{current_count}_{active_run.completed_epochs}"

                    if current_key != last_sent_key:
                        epoch_data = [
                            {
                                "epoch": e.epoch,
                                "train_loss": e.train_loss,
                                "val_loss": e.val_loss,
                                "train_mae": e.train_mae,
                                "val_mae": e.val_mae,
                                "train_f1": e.train_f1,
                                "val_f1": e.val_f1,
                            }
                            for e in epochs
                        ]

                        message = {
                            "type": "training_update",
                            "run_id": active_run.id,
                            "status": active_run.status,
                            "total_epochs": active_run.total_epochs,
                            "completed_epochs": active_run.completed_epochs or 0,
                            "best_epoch": active_run.best_epoch,
                            "best_val_loss": active_run.best_val_loss,
                            "epochs": epoch_data,
                        }
                        await websocket.send_text(json.dumps(message))
                        last_sent_key = current_key
                else:
                    latest = (
                        db.query(TrainingRun)
                        .order_by(desc(TrainingRun.created_at))
                        .first()
                    )
                    status_str = latest.status if latest else "idle"
                    run_id = latest.id if latest else 0
                    current_key = f"idle_{run_id}_{status_str}"

                    if current_key != last_sent_key:
                        message = {
                            "type": "no_training",
                            "status": status_str,
                            "run_id": run_id,
                        }
                        await websocket.send_text(json.dumps(message))
                        last_sent_key = current_key

            finally:
                db.close()

            await asyncio.sleep(2)  # Check DB every 2 seconds silently

    except WebSocketDisconnect:
        logger.info("Training WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
