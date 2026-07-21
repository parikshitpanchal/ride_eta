"""
Pydantic schemas for training API responses.
"""
from pydantic import BaseModel
from datetime import datetime


class TrainingEpochResponse(BaseModel):
    id: int
    epoch: int
    train_loss: float | None = None
    val_loss: float | None = None
    train_mae: float | None = None
    val_mae: float | None = None
    train_f1: float | None = None
    val_f1: float | None = None

    class Config:
        from_attributes = True


class TrainingRunResponse(BaseModel):
    id: int
    status: str
    total_epochs: int | None = None
    completed_epochs: int | None = None
    best_epoch: int | None = None
    best_val_loss: float | None = None
    final_train_loss: float | None = None
    final_val_loss: float | None = None
    threshold: float | None = None
    learning_rate: float | None = None
    batch_size: int | None = None
    train_samples: int | None = None
    val_samples: int | None = None
    test_mae: float | None = None
    test_rmse: float | None = None
    test_r2: float | None = None
    test_accuracy: float | None = None
    test_precision: float | None = None
    test_recall: float | None = None
    test_f1: float | None = None
    test_roc_auc: float | None = None
    confusion_matrix: list | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    created_at: datetime | None = None

    class Config:
        from_attributes = True


class TrainingRunDetailResponse(TrainingRunResponse):
    epochs: list[TrainingEpochResponse] = []


class TrainingStatusResponse(BaseModel):
    is_training: bool
    current_run: TrainingRunResponse | None = None
