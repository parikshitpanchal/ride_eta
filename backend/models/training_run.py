"""
ORM model for the training_runs table.
Stores metadata and final metrics for each training run.
"""
from sqlalchemy import Column, Integer, String, Float, DateTime, JSON
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from backend.database import Base


class TrainingRun(Base):
    __tablename__ = "training_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    status = Column(String(20), default="running")  # running / completed / failed / stopped

    # Training config
    total_epochs = Column(Integer)
    completed_epochs = Column(Integer, default=0)
    best_epoch = Column(Integer)
    best_val_loss = Column(Float)
    final_train_loss = Column(Float)
    final_val_loss = Column(Float)
    threshold = Column(Float)
    learning_rate = Column(Float)
    batch_size = Column(Integer)
    train_samples = Column(Integer)
    val_samples = Column(Integer)

    # Test metrics (computed after training on prediction data)
    test_mae = Column(Float)
    test_rmse = Column(Float)
    test_r2 = Column(Float)
    test_accuracy = Column(Float)
    test_precision = Column(Float)
    test_recall = Column(Float)
    test_f1 = Column(Float)
    test_roc_auc = Column(Float)
    confusion_matrix = Column(JSON)

    # Timestamps
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationship to epochs
    epochs = relationship("TrainingEpoch", back_populates="training_run", order_by="TrainingEpoch.epoch")
