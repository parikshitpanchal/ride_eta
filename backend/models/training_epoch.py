"""
ORM model for the training_epochs table.
Stores per-epoch metrics for each training run.
"""
from sqlalchemy import Column, Integer, Float, DateTime, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from backend.database import Base


class TrainingEpoch(Base):
    __tablename__ = "training_epochs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    training_run_id = Column(Integer, ForeignKey("training_runs.id"), nullable=False, index=True)
    epoch = Column(Integer, nullable=False)

    train_loss = Column(Float)
    val_loss = Column(Float)
    train_mae = Column(Float)
    val_mae = Column(Float)
    train_f1 = Column(Float)
    val_f1 = Column(Float)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationship back to training run
    training_run = relationship("TrainingRun", back_populates="epochs")
