"""
Pydantic schemas for admin API requests/responses.
"""
from pydantic import BaseModel


class ConfigResponse(BaseModel):
    delay_threshold: float
    learning_rate: float
    epochs: int
    batch_size: int
    weight_decay: float
    train_size: float
    validation_size: float
    dropout: float
    hidden_layers: list[int]


class ConfigUpdateRequest(BaseModel):
    delay_threshold: float | None = None
    learning_rate: float | None = None
    epochs: int | None = None
    batch_size: int | None = None
    weight_decay: float | None = None
    dropout: float | None = None


class PipelineStatusResponse(BaseModel):
    total_raw_orders: int
    engineered_orders: int
    unprocessed_orders: int
    total_predictions: int
    total_training_runs: int
    total_drivers: int


class UploadResponse(BaseModel):
    message: str
    rows_inserted: int
    filename: str
