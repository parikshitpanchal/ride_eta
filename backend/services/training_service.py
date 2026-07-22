"""
Training service — wraps the existing ML training pipeline.
Runs training in a background thread, logs epoch data to DB, and updates training_runs.
"""
import sys
import threading
import logging
from datetime import datetime, timezone

import numpy as np
import torch
from torch.optim import Adam
from sqlalchemy.orm import Session

from backend.config import ML_DIR
from backend.database import SessionLocal
from backend.models.training_run import TrainingRun
from backend.models.training_epoch import TrainingEpoch
from ml.evaluate import evaluate_model

# Add ML code to path
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

logger = logging.getLogger(__name__)

# Global flag to stop training
stop_training_flag = threading.Event()


def start_training_background(db: Session) -> TrainingRun:
    """Create a training run record and start training in a background thread."""
    from ml.configs import config as ml_config

    run = TrainingRun(
        status="running",
        total_epochs=ml_config.EPOCHS,
        completed_epochs=0,
        threshold=ml_config.DELAY_THRESHOLD,
        learning_rate=ml_config.LEARNING_RATE,
        batch_size=ml_config.BATCH_SIZE,
        started_at=datetime.now(timezone.utc),
    )
    db.add(run)
    db.commit()
    db.refresh(run)

    run_id = run.id
    stop_training_flag.clear()

    thread = threading.Thread(target=_run_training, args=(run_id,), daemon=True)
    thread.start()

    return run


def _run_training(run_id: int):
    """The actual training loop, runs in a background thread."""
    db = SessionLocal()
    try:
        from ml.configs import config as ml_config
        from ml.preprocessing.encoder import CategoricalEncoder
        from ml.preprocessing.scaler import NumericalScaler
        from ml.preprocessing.validator import DataValidator
        from ml.preprocessing.pipeline import DataPipeline
        from ml.datasets.ride_dataset import RideDataset
        from ml.datasets.dataloader import create_dataloader
        from ml.models.ride_eta_network import RideETANetwork
        from ml.losses.losses import MultiTaskLoss
        from ml.utils.model_metadata import save_model_metadata
        from ml.metrics.classification_metrics import calculate_classification_metrics
        from ml.metrics.regression_metrics import calculate_regression_metrics
        from ml.callbacks.checkpoint import ModelCheckpoint
        from ml.callbacks.early_stopping import EarlyStopping
        import pandas as pd

        # Set seeds
        torch.manual_seed(ml_config.RANDOM_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(ml_config.RANDOM_SEED)
            torch.cuda.manual_seed_all(ml_config.RANDOM_SEED)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # Load and preprocess training data from DB
        from backend.services.data_service import load_training_data_from_db
        train_df = load_training_data_from_db(db)

        if train_df is None or len(train_df) == 0:
            logger.error("Training aborted. No training data found in database.")
            run = db.query(TrainingRun).filter(TrainingRun.id == run_id).first()

            run.status = "failed"
            run.completed_at = datetime.now(timezone.utc)

            db.commit()
            # logger.warning("No training data found in DB, falling back to CSV")
            # train_df = pd.read_csv(ml_config.TRAIN_DATA_FILE)
            return
        validator = DataValidator()
        encoder = CategoricalEncoder()
        scaler = NumericalScaler()
        pipeline = DataPipeline(validator=validator, encoder=encoder, scaler=scaler)
        train_df = pipeline.fit_transform(train_df)

        # Split 85/15
        train_dataframe, validation_dataframe = pipeline.split_dataset(train_df)

        run = db.query(TrainingRun).filter(TrainingRun.id == run_id).first()
        run.train_samples = len(train_dataframe)
        run.val_samples = len(validation_dataframe)
        db.commit()

        train_dataset = RideDataset(dataframe=train_dataframe)
        val_dataset = RideDataset(dataframe=validation_dataframe)
        train_loader = create_dataloader(dataset=train_dataset, shuffle=ml_config.SHUFFLE)
        val_loader = create_dataloader(dataset=val_dataset, shuffle=False)

        # Build model
        model = RideETANetwork(
            numerical_feature_count=len(ml_config.NUMERICAL_COLUMNS),
            categorical_cardinalities=encoder.get_cardinalities(),
        )
        model.to(ml_config.DEVICE)

        optimizer = Adam(
            params=model.parameters(),
            lr=ml_config.LEARNING_RATE,
            weight_decay=ml_config.WEIGHT_DECAY,
        )
        criterion = MultiTaskLoss()
        checkpoint = ModelCheckpoint(
            save_directory=ml_config.SAVED_MODEL_DIR,
            checkpoint_name=ml_config.CHECKPOINT_NAME,
        )
        early_stopping = EarlyStopping(patience=ml_config.EARLY_STOPPING_PATIENCE)

        best_val_loss = float("inf")
        best_epoch = 0

        for epoch in range(ml_config.EPOCHS):
            if stop_training_flag.is_set():
                logger.info(f"Training stopped by user at epoch {epoch + 1}")
                run = db.query(TrainingRun).filter(TrainingRun.id == run_id).first()
                run.status = "stopped"
                run.completed_at = datetime.now(timezone.utc)
                db.commit()
                break

            # Train epoch
            train_loss, train_eta_metrics, train_delay_metrics = _run_epoch(
                model, train_loader, criterion, ml_config.DEVICE, optimizer=optimizer, training=True
            )

            # Validate epoch
            val_loss, val_eta_metrics, val_delay_metrics = _run_epoch(
                model, val_loader, criterion, ml_config.DEVICE, optimizer=None, training=False
            )

            # Save checkpoint
            checkpoint(model=model, validation_loss=val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1

            # Log epoch to DB
            epoch_record = TrainingEpoch(
                training_run_id=run_id,
                epoch=epoch + 1,
                train_loss=train_loss,
                val_loss=val_loss,
                train_mae=train_eta_metrics["mae"],
                val_mae=val_eta_metrics["mae"],
                train_f1=train_delay_metrics["f1"],
                val_f1=val_delay_metrics["f1"],
            )
            db.add(epoch_record)

            # Update run progress
            run = db.query(TrainingRun).filter(TrainingRun.id == run_id).first()
            run.completed_epochs = epoch + 1
            run.best_epoch = best_epoch
            run.best_val_loss = best_val_loss
            run.final_train_loss = train_loss
            run.final_val_loss = val_loss
            db.commit()

            logger.info(
                f"Epoch {epoch+1}/{ml_config.EPOCHS} | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Train MAE: {train_eta_metrics['mae']:.4f} | Val MAE: {val_eta_metrics['mae']:.4f} | "
                f"Train F1: {train_delay_metrics['f1']:.4f} | Val F1: {val_delay_metrics['f1']:.4f}"
            )

            # Early stopping
            if early_stopping(validation_loss=val_loss):
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

        # Save artifacts
        save_model_metadata(
            file_path=ml_config.SAVED_MODEL_DIR / ml_config.MODEL_METADATA_FILE,
            numerical_feature_count=len(ml_config.NUMERICAL_COLUMNS),
            categorical_cardinalities=encoder.get_cardinalities(),
        )
        encoder.save()
        scaler.save()
# --------------------------------------------------------------------------------
        # test_loss, eta_metrics, delay_metrics = evaluate(...)
        validation_loss, eta_metrics, delay_metrics = evaluate_model(
            model=model,
            dataframe=validation_dataframe,
            validator=validator,
            encoder=encoder,
            scaler=scaler,
                )


        run = db.query(TrainingRun).filter(TrainingRun.id == run_id).first()

        run.test_mae = eta_metrics["mae"]
        run.test_rmse = eta_metrics["rmse"]
        run.test_r2 = eta_metrics["r2"]

        run.test_accuracy = delay_metrics["accuracy"]
        run.test_precision = delay_metrics["precision"]
        run.test_recall = delay_metrics["recall"]
        run.test_f1 = delay_metrics["f1"]
        run.test_roc_auc = delay_metrics["roc_auc"]
        run.confusion_matrix = delay_metrics["confusion_matrix"]

        db.commit()

        # Mark completed
        run = db.query(TrainingRun).filter(TrainingRun.id == run_id).first()
        if run.status == "running":
            run.status = "completed"
        run.completed_at = datetime.now(timezone.utc)
        db.commit()

        logger.info(f"Training run {run_id} completed successfully")

    except Exception as e:
        logger.error(f"Training run {run_id} failed: {e}", exc_info=True)
        run = db.query(TrainingRun).filter(TrainingRun.id == run_id).first()
        if run:
            run.status = "failed"
            run.completed_at = datetime.now(timezone.utc)
            db.commit()
    finally:
        db.close()


def _run_epoch(model, dataloader, criterion, device, optimizer=None, training=True):
    """Run a single training or validation epoch. Returns (loss, eta_metrics, delay_metrics)."""
    from ml.metrics.classification_metrics import calculate_classification_metrics
    from ml.metrics.regression_metrics import calculate_regression_metrics

    if training:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    eta_preds, eta_tgts = [], []
    delay_probs, delay_tgts = [], []

    for batch in dataloader:
        numerical = batch["numerical_features"].to(device)
        categorical = batch["categorical_features"].to(device)
        eta_target = batch["eta_target"].to(device)
        delay_target = batch["delay_target"].to(device)
        targets = {"eta_target": eta_target, "delay_target": delay_target}

        if training and optimizer:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            outputs = model(numerical, categorical)
            loss = criterion(predictions=outputs, targets=targets)
            if training and optimizer:
                loss.backward()
                optimizer.step()

        running_loss += loss.item()
        eta_preds.append(outputs["eta"].detach().cpu().numpy().ravel())
        eta_tgts.append(eta_target.detach().cpu().numpy().ravel())
        delay_probs.append(torch.sigmoid(outputs["delay"]).detach().cpu().numpy().ravel())
        delay_tgts.append(delay_target.detach().cpu().numpy().ravel())

    eta_preds = np.concatenate(eta_preds)
    eta_tgts = np.concatenate(eta_tgts)
    delay_probs = np.concatenate(delay_probs)
    delay_tgts = np.concatenate(delay_tgts)

    eta_metrics = calculate_regression_metrics(predictions=eta_preds, targets=eta_tgts)
    delay_metrics = calculate_classification_metrics(probabilities=delay_probs, targets=delay_tgts)

    epoch_loss = running_loss / len(dataloader)
    return epoch_loss, eta_metrics, delay_metrics
