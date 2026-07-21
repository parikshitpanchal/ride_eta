"""
Ride ETA Platform — FastAPI Application
Main entry point for the backend server.
"""
import logging
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import CORS_ORIGINS, ML_DIR
from backend.database import engine, Base

# Import all models so SQLAlchemy knows about them
from backend.models.raw_ride_order import RawRideOrder
from backend.models.engineered_ride_order import EngineeredRideOrder
from backend.models.prediction import Prediction
from backend.models.training_run import TrainingRun
from backend.models.training_epoch import TrainingEpoch
from backend.models.driver import Driver
from backend.models.user import User

# Import routers
from backend.routers import predictions, model_metrics, training, drivers, admin, auth, websocket

# Add ML code to path for services
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# Create tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Ride ETA Platform",
    description="Production-grade ETA prediction and delay classification system",
    version="1.0.0",
)

# CORS middleware for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(predictions.router)
app.include_router(model_metrics.router)
app.include_router(training.router)
app.include_router(drivers.router)
app.include_router(admin.router)
app.include_router(auth.router)
app.include_router(websocket.router)


@app.get("/api/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "ride-eta-platform"}


@app.post("/api/admin/refresh-drivers")
def refresh_drivers_endpoint():
    """Trigger driver stats refresh from raw order data."""
    from backend.database import SessionLocal
    from backend.services.driver_service import refresh_driver_stats
    db = SessionLocal()
    try:
        result = refresh_driver_stats(db)
        return result
    finally:
        db.close()
