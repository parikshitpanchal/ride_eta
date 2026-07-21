"""
Backend configuration for the Ride ETA Platform.
"""
from pathlib import Path

# =====================================================
# PROJECT PATHS
# =====================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # Ride_ETA/
BACKEND_DIR = Path(__file__).resolve().parent  # Ride_ETA/backend/
ML_DIR = PROJECT_ROOT  # ML code is at the project root
DATA_DIR = PROJECT_ROOT / "data"
SAVED_MODEL_DIR = PROJECT_ROOT / "saved_models"
UPLOAD_DIR = DATA_DIR / "uploads"

# Create upload directory if it doesn't exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================
# DATABASE
# =====================================================
DATABASE_URL = "postgresql://postgres:1632@localhost:5432/ride_eta_db"

# =====================================================
# AUTHENTICATION & SECURITY
# =====================================================
SECRET_KEY = "ride-eta-secret-key-change-in-production-super-secure"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours

# =====================================================
# CORS (for Next.js frontend on port 3000)
# =====================================================
CORS_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

# =====================================================
# API
# =====================================================
API_PREFIX = "/api"
