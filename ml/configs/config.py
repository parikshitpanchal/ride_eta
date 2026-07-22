from pathlib import Path
# pyrefly: ignore [missing-import]
import torch
import logging

# =====================================================
# PROJECT PATHS
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"

PROCESSED_DATA_DIR = DATA_DIR / "processed"

ARTIFACTS_DIR = DATA_DIR / "artifacts"

SAVED_MODEL_DIR = PROJECT_ROOT / "saved_models"

LOG_DIR = PROJECT_ROOT / "logs"

TRAIN_DATA_FILE = DATA_DIR / "ride_orders_engineered_D30K.csv"
TEST_DATA_FILE = DATA_DIR / "ride_orders_test_v24.csv"

# Create directories if they don't exist
for directory in [DATA_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Quick Demo Mode (set to True for faster execution with sampling)
QUICK_DEMO_MODE = False  # Set to True for 10k sample
QUICK_DEMO_SAMPLE_SIZE = 30000  # Sample size for quick demo

# =====================================================
# DATASET CONFIGURATION
# =====================================================

TRAIN_SIZE = 0.85
VALIDATION_SIZE = 0.15
TEST_SIZE = 0

# ETA_TARGET = "eta_minutes"
# DELAY_TARGET = "is_delayed"

ORDER_ID_COLUMN = "order_id"

TIMESTAMP_COLUMN = "booking_timestamp"

# =====================================================
# METADATA COLUMNS (NOT USED FOR TRAINING)
# =====================================================

METADATA_COLUMNS = [
    "order_id",
    "booking_timestamp",
]


# =====================================================
# TARGET COLUMNS
# =====================================================

ETA_TARGET = "actual_total_time_minutes"
DELAY_TARGET = "order_delayed"


# =====================================================
# INTERNAL COLUMNS
# Used during feature engineering only.
# Never passed to the neural network.
# =====================================================

INTERNAL_COLUMNS = [
    "actual_pickup_time_minutes",
    "actual_trip_time_minutes",
    "pickup_delay_minutes",
    "trip_delay_minutes",
    "total_delay_minutes",
    "individual_trip_ratio",
    "delay_severity",
]

# =====================================================
# NUMERICAL FEATURES
# =====================================================

CONTINUOUS_COLUMNS = [# Location Coordinates
    "pickup_latitude",
    "pickup_longitude",
    "drop_latitude",
    "drop_longitude",
    
    # DistanceMetrics
    "driver_to_pickup_distance_km",
    "trip_distance_km",
    
    # Driver Metrics
    "driver_base_rating",
    "driver_experience_days",
    "driver_acceptance_rate",
    "driver_cancellation_rate",
    
    # Google ETA
    "google_pickup_eta_minutes",
    "google_trip_eta_minutes",
    
    # Historical Features & Delays
    "average_delay",
    "driver_avg_pickup_delay",
    "driver_avg_trip_delay",
    "driver_last_7_day_delay_rate",
    "driver_last_30_day_delay_rate",
    "driver_route_avg_delay",
    "driver_zone_avg_pickup_delay",
    "driver_zone_avg_trip_delay",
    "historical_route_ratio_avg",
    "driver_reliability",
    "driver_risk_score",
    "driver_efficiency_score",
    "driver_recent_delay_rate",
    "route_frequency",
    "historical_route_delay_rate",
    "route_reliability_score",]
TIME_COLUMNS = ["hour",
    "day_of_week",
    "week_of_year",
    "month",]
BINARY_COLUMNS = ["is_weekend",
    "festival_flag",
    "holiday_flag",
    "is_peak_hour",]

NUMERICAL_COLUMNS = (CONTINUOUS_COLUMNS+ TIME_COLUMNS+ BINARY_COLUMNS)


CATEGORICAL_COLUMNS = [

    "driver_id",

    "pickup_zone",
    "drop_zone",

    "pickup_h3_cell",
    "drop_h3_cell",

    "time_of_day",
]

# =====================================================
# MODEL INPUT FEATURES
# =====================================================


FEATURE_COLUMNS = NUMERICAL_COLUMNS + CATEGORICAL_COLUMNS

SCALING_COLUMNS = (CONTINUOUS_COLUMNS + TIME_COLUMNS)

# =====================================================
# MODEL CONFIGURATION
# =====================================================

HIDDEN_LAYERS = [256, 128, 64]

NUMERICAL_PROJECTION_SIZE = 64

DROPOUT = 0.30

ETA_OUTPUT_SIZE = 1

DELAY_OUTPUT_SIZE = 1

# =====================================================
# Data Generation Settings
# =====================================================

RANDOM_SEED = 42
N_ROWS = 30000  # 300k rows for robust training
START_DATE = "2024-01-01"
END_DATE = "2024-12-31"

# =====================================================
# Driver Pool Settings
# =====================================================
N_DRIVERS = 5000
DRIVER_RATING_MEAN = 4.2
DRIVER_RATING_STD = 0.6

# =====================================================
# Geography Settings (Ahmedabad, Gujarat, India - Real locations)
# =====================================================
ZONES = [
    "Prahlad_Nagar", "Satellite", "SG_Highway", "Bodakdev", "Vastrapur",
    "Thaltej", "Navrangpura", "CG_Road", "Law_Garden", "Maninagar",
    "Bopal", "South_Bopal", "Gota", "Chandkheda", "Motera",
    "Sabarmati", "Nikol", "Naroda", "Iskcon", "Science_City",
    "Airport_Area", "Gandhinagar_Corridor", "Gift_City_Corridor"
]

# =====================================================
# Ahmedabad geographic boundaries (lat, lon)
# =====================================================

AHMEDABAD_LAT_MIN = 22.95
AHMEDABAD_LAT_MAX = 23.15
AHMEDABAD_LON_MIN = 72.45
AHMEDABAD_LON_MAX = 72.75

# =====================================================
# Zone coordinates (approximate real locations in Ahmedabad)
# =====================================================

ZONE_COORDINATES = {
    "Prahlad_Nagar": (23.032, 72.527),
    "Satellite": (23.026, 72.506),
    "SG_Highway": (23.070, 72.545),
    "Bodakdev": (23.038, 72.518),
    "Vastrapur": (23.039, 72.527),
    "Thaltej": (23.050, 72.517),
    "Navrangpura": (23.037, 72.563),
    "CG_Road": (23.031, 72.561),
    "Law_Garden": (23.026, 72.566),
    "Maninagar": (22.998, 72.598),
    "Bopal": (23.040, 72.462),
    "South_Bopal": (23.025, 72.455),
    "Gota": (23.088, 72.516),
    "Chandkheda": (23.126, 72.597),
    "Motera": (23.083, 72.592),
    "Sabarmati": (23.058, 72.591),
    "Nikol": (23.055, 72.641),
    "Naroda": (23.064, 72.664),
    "Iskcon": (23.022, 72.508),
    "Science_City": (23.056, 72.520),
    "Airport_Area": (23.077, 72.630),
    "Gandhinagar_Corridor": (23.130, 72.645),
    "Gift_City_Corridor": (23.145, 72.678)
}
# =====================================================
# High congestion zones in Ahmedabad
# =====================================================

CONGESTION_ZONES = [
    "SG_Highway", "Iskcon", "Shivranjani", "CG_Road", 
    "Airport_Area", "Navrangpura"
]

# =====================================================
# Ahmedabad festival dates (2024)
# =====================================================

AHMEDABAD_FESTIVALS = [
    # Navratri (October 2024)
    "2024-10-03", "2024-10-04", "2024-10-05", "2024-10-06", "2024-10-07",
    "2024-10-08", "2024-10-09", "2024-10-10", "2024-10-11", "2024-10-12",
    # Diwali (November 2024)
    "2024-11-01", "2024-11-02", "2024-11-03",
    # Uttarayan (January)
    "2024-01-14", "2024-01-15",
    # IPL Season (April-May)
    "2024-04-15", "2024-04-20", "2024-04-25", "2024-05-05", "2024-05-10"
]

# =====================================================
# Peak hours (Ahmedabad-specific)
# =====================================================
MORNING_PEAK_START = 8
MORNING_PEAK_END = 11
EVENING_PEAK_START = 17  # 5 PM
EVENING_PEAK_END = 21    # 9 PM


# =====================================================
# TRAINING CONFIGURATION
# =====================================================

BATCH_SIZE = 256

EPOCHS = 100

LEARNING_RATE = 0.001

WEIGHT_DECAY = 1e-5

OPTIMIZER = "adam"

SHUFFLE = True

# =====================================================
# ARTIFACTS
# =====================================================

ENCODER_FILE = "categorical_encoder.pkl"

SCALER_FILE = "numerical_scaler.pkl"

MODEL_METADATA_FILE = "ride_eta_metadata.json"

# =====================================================
# DEVICE CONFIGURATION
# =====================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RANDOM_SEED = 42

# =====================================================
# LOSS CONFIGURATION
# =====================================================

ETA_LOSS_WEIGHT = 1.0

DELAY_LOSS_WEIGHT = 1.0

DELAY_THRESHOLD = 0.3

# =====================================================
# CHECKPOINT CONFIGURATION
# =====================================================

SAVE_BEST_MODEL = True

CHECKPOINT_NAME = "ride_eta_model.pth"

EARLY_STOPPING_PATIENCE = 10


# =====================================================
# LOGGING CONFIGURATION
# =====================================================


SAVE_TRAINING_HISTORY = True


# =====================================================
# DATA VALIDATION
# =====================================================

REQUIRED_COLUMNS = (
    FEATURE_COLUMNS
    + [ETA_TARGET]
    + [DELAY_TARGET]
    + METADATA_COLUMNS
)

# =====================================================
# DATALOADER CONFIGURATION
# =====================================================

NUM_WORKERS = 0

PIN_MEMORY = torch.cuda.is_available()

DROP_LAST = False

LOG_LEVEL = logging.INFO

LOG_FORMAT = (
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

MAX_EMBEDDING_DIM = 50