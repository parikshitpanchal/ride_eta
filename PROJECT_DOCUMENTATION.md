# 🚖 Ride ETA — Full Project Documentation

> **A production-grade ride-hailing ETA prediction and delay classification system built with PyTorch, FastAPI, and Next.js**

---

## 📋 Table of Contents

1. [What Is This Project?](#1-what-is-this-project)
2. [Big Picture Architecture](#2-big-picture-architecture)
3. [Project Folder Structure](#3-project-folder-structure)
4. [Data Flow — From Start to Prediction](#4-data-flow--from-start-to-prediction)
5. [ML Layer — Every File Explained](#5-ml-layer--every-file-explained)
6. [Backend Layer — Every File Explained](#6-backend-layer--every-file-explained)
7. [Frontend Layer — Every File Explained](#7-frontend-layer--every-file-explained)
8. [Database Tables Explained](#8-database-tables-explained)
9. [API Endpoints Reference](#9-api-endpoints-reference)
10. [How to Run the Project](#10-how-to-run-the-project)
11. [Configuration Reference](#11-configuration-reference)

---

## 1. What Is This Project?

**Ride ETA** is an AI-powered system that predicts two things for every ride:

| Prediction | What it answers | Output |
|---|---|---|
| **ETA (Estimated Time of Arrival)** | How many minutes will this ride take? | A number (e.g., `18.5 minutes`) |
| **Delay Classification** | Will this ride be delayed? | Yes / No + a probability (e.g., `72% chance of delay`) |

Think of it like Uber or Ola's internal prediction system, built from scratch.

### Why is this impressive?

- Uses a **real neural network** (PyTorch) trained on synthetic Ahmedabad, India ride data
- Has a **full web dashboard** to manage training, view predictions, and analyze drivers
- Training happens **live in the browser** — watch each epoch update in real-time via WebSocket
- **No data leakage** — historical features are computed using only past rides (time-aware)

---

## 2. Big Picture Architecture

```
+-------------------------------------------------------------+
|                     USER (Browser)                          |
|                   Next.js Frontend                          |
|         (Dashboard, Training, Predictions, Drivers)         |
+------------------------+------------------------------------+
                         | HTTP REST + WebSocket
                         v
+-------------------------------------------------------------+
|                 FastAPI Backend (Port 8000)                  |
|   Routers: auth, predictions, training, drivers, admin      |
|   Services: training_service, prediction_service, etc.      |
|   Database: SQLite via SQLAlchemy ORM                        |
+------------------------+------------------------------------+
                         | Python imports
                         v
+-------------------------------------------------------------+
|                   ML Core (Python)                           |
|   configs -> preprocessing -> datasets -> models            |
|   trainer -> predictor -> callbacks -> metrics -> losses     |
+------------------------+------------------------------------+
                         | reads/writes
                         v
+-------------------------------------------------------------+
|                  Files on Disk                               |
|  data/  (CSV training files)                                 |
|  saved_models/  (model weights + preprocessing artifacts)    |
+-------------------------------------------------------------+
```

---

## 3. Project Folder Structure

```
Ride_ETA/
|
+-- configs/              # All project settings (one file to rule them all)
+-- data/                 # CSV datasets (training + test)
+-- datasets/             # PyTorch Dataset and DataLoader classes
+-- models/               # Neural network architecture (PyTorch)
+-- preprocessing/        # Data cleaning, encoding, scaling pipeline
+-- trainer/              # Training loop logic
+-- predictor/            # Inference (running model on new data)
+-- callbacks/            # Early stopping, model checkpointing
+-- losses/               # Custom multi-task loss function
+-- metrics/              # Evaluation metrics (MAE, F1, accuracy, etc.)
+-- utils/                # Model metadata helpers
|
+-- backend/              # FastAPI web server
|   +-- main.py           # App startup, middleware, router registration
|   +-- config.py         # Backend-specific settings
|   +-- database.py       # SQLAlchemy engine + session
|   +-- routers/          # HTTP endpoint handlers (grouped by feature)
|   +-- services/         # Business logic (calls ML code)
|   +-- schemas/          # Pydantic request/response models
|   +-- models/           # SQLAlchemy database table definitions
|
+-- frontend/             # Next.js web dashboard
|   +-- src/
|       +-- app/          # Pages (dashboard, training, predictions, drivers)
|       +-- components/   # Reusable UI components (Sidebar)
|       +-- context/      # React auth state (AuthContext)
|       +-- lib/          # API utility functions
|
+-- feature_engineering.py    # Script to add historical features to raw data
+-- generate_synthetic_data_ahmedabad_v2.py  # Creates fake-but-realistic ride data
+-- train.py                  # Standalone training entry point
+-- run_pipeline.py           # Full end-to-end pipeline (train + predict)
+-- evaluate.py               # Standalone evaluation script
+-- predict.py                # Standalone prediction script
+-- requirements.txt          # Python dependencies
```

---

## 4. Data Flow — From Start to Prediction

This is the most important section. Here is the entire journey of data, step by step.

---

### PHASE 1 — Data Creation (One Time)

```
generate_synthetic_data_ahmedabad_v2.py
     |
     |  Creates fake but realistic ride orders for Ahmedabad (30,000 rows)
     |  Each row = one ride with pickup/drop location, driver info, timestamps
     |
     v
data/ride_orders_raw.csv
```

Each row contains raw fields like:
- `order_id`, `booking_timestamp`
- `pickup_zone`, `drop_zone` (e.g., "Satellite", "Bodakdev")
- `pickup_latitude`, `pickup_longitude`, `drop_latitude`, `drop_longitude`
- `driver_id`, `driver_base_rating`, `driver_experience_days`
- `google_pickup_eta_minutes`, `google_trip_eta_minutes` (what Google Maps predicted)
- `actual_total_time_minutes` (ground truth — what actually happened)
- `order_delayed` (0 or 1 — was it actually delayed?)

---

### PHASE 2 — Feature Engineering (One Time)

```
feature_engineering.py -> FeatureEngineer class
     |
     |  Reads the raw CSV and adds 16 extra historical features per row
     |  IMPORTANT: Uses ONLY past rides to compute each row's features
     |              (no future data is used — no data leakage!)
     |
     v
data/ride_orders_engineered_D30K.csv  (training set)
data/ride_orders_test_v24.csv         (test set)
```

New features added:

| Feature | What it means |
|---|---|
| `average_delay` | This driver's average total delay across ALL past rides (EWMA) |
| `driver_avg_pickup_delay` | Average time this driver was late to pick up (EWMA) |
| `driver_avg_trip_delay` | Average extra time during trips for this driver (EWMA) |
| `driver_last_7_day_delay_rate` | % of rides delayed in last 7 days |
| `driver_last_30_day_delay_rate` | % of rides delayed in last 30 days |
| `driver_route_avg_delay` | This driver's average delay on THIS specific route |
| `driver_zone_avg_pickup_delay` | Avg pickup delay for this driver in this pickup zone |
| `driver_reliability` | % of on-time rides overall |
| `driver_risk_score` | 1 - reliability (higher = riskier driver) |
| `driver_efficiency_score` | Mix of reliability (70%) + not cancelling (30%) |
| `driver_recent_delay_rate` | % delayed in last 10 rides |
| `route_frequency` | How often this driver does this exact route |
| `historical_route_delay_rate` | % of rides delayed on this route (any driver) |
| `route_reliability_score` | 1 - route delay rate |
| `historical_route_ratio_avg` | Average ratio: actual time / google estimate on this route |

> EWMA = Exponentially Weighted Moving Average — recent rides matter more than old ones.

---

### PHASE 3 — Preprocessing (Every Training Run)

```
DataValidator -> CategoricalEncoder -> NumericalScaler
     |
     |  1. Validator: checks all required columns exist, no nulls
     |  2. Encoder: converts text categories to integers
     |              e.g., "Satellite" -> 7, "Bodakdev" -> 2
     |  3. Scaler:  normalizes numbers to mean=0, std=1
     |              e.g., 45 km -> 0.8
     |
     |  DataPipeline ties all 3 together
     |
     v
Clean numeric DataFrame ready for PyTorch
```

Saved artifacts (after training):
- `saved_models/categorical_encoder.pkl` — knows how to convert zones/drivers to numbers
- `saved_models/numerical_scaler.pkl` — knows the mean/std for normalization
- `saved_models/ride_eta_metadata.json` — records model architecture info

---

### PHASE 4 — PyTorch Dataset + DataLoader

```
RideDataset (datasets/ride_dataset.py)
     |
     |  Wraps the DataFrame into a PyTorch Dataset
     |  Each item returns a dict:
     |  {
     |    "numerical_features": tensor of 38 numbers
     |    "categorical_features": tensor of 6 integers
     |    "eta_target": the actual total time
     |    "delay_target": 0 or 1
     |  }
     |
     v
DataLoader (datasets/dataloader.py)
     |
     |  Batches the data (default: 256 rides per batch)
     |  Shuffles training data each epoch
     |
     v
Batches fed into the neural network
```

---

### PHASE 5 — Neural Network Forward Pass

```
RideETANetwork (models/ride_eta_network.py)
|
|  INPUT: 38 numerical features + 6 categorical features
|
+-- NumericalProjection
|      Linear(38 -> 64) + BatchNorm + ReLU
|      Takes the 38 numbers and projects them to a 64-dim space
|
+-- CategoricalEmbedding
|      6 separate Embedding tables (one per category)
|      e.g., driver_id -> 50-dim vector, pickup_zone -> 11-dim vector
|      All embeddings concatenated -> ~160-dim vector
|
+-- SharedBackbone
|      Linear(64+160=224 -> 256) + BatchNorm + ReLU + Dropout(0.3)
|      Linear(256 -> 128) + BatchNorm + ReLU + Dropout(0.3)
|      Linear(128 -> 64)  + BatchNorm + ReLU + Dropout(0.3)
|
+-- ETA Head (PredictionHead)
|      Linear(64 -> 32) + ReLU + Dropout + Linear(32 -> 1)
|      softplus activation ensures output is always positive
|      OUTPUT: predicted minutes (e.g., 18.5)
|
+-- Delay Head (PredictionHead)
       Linear(64 -> 32) + ReLU + Dropout + Linear(32 -> 1)
       sigmoid applied later -> probability between 0 and 1
       OUTPUT: delay probability (e.g., 0.72 = 72% chance of delay)
```

---

### PHASE 6 — Loss Calculation + Backpropagation

```
MultiTaskLoss (losses/losses.py)
|
|  ETA Loss   = HuberLoss(predicted_minutes, actual_minutes)
|               Huber is like MAE but less sensitive to outliers
|
|  Delay Loss = BCEWithLogitsLoss(delay_logit, 0_or_1)
|               Standard binary cross-entropy for yes/no classification
|
|  Total Loss = (1.0 x ETA_loss) + (1.0 x Delay_loss)
|
|  loss.backward() -> computes gradients
|  optimizer.step() -> updates all model weights (Adam optimizer)
```

---

### PHASE 7 — Training Loop (Per Epoch)

```
Trainer (trainer/trainer.py)
|
|  For each epoch (up to 100):
|  +-- _train_epoch(): runs all training batches, updates weights
|  +-- _validate_epoch(): runs validation batches (no weight updates)
|  +-- Logs: loss, MAE (ETA), F1 (delay) for both train/val
|  +-- ModelCheckpoint: saves best model weights if val_loss improved
|  +-- EarlyStopping: stops if val_loss does not improve for 10 epochs
|
+-- Returns training history dict
```

---

### PHASE 8 — Prediction / Inference

```
Predictor (predictor/predictor.py)
|
|  Loads trained model + preprocessing artifacts
|  Applies the SAME encoder/scaler used during training
|  (uses .transform() NOT .fit_transform())
|  Runs model in eval mode (no gradients, dropout disabled)
|
|  Outputs:
|  {
|    "eta": [18.5, 22.1, 31.7, ...]  <- predicted minutes
|    "delay_probability": [0.72, 0.1, 0.9, ...]
|    "delay": [1, 0, 1, ...]  <- 1 if probability >= 0.3 threshold
|  }
```

---

### PHASE 9 — API + Frontend

```
User clicks "Run Prediction" in the browser
     |
     v
Next.js frontend (api.js) -> POST /api/admin/run-prediction (with CSV file)
     |
     v
FastAPI backend (admin.py router) -> prediction_service.py
     |
     |  Runs the entire prediction pipeline
     |  Saves results to SQLite database
     |
     v
Frontend fetches predictions -> GET /api/predictions
     |
     v
Table with order_id, predicted ETA, delay probability, actual time shown to user
```

---

## 5. ML Layer — Every File Explained

### configs/config.py
**The brain of all settings.** Every number, path, and flag in the project lives here.

| Section | Key Settings |
|---|---|
| **Paths** | DATA_DIR, SAVED_MODEL_DIR, TRAIN_DATA_FILE, TEST_DATA_FILE |
| **Features** | CONTINUOUS_COLUMNS (26), TIME_COLUMNS (4), BINARY_COLUMNS (4), CATEGORICAL_COLUMNS (6) |
| **Targets** | ETA_TARGET = "actual_total_time_minutes", DELAY_TARGET = "order_delayed" |
| **Model** | HIDDEN_LAYERS = [256, 128, 64], DROPOUT = 0.30, NUMERICAL_PROJECTION_SIZE = 64 |
| **Training** | EPOCHS = 100, BATCH_SIZE = 256, LEARNING_RATE = 0.001, WEIGHT_DECAY = 1e-5 |
| **Stopping** | EARLY_STOPPING_PATIENCE = 10 |
| **Data split** | TRAIN_SIZE = 0.85, VALIDATION_SIZE = 0.15 |
| **Geography** | 23 Ahmedabad zones with real coordinates, festival dates, peak hours |

---

### generate_synthetic_data_ahmedabad_v2.py
Generates realistic fake ride data for Ahmedabad, Gujarat.

- Creates **30,000 ride orders** with real zone names (Satellite, Bopal, SG Highway, etc.)
- Simulates driver behavior: some drivers are reliably on time, others always late
- Adds seasonal effects: Navratri, Diwali, Uttarayan festivals cause more delays
- Adds peak hour effects: 8-11 AM and 5-9 PM have higher delay rates
- Uses H3 geospatial indexing for pickup/drop cell encoding

---

### feature_engineering.py -> FeatureEngineer class

Processes rows **one by one in chronological order**, building up a history of what each driver has done before this ride.

**Key method: `_create_historical_features(df)`**

Uses 4 dictionaries that grow as rides are processed:
- `driver_history`: stores all past delays, timestamps per driver
- `driver_route_history`: stores delays per (driver, pickup_zone, drop_zone)
- `driver_zone_history`: stores delays per (driver, pickup_zone)
- `route_history`: stores delays for this route regardless of driver

For each ride, it looks backwards into these dictionaries. After computing, it adds the current ride to the dictionaries for future rows.

**Cold Start Problem:** When a new driver appears for the first time, default values are used:
- Average delay -> 5.0 minutes
- Delay rates -> 0.3 (30% default)
- Reliability -> 0.7 (70% default)

---

### preprocessing/validator.py -> DataValidator
Checks the DataFrame before training. All required columns must exist, raises an error if any column is missing.

### preprocessing/encoder.py -> CategoricalEncoder
Converts text categories to integers using sklearn LabelEncoder.
- `fit_transform()`: learns the mapping AND applies it (training only)
- `transform()`: applies the already-learned mapping (inference)
- Saves/loads to `saved_models/categorical_encoder.pkl`
- `get_cardinalities()`: returns unique value counts (needed for embedding layer sizes)

### preprocessing/scaler.py -> NumericalScaler
Normalizes numbers to mean=0, std=1 using sklearn StandardScaler.
- Only applies to continuous + time columns
- Binary columns (0/1) are left unchanged
- Saves/loads to `saved_models/numerical_scaler.pkl`

### preprocessing/pipeline.py -> DataPipeline
Chains the 3 preprocessing steps together.
- `fit_transform(df)`: Validates + fits encoder + fits scaler + transforms (use for training)
- `transform(df)`: Validates + applies fitted encoder + scaler (use for inference)
- `split_dataset(df)`: Splits 85% train / 15% validation chronologically

---

### datasets/ride_dataset.py -> RideDataset
A PyTorch Dataset class. Each item returns:
```
{
  "numerical_features": FloatTensor of 38 numbers,
  "categorical_features": LongTensor of 6 integers,
  "eta_target": FloatTensor with actual minutes,
  "delay_target": FloatTensor with 0 or 1
}
```

### datasets/dataloader.py -> create_dataloader()
Wraps the Dataset in a PyTorch DataLoader. Batches 256 rides together, shuffles during training.

---

### models/ride_eta_network.py -> RideETANetwork

**NumericalProjection**: Linear(38->64) + BatchNorm + ReLU. Learns a dense representation of the numbers.

**CategoricalEmbedding**: 6 separate embedding tables (one per category). Embedding size = min(50, round(1.6 x cardinality^0.56)). All concatenated into ~160-dim vector.

**SharedBackbone**: 3 hidden layers [256, 128, 64] with BatchNorm + ReLU + Dropout(30%). Both ETA and delay predictions share this feature extractor.

**PredictionHead** (used twice — ETA and delay):
- Linear(64->32) + ReLU + Dropout + Linear(32->1)
- ETA uses softplus (always positive)
- Delay outputs raw logit (sigmoid applied during loss)

---

### losses/losses.py -> MultiTaskLoss
Combines two objectives:
- **HuberLoss** for ETA: robust to outlier rides
- **BCEWithLogitsLoss** for delay: standard binary classification loss
- `total = (1.0 x eta_loss) + (1.0 x delay_loss)`

### trainer/trainer.py -> Trainer
Manages the full training loop. Runs epochs, calls checkpoint and early stopping, returns full history dict.

### callbacks/checkpoint.py -> ModelCheckpoint
Saves model weights whenever validation loss improves using `torch.save()`.

### callbacks/early_stopping.py -> EarlyStopping
Stops training if validation loss doesn't improve for `patience=10` consecutive epochs.

### metrics/regression_metrics.py
Computes MAE and RMSE for ETA predictions.

### metrics/classification_metrics.py
Computes Accuracy, Precision, Recall, F1-Score for delay classification.

### predictor/predictor.py -> Predictor
Runs model in eval mode (dropout disabled). Returns eta, delay_probability, and delay tensors.

### utils/model_metadata.py
Saves and loads a JSON file recording `numerical_feature_count` and `categorical_cardinalities`. Needed at inference time to rebuild model architecture.

---

### train.py
Standalone training script: load CSV -> DataPipeline.fit_transform -> split 85/15 -> build model -> train -> save artifacts.

### run_pipeline.py
Full pipeline: train on training set, evaluate on test set, then run inference and print sample predictions.

### evaluate.py
Load a saved model, evaluate it on test data only (no training).

### predict.py
Load a saved model, run inference on a new CSV file.

---

## 6. Backend Layer — Every File Explained

### backend/config.py
Backend-specific settings:
- `DATABASE_URL = "sqlite:///./ride_eta.db"` — SQLite database file location
- `ML_DIR = PROJECT_ROOT` — where the ML Python code lives (added to sys.path)
- `CORS_ORIGINS = ["http://localhost:3000"]` — allows Next.js to call the API

### backend/database.py
Sets up the database connection:
- `engine` — the SQLite connection
- `SessionLocal` — creates database sessions
- `Base` — parent class for all ORM models
- `get_db()` — FastAPI dependency, creates a session per request and closes it after

### backend/main.py
The FastAPI app entry point:
1. Creates all database tables (`Base.metadata.create_all`)
2. Adds CORS middleware so Next.js (port 3000) can call the API (port 8000)
3. Registers all 7 routers
4. Exposes health check at `GET /api/health`

### backend/seed_data.py
Pre-populates the database with sample data so the dashboard has something to show on first launch. Creates demo users, drivers, ride orders, and predictions.

---

### Backend Database Models (backend/models/)

| File | Table | Purpose |
|---|---|---|
| `user.py` | `users` | Login credentials + role (admin/viewer) |
| `driver.py` | `drivers` | Driver profiles with aggregated stats |
| `raw_ride_order.py` | `raw_ride_orders` | Original ride data as uploaded |
| `engineered_ride_order.py` | `engineered_ride_orders` | Ride data after feature engineering |
| `prediction.py` | `predictions` | Model output for each ride |
| `training_run.py` | `training_runs` | One row per training session |
| `training_epoch.py` | `training_epochs` | One row per epoch per training session |

---

### Backend Schemas (backend/schemas/)
Pydantic models defining the exact shape of API request/response JSON.

| File | What it validates |
|---|---|
| `auth.py` | Login request body, JWT token response |
| `prediction.py` | Prediction response, stats, paginated list |
| `training.py` | Training run details, epoch data, status |
| `driver.py` | Driver profile response |
| `admin.py` | Pipeline status, config updates |

---

### Backend Routers (backend/routers/)

**auth.py**: POST /api/auth/login (get JWT token), GET /api/auth/me (current user)

**predictions.py**: GET /api/predictions (list with pagination/filter/sort), GET /api/predictions/stats (summary), GET /api/predictions/{id}

**training.py**: POST /api/training/start (admin), POST /api/training/stop (admin), GET /api/training/status, GET /api/training/runs, GET /api/training/runs/{id}/epochs

**drivers.py**: GET /api/drivers (paginated), GET /api/drivers/top-performers, GET /api/drivers/worst-performers

**admin.py**: GET|PUT /api/admin/config, POST /api/admin/upload-csv, POST /api/admin/run-feature-engineering, POST /api/admin/run-prediction

**model_metrics.py**: GET /api/model/metrics, GET /api/model/history

**websocket.py**: WS /ws/training — polls DB every 2 seconds, sends epoch updates to the browser ONLY when something new happened (efficient).

---

### Backend Services (backend/services/)

| File | What it does |
|---|---|
| `auth_service.py` | JWT token creation/validation, password hashing, role checking |
| `training_service.py` | Runs the full ML training loop in a **background thread**, logs each epoch to DB |
| `prediction_service.py` | Loads saved model, runs predictions on uploaded CSV, saves results to DB |
| `data_service.py` | Loads training data from database or falls back to CSV |
| `driver_service.py` | Computes driver stats from raw ride orders |
| `feature_engineering_service.py` | Runs the FeatureEngineer class from the ML layer |

**Key design: training_service.py uses a background thread**
```python
thread = threading.Thread(target=_run_training, args=(run_id,), daemon=True)
thread.start()
# Returns immediately so the HTTP response does not hang
```
The frontend connects via WebSocket to receive progress as training runs.

---

## 7. Frontend Layer — Every File Explained

### frontend/src/lib/api.js
Central API utility. All backend calls go through `fetchApi()` which automatically adds the JWT token. Exports functions for auth, predictions, training, drivers, and admin operations.

### frontend/src/context/AuthContext.js
React Context managing authentication state globally. Stores user + token, exposes `login()` and `logout()`. Token stored in localStorage so users stay logged in on refresh.

### frontend/src/components/Sidebar.js
Navigation sidebar that appears on every page. Links to: Dashboard, Predictions, Training, Drivers, Admin.

### frontend/src/app/layout.js
Root layout wrapping every page with AuthProvider context, global CSS, and Sidebar.

### frontend/src/app/page.js
Root page — redirects to /dashboard or /login based on auth state.

### frontend/src/app/login/page.js
Login form. Calls `loginUser()` and stores JWT token in localStorage.

### frontend/src/app/dashboard/page.js
Main dashboard showing summary stats cards, ETA accuracy chart, delay distribution chart, and top/worst driver summary. Uses Chart.js for visualizations.

### frontend/src/app/training/page.js
Training control panel with Start/Stop buttons. Shows real-time charts for training loss, validation loss, MAE, F1 per epoch via WebSocket. Displays epoch progress bar and historical training runs.

### frontend/src/app/predictions/page.js
Predictions table with CSV upload to trigger new predictions. Shows each ride's predicted ETA, actual time, delay probability. Filter by delayed/on-time, search by order ID, paginated and sortable.

### frontend/src/app/drivers/page.js
Driver analytics showing top 10 performers, worst 10 performers, and full driver table with delay rate, reliability score, and experience. Search and sort supported.

### frontend/src/app/admin/page.js
Admin-only panel for uploading raw CSV data, triggering feature engineering, viewing/editing model configuration, and checking pipeline status.

---

## 8. Database Tables Explained

```
users
+-- id, username, hashed_password, role (admin/viewer)

drivers
+-- id, driver_id (original string), total_rides, delay_rate
+-- avg_pickup_delay, avg_trip_delay, reliability_score, efficiency_score

raw_ride_orders
+-- id + all original CSV columns (50+ fields)

engineered_ride_orders
+-- id + all engineered features (66+ fields)

predictions
+-- id, order_id, booking_timestamp
+-- actual_total_time_minutes   <- ground truth
+-- google_total_eta_minutes    <- what Google Maps predicted
+-- predicted_eta_minutes       <- what OUR model predicted
+-- predicted_delay_probability <- 0.0 to 1.0
+-- predicted_delay             <- 0 or 1 (threshold: 0.3)
+-- order_delayed               <- actual label
+-- training_run_id             <- which model produced this

training_runs
+-- id, status (running/completed/failed/stopped)
+-- total_epochs, completed_epochs, best_epoch
+-- best_val_loss, final_train_loss, final_val_loss
+-- learning_rate, batch_size, threshold
+-- train_samples, val_samples
+-- started_at, completed_at

training_epochs
+-- id, training_run_id, epoch
+-- train_loss, val_loss
+-- train_mae, val_mae         <- ETA regression metrics
+-- train_f1, val_f1           <- delay classification metrics
```

---

## 9. API Endpoints Reference

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| GET | /api/health | None | Server health check |
| POST | /api/auth/login | None | Get JWT token |
| GET | /api/auth/me | Bearer | Current user info |
| GET | /api/predictions | Bearer | List predictions (paginated) |
| GET | /api/predictions/stats | Bearer | Summary statistics |
| GET | /api/predictions/{id} | Bearer | Single prediction |
| POST | /api/training/start | Admin | Start training |
| POST | /api/training/stop | Admin | Stop training |
| GET | /api/training/status | Bearer | Is training running? |
| GET | /api/training/runs | Bearer | All training sessions |
| GET | /api/training/runs/{id}/epochs | Bearer | Per-epoch data |
| GET | /api/model/metrics | Bearer | Latest model metrics |
| GET | /api/model/history | Bearer | Training history |
| GET | /api/drivers | Bearer | Driver list (paginated) |
| GET | /api/drivers/top-performers | Bearer | Best 10 drivers |
| GET | /api/drivers/worst-performers | Bearer | Worst 10 drivers |
| GET | /api/admin/config | Admin | View config |
| PUT | /api/admin/config | Admin | Update config |
| POST | /api/admin/upload-csv | Admin | Upload raw data |
| POST | /api/admin/run-feature-engineering | Admin | Run feature engineering |
| POST | /api/admin/run-prediction | Admin | Run predictions on CSV |
| WS | /ws/training | None | Live training updates |

---

## 10. How to Run the Project

### Prerequisites
- Python 3.10+
- Node.js 18+

### Backend Setup

```bash
# 1. Create virtual environment
python -m venv venv
venv\Scripts\activate   # Windows

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Start the backend server
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Backend API live at: http://localhost:8000
Interactive API docs: http://localhost:8000/docs

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

Dashboard live at: http://localhost:3000

### First Time: Generate Training Data

```bash
# Step 1: Generate synthetic Ahmedabad ride data
python generate_synthetic_data_ahmedabad_v2.py

# Step 2: Add historical features to the data
python feature_engineering.py

# Step 3: Train the model
python train.py

# OR run everything at once:
python run_pipeline.py
```

---

## 11. Configuration Reference

All settings live in configs/config.py.

| Setting | Default | What it controls |
|---|---|---|
| EPOCHS | 100 | Max training epochs |
| BATCH_SIZE | 256 | Rides per training batch |
| LEARNING_RATE | 0.001 | How fast the model learns |
| EARLY_STOPPING_PATIENCE | 10 | Epochs without improvement before stopping |
| HIDDEN_LAYERS | [256, 128, 64] | Neural network depth/width |
| DROPOUT | 0.30 | Regularization strength |
| DELAY_THRESHOLD | 0.3 | Probability cutoff for classifying as "delayed" |
| TRAIN_SIZE | 0.85 | 85% for training, 15% for validation |
| N_ROWS | 30000 | How many synthetic rides to generate |
| N_DRIVERS | 5000 | Number of unique drivers |
| QUICK_DEMO_MODE | False | Set True for faster testing with 10k sample |
| ETA_LOSS_WEIGHT | 1.0 | How much ETA loss contributes to total loss |
| DELAY_LOSS_WEIGHT | 1.0 | How much delay loss contributes to total loss |

---

## Key Design Decisions

| Decision | Why |
|---|---|
| Multi-task learning | One model predicts both ETA and delay — shared backbone learns better representations |
| Embedding layers for categories | Better than one-hot encoding — captures similarity between zones/drivers |
| EWMA for historical features | Recent behavior matters more than old history |
| Chronological split (not random) | Prevents data leakage — never train on "future" data |
| Background thread for training | Training takes minutes/hours — cannot block the HTTP request |
| WebSocket for live updates | Polling would flood the server; WebSocket is efficient and real-time |
| SQLite database | Simple, file-based, no setup needed for local development |
| HuberLoss for ETA | More robust than MSE — outlier rides do not break the model |
| softplus on ETA head | ETA can never be negative (time does not go backwards) |
| Threshold = 0.3 for delay | Lower than 0.5 because missing a delay is worse than a false alarm |

---

*Ride ETA v1.0 — Ahmedabad, India synthetic dataset*
