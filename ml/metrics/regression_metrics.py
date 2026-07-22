import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

def calculate_regression_metrics(predictions: np.ndarray,targets: np.ndarray,) -> dict[str, float]:
    
    mae = mean_absolute_error(targets,predictions,)
    rmse = np.sqrt(mean_squared_error(targets,predictions))
    r2 = r2_score(targets,predictions,)

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
    }