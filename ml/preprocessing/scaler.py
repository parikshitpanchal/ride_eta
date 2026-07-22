import logging
import joblib
import pandas as pd

from sklearn.preprocessing import StandardScaler

from ml.configs import config

logger = logging.getLogger(__name__)

class NumericalScaler:

    def __init__(self):

        logger.info("Initializing Numerical Scaler...")

        self.scaler = StandardScaler()

    def fit(self, df: pd.DataFrame):

        logger.info("Fitting numerical scaler...")

        self.scaler.fit(df[config.SCALING_COLUMNS])

        logger.info("Numerical scaler fitted successfully.")

    def transform(self, df: pd.DataFrame):

        logger.info("Scaling numerical features...")

        transformed_df = df.copy()
        
        transformed_df[config.SCALING_COLUMNS] = (
            self.scaler.transform(
                transformed_df[config.SCALING_COLUMNS]
            )
        )

        return transformed_df

    
    def fit_transform(self, df: pd.DataFrame):

        self.fit(df)

        return self.transform(df)

    def load(self):

        logger.info("Loading numerical scaler...")

        scaler_path = (
            config.ARTIFACTS_DIR /
            config.SCALER_FILE
        )

        if not scaler_path.exists():
            raise FileNotFoundError(
                "Numerical scaler has not been trained yet."
            )

        self.scaler = joblib.load(scaler_path)

        logger.info("Numerical scaler loaded successfully.")

    def save(self):

        logger.info("Saving numerical scaler...")

        config.ARTIFACTS_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        joblib.dump(
        self.scaler,
        config.ARTIFACTS_DIR / config.SCALER_FILE
        )

        logger.info("Numerical scaler saved successfully.")