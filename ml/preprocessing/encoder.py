import logging
import joblib
import pandas as pd

from sklearn.preprocessing import OrdinalEncoder

from ml.configs import config

logger = logging.getLogger(__name__)


class CategoricalEncoder:

    def __init__(self):

        logger.info("Initializing Categorical Encoder...")

        self.encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1
        )

    def fit(self, df: pd.DataFrame):

        logger.info("Fitting categorical encoder...")

        missing = set(config.CATEGORICAL_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(
                f"Missing categorical columns: {sorted(missing)}"
            )

        self.encoder.fit(df[config.CATEGORICAL_COLUMNS])

        logger.info("Categorical encoder fitted successfully.")

    def transform(self, df: pd.DataFrame):

        logger.info(f"Encoding {len(config.CATEGORICAL_COLUMNS)} categorical columns...")

        transformed_df = df.copy()

        transformed_df[config.CATEGORICAL_COLUMNS] = (
            self.encoder.transform(
                transformed_df[config.CATEGORICAL_COLUMNS]
            )
        )

        return transformed_df

    def fit_transform(self, df: pd.DataFrame):

        self.fit(df)

        return self.transform(df)


    def load(self):
        logger.info("Loading categorical encoder...")

        if not (config.ARTIFACTS_DIR / config.ENCODER_FILE).exists():
            raise FileNotFoundError(
                "Categorical encoder has not been trained yet."
            )

        self.encoder = joblib.load(
            config.ARTIFACTS_DIR / config.ENCODER_FILE
        )

        logger.info("Categorical encoder loaded successfully.")

    def save(self):

        config.ARTIFACTS_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        logger.info("Saving categorical encoder...")
        joblib.dump(
            self.encoder,
            config.ARTIFACTS_DIR / config.ENCODER_FILE
        )
        logger.info("Categorical encoder saved successfully.")

    def get_cardinalities(self,) -> dict[str, int]:
        return {
            column: len(categories)
            for column, categories in zip(
                config.CATEGORICAL_COLUMNS,
                self.encoder.categories_,
            )
        }
    