import logging

import pandas as pd

from ml.configs import config

logger = logging.getLogger(__name__)

class DataValidator:

    def __init__(self):
        logger.info("Initialized Data Validator")

    def validate(self, df):
        self._check_empty_dataset(df)
        self._check_required_columns(df)
        self._check_duplicate_orders(df)
        self._check_missing_targets(df)
        self._report_missing_features(df)

        logger.info("Dataset validation completed.")

    def _check_empty_dataset(self, df: pd.DataFrame):

        logger.info("Checking for empty dataset...")

        if df.empty:
            raise ValueError("Dataset is empty.")

        logger.info(f"Dataset contains {len(df)} rows.")


    def _check_required_columns(self, df: pd.DataFrame):
        logger.info("Checking required columns...")

        missing_columns = set(config.REQUIRED_COLUMNS) - set(df.columns)

        if missing_columns:
            raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

        logger.info("All required columns are present.")

    def _check_duplicate_orders(self, df: pd.DataFrame):

        logger.info("Checking duplicate order IDs...")

        duplicate_count = df[config.ORDER_ID_COLUMN].duplicated().sum()

        if duplicate_count > 0:
            raise ValueError(
                f"Found {duplicate_count} duplicate order IDs."
            )

        logger.info("No duplicate order IDs found.")

    def _check_missing_targets(self, df: pd.DataFrame):

        logger.info("Checking target columns...")

        eta_missing = df[config.ETA_TARGET].isna().sum()
        delay_missing = df[config.DELAY_TARGET].isna().sum()

        if eta_missing > 0:
            raise ValueError(
                f"{config.ETA_TARGET} contains {eta_missing} missing values."
            )

        if delay_missing > 0:
            raise ValueError(
                f"{config.DELAY_TARGET} contains {delay_missing} missing values."
            )

        logger.info("Target columns are valid.")
    
    def _report_missing_features(self, df: pd.DataFrame):

        logger.info("Checking feature columns...")

        missing_report = {}

        for column in config.FEATURE_COLUMNS:

            missing = df[column].isna().sum()

            if missing > 0:
                missing_report[column] = missing

        if missing_report:

            logger.warning(
                f"Missing feature values found: {missing_report}"
            )

        else:
            logger.info("No missing feature values found.")


