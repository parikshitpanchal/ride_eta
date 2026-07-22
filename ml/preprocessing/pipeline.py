import pandas as pd

from ml.configs import config
from ml.preprocessing.encoder import CategoricalEncoder
from ml.preprocessing.scaler import NumericalScaler
from ml.preprocessing.validator import DataValidator


class DataPipeline:
    def __init__(self,validator: DataValidator,encoder: CategoricalEncoder,scaler: NumericalScaler) -> None:

        self.validator = validator
        self.encoder = encoder
        self.scaler = scaler

    def fit_transform(self,dataframe: pd.DataFrame,) -> pd.DataFrame:

        self.validator.validate(dataframe)

        dataframe = self.encoder.fit_transform(dataframe)

        dataframe = self.scaler.fit_transform(dataframe)

        return dataframe

    def transform(self,dataframe: pd.DataFrame,) -> pd.DataFrame:

        self.validator.validate(dataframe)

        dataframe = self.encoder.transform(dataframe)

        dataframe = self.scaler.transform(dataframe)

        return dataframe

    def split_dataset(self,dataframe: pd.DataFrame,) -> tuple[pd.DataFrame,pd.DataFrame,]:

        train_end = int(
            len(dataframe) * config.TRAIN_SIZE
    )

    #     validation_end = train_end + int(
    #         len(dataframe) * config.VALIDATION_SIZE
    # )

        train_dataframe = dataframe[:train_end].copy()
        validation_dataframe = dataframe[train_end:].copy()

        # validation_dataframe = dataframe.iloc[train_end:validation_end].copy()

        # test_dataframe = dataframe.iloc[validation_end:].copy()

        return (
            train_dataframe,
            validation_dataframe
            # test_dataframe,
        )

