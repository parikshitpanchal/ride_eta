import logging

import numpy as np
import pandas as pd

# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
from torch.utils.data import Dataset

from configs import config

logger = logging.getLogger(__name__)

class RideDataset(Dataset):

    def __init__(self, dataframe: pd.DataFrame):

        logger.info("Initializing RideDataset...")
        
        self.length = len(dataframe)

        self.numerical_features = dataframe[
            config.NUMERICAL_COLUMNS
            ].to_numpy(dtype=np.float32)

        self.categorical_features = dataframe[
            config.CATEGORICAL_COLUMNS
            ].to_numpy(dtype=np.int64)

        self.eta_targets = dataframe[
            config.ETA_TARGET
            ].to_numpy(dtype=np.float32)

        self.delay_targets = dataframe[
            config.DELAY_TARGET
            ].to_numpy(dtype=np.float32)

    def __len__(self) -> int:

        return self.length

    def __getitem__(self, index):

         # Extract numerical features
        numerical_features = torch.tensor(
            self.numerical_features[index],
            dtype=torch.float32,
        )

        # Extract categorical features
        categorical_features = torch.tensor(
            self.categorical_features[index],
            dtype=torch.long,
        )

        # Extract targets
        eta_target = torch.tensor(
            [self.eta_targets[index]],
            dtype=torch.float32,
        )

        delay_target = torch.tensor(
            [self.delay_targets[index]],
            dtype=torch.float32,
        )

        # Return one complete sample
        return {
            "numerical_features": numerical_features,
            "categorical_features": categorical_features,
            "eta_target": eta_target,
            "delay_target": delay_target
        }
        