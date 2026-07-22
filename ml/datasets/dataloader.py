import logging

# pyrefly: ignore [missing-import]
from torch.utils.data import DataLoader

from ml.configs import config

logger = logging.getLogger(__name__)

def create_dataloader(dataset,shuffle: bool = True):
    logger.info("Creating DataLoader...")

    loader = DataLoader(
        dataset=dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=shuffle,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        drop_last=config.DROP_LAST
    )

    logger.info(f"DataLoader created with batch size: {config.BATCH_SIZE}")

    return loader


