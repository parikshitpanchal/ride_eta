from pathlib import Path

# pyrefly: ignore [missing-import]
import torch

from configs import config

class ModelCheckpoint:
    def __init__(self,save_directory: Path,checkpoint_name: str,) -> None:

        save_directory.mkdir(parents=True,exist_ok=True,)
        self.save_path = save_directory / checkpoint_name
        self.best_loss = float("inf")

    def __call__(self,model: torch.nn.Module,validation_loss: float,) -> bool:

        if validation_loss >= self.best_loss:
            return False

        self.best_loss = validation_loss

        torch.save(
            model.state_dict(),
            self.save_path,
        )

        return True