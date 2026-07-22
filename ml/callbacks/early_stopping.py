import numpy as np 

class EarlyStopping:
    def __init__(self,patience: int,) -> None:
        self.patience = patience

        self.best_loss = np.inf

        self.counter = 0

        self.should_stop = False

    def __call__(self,validation_loss: float,) -> bool:

        if validation_loss < self.best_loss:
            self.best_loss = validation_loss
            self.counter = 0

        else:
            self.counter += 1

            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop