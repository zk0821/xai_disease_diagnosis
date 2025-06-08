class EarlyStoppage:
    def __init__(self, patience=1, min_delta=0.1, max_patience=10, final_epoch=None):
        self.patience = patience
        self.min_delta = min_delta
        self.max_patience = max_patience
        self.counter = 0
        self.max_counter = 0
        self.min_validation_loss = float("inf")
        self.final_epoch = final_epoch

    def get_min_validation_loss(self):
        return self.min_validation_loss

    def early_stop(self, validation_loss):
        if self.final_epoch is None:
            if validation_loss < self.min_validation_loss:
                self.min_validation_loss = validation_loss
                self.counter = 0
                self.max_counter = 0
            elif validation_loss > (self.min_validation_loss + self.min_delta):
                self.counter += 1
                self.max_counter += 1
                if self.counter >= self.patience:
                    return True
                elif self.max_counter >= self.max_patience:
                    return True
            else:
                self.max_counter += 1
                if self.max_counter >= self.max_patience:
                    return True
            return False
        else:
            if self.counter > self.final_epoch:
                return True
            else:
                self.counter += 1
                return False
