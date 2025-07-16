class ModelCheckpoint():
    def __init__(self, enabled, use_loss=True):
        self.enabled = enabled
        self.use_loss = use_loss
        self.min_validation_loss = float("inf")
        self.max_balanced_accuracy = float("-inf")

    def save_checkpoint(self, validation_loss, validation_balanced_accuracy):
        if self.enabled:
            if self.use_loss:
                if validation_loss < self.min_validation_loss:
                    self.min_validation_loss = validation_loss
                    return True
                else:
                    return False
            else:
                if validation_balanced_accuracy > self.max_balanced_accuracy:
                    self.max_balanced_accuracy = validation_balanced_accuracy
                    return True
                else:
                    return False
        else:
            return True