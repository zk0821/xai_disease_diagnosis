class ModelCheckpoint:
    def __init__(self, enabled, use_loss=True, use_kappa=False):
        self.enabled = enabled
        self.use_loss = use_loss
        self.use_kappa = use_kappa
        self.min_validation_loss = float("inf")
        self.max_balanced_accuracy = float("-inf")
        self.max_kappa = float("-inf")

    def save_checkpoint(self, validation_loss, validation_balanced_accuracy, validation_kappa):
        if self.enabled:
            if self.use_loss:
                if validation_loss < self.min_validation_loss:
                    self.min_validation_loss = validation_loss
                    return True
                else:
                    return False
            elif self.use_kappa:
                if validation_kappa > self.max_kappa:
                    self.max_kappa = validation_kappa
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
