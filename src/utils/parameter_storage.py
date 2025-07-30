class ParameterStorage:

    def __init__(
        self,
        name,
        model_architecture,
        model_type,
        dataset,
        size,
        class_weights,
        weight_strategy,
        optimizer,
        learning_rate,
        weight_decay,
        criterion,
        scheduler,
        model_checkpoint,
        early_stoppage,
        epochs,
        batch_size,
        focal_loss_gamma,
        train_augmentation_policy,
        train_augmentation_probability,
        train_augmentation_magnitude,
        test_augmentation_policy,
        random_seed,
    ):
        self.name = name
        self.model_architecture = model_architecture
        self.model_type = model_type
        self.dataset = dataset
        self.size = size
        self.class_weights = class_weights
        self.weight_strategy = weight_strategy
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.criterion = criterion
        self.scheduler = scheduler
        self.model_checkpoint = model_checkpoint
        self.early_stoppage = early_stoppage
        self.epochs = epochs
        self.batch_size = batch_size
        self.focal_loss_gamma = focal_loss_gamma
        self.train_augmentation_policy = train_augmentation_policy
        self.train_augmentation_probability = train_augmentation_probability
        self.train_augmentation_magnitude = train_augmentation_magnitude
        self.test_augmentation_policy = test_augmentation_policy
        self.random_seed = random_seed
