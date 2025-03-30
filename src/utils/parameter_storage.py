class ParameterStorage:

    def __init__(
        self,
        name,
        model_architecture,
        model_type,
        dataset,
        size,
        do_oversampling,
        do_class_weights,
        optimizer,
        learning_rate,
        weight_decay,
        criterion,
        scheduler,
        epochs,
        batch_size,
        solarize,
        saturation,
        contrast,
        brightness,
        sharpness,
        hue,
        posterization,
        rotation,
        erasing,
        affine,
        crop,
        gaussian_noise,
    ):
        self.name = name
        self.model_architecture = model_architecture
        self.model_type = model_type
        self.dataset = dataset
        self.size = size
        self.do_oversampling = do_oversampling
        self.do_class_weights = do_class_weights
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.criterion = criterion
        self.scheduler = scheduler
        self.epochs = epochs
        self.batch_size = batch_size
        self.solarize = solarize
        self.saturation = saturation
        self.contrast = contrast
        self.brightness = brightness
        self.sharpness = sharpness
        self.hue = hue
        self.posterization = posterization
        self.rotation = rotation
        self.erasing = erasing
        self.affine = affine
        self.crop = crop
        self.gaussian_noise = gaussian_noise
