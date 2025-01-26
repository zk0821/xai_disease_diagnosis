from evaluation.set_evaluator import SetEvaluator

class Evaluator():

    def __init__(self, classes):
        self.classes = classes
        self.num_classes = len(classes)
        self.train_evaluator = SetEvaluator(self.classes, self.num_classes)
        self.validation_evaluator = SetEvaluator(self.classes, self.num_classes)
        self.test_evaluator = SetEvaluator(self.classes, self.num_classes)