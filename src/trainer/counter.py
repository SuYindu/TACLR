class CustomCounter:
    def __init__(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.support = 0

        self.precision = 0.0
        self.recall = 0.0
        self.f1_score = 0.0

    def add(self, tp, fp, fn):
        self.tp += tp
        self.fp += fp
        self.fn += fn

    def compute_scores(self):
        if self.tp != 0:
            self.precision = self.tp / (self.tp + self.fp)
            self.recall = self.tp / (self.tp + self.fn)
            self.f1_score = 2 * self.tp / (2 * self.tp + self.fp + self.fn)
        self.support = self.tp + self.fp + self.fn
