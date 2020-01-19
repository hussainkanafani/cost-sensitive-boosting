class ModelRunner():

    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.predicts = []

    def run(self):
        self.model.fit(self.data["trainX"], self.data["trainY"])
        self.predicts = self.model.predict(self.data["testX"])
        return self.predicts
