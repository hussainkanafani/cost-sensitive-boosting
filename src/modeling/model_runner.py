class ModelRunner():

    def __init__(self, model, data):
        self.model = model
        self.x_train, self.y_train, self.x_test, self.y_test = data
        self.predicts = []

    def run(self):
        self.model.fit(self.x_train, self.y_train)
        self.predicts = self.model.predict(self.x_test)
        return self.predicts
