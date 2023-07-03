from sklearn import svm

class EmotionRegressor():

    def __init__(self):
        self.model = svm.SVR(epsilon=0.2)

    def fit(self, features, targets):
        self.model.fit(features, targets)

    def predict(self, input):
        self.model.predict(input)

    def save(self):
        pass